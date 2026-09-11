#!/usr/bin/env python3
"""Compare independent rollouts with executor-constrained event search.

The comparison uses one frozen in-place-grounded Qwen adapter and the same
maximum number of response slots and tokens per response for both methods.
Gold endpoints and gold states are evaluation-only and never enter prompts,
tool observations, branch pruning, or ranking.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from agent_model_init import path_sha256
from mechet.agent_inference import parse_tool_calls
from mechet.grounded_event_search import (
    GroundedProposal,
    GroundedSearchNode,
    GroundedTerminal,
    advance_grounded_beam,
    append_rejection_feedback,
    make_root,
)
from mechet.model import resolve_qwen_model_path
from mechet.proof_program import sides_equal


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("id") or row.get("source_id") or "")
    if not value:
        raise ValueError("row has no stable id")
    return value


def frozen_subset(
    rows: Sequence[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=stable_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    selected = ordered[:count]
    if len(selected) != count:
        raise ValueError(f"requested {count} rows, found {len(selected)}")
    return selected


def subset_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(stable_id(row) for row in rows) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def load_model(base_model: str, adapter: Path, *, load_in_4bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    manifest_path = next(
        (
            parent / "adapter_manifest.json"
            for parent in (adapter, *adapter.parents)
            if (parent / "adapter_manifest.json").is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(f"no adapter_manifest.json above {adapter}")
    revision = json.loads(manifest_path.read_text())["base_model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    quantization = None
    if load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=None if Path(base_model).exists() else revision,
        trust_remote_code=True,
        local_files_only=Path(base_model).exists(),
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, tokenizer, revision


@contextmanager
def independent_streams(seeds: Sequence[int], device: Any) -> Iterator[None]:
    """Use one stable multinomial stream per row in a generation microbatch."""

    import torch

    original = torch.multinomial
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]

    def multinomial(input, num_samples, replacement=False, *, out=None, generator=None):
        if (
            generator is None
            and out is None
            and getattr(input, "ndim", 0) == 2
            and int(input.shape[0]) == len(generators)
        ):
            return torch.cat(
                [
                    original(
                        input[index : index + 1],
                        num_samples,
                        replacement,
                        generator=generators[index],
                    )
                    for index in range(len(generators))
                ],
                dim=0,
            )
        return original(
            input,
            num_samples,
            replacement,
            out=out,
            generator=generator,
        )

    torch.multinomial = multinomial
    try:
        yield
    finally:
        torch.multinomial = original


class TransformersProposalGenerator:
    def __init__(self, model: Any, tokenizer: Any, *, microbatch_size: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.microbatch_size = microbatch_size

    def _generate_batch(
        self,
        requests: Sequence[tuple[GroundedSearchNode, int]],
        tools: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[tuple[GroundedProposal | None, dict[str, Any]]]:
        import torch

        tokenizer = self.tokenizer
        previous_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            prompts = [
                tokenizer.apply_chat_template(
                    node.transcript(),
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for node, _ in requests
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = previous_padding
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in encoded.items()}
        input_length = int(inputs["input_ids"].shape[1])
        seeds = [seed for _, seed in requests]
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.inference_mode(), independent_streams(seeds, device):
            output = self.model.generate(**inputs, **kwargs)
        generated = output.sequences[:, input_length:]
        eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
        configured = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(configured, int):
            eos_ids.add(configured)
        elif configured is not None:
            eos_ids.update(int(item) for item in configured)

        records: list[tuple[GroundedProposal | None, dict[str, Any]]] = []
        for row_index, ((_, seed), tokens) in enumerate(zip(requests, generated, strict=True)):
            values = [int(item) for item in tokens.tolist()]
            stop = next(
                (index + 1 for index, token in enumerate(values) if token in eos_ids),
                len(values),
            )
            values = values[:stop]
            logprob_sum = 0.0
            for offset, token in enumerate(values):
                if offset >= len(output.scores):
                    break
                score = torch.log_softmax(output.scores[offset][row_index].float(), dim=-1)
                logprob_sum += float(score[token].item())
            raw = tokenizer.decode(values, skip_special_tokens=False)
            attention = inputs.get("attention_mask")
            prefix = (
                inputs["input_ids"][row_index][attention[row_index].bool()]
                if attention is not None
                else inputs["input_ids"][row_index]
            )
            diagnostic = {
                "seed": seed,
                "raw": raw,
                "generated_tokens": len(values),
                "logprob_sum": logprob_sum,
            }
            try:
                calls = parse_tool_calls(raw, tokenizer=tokenizer, prefix=prefix)
                if len(calls) != 1:
                    raise ValueError(f"expected one tool call, parsed {len(calls)}")
                call = calls[0]
                proposal = GroundedProposal(
                    name=call.name,
                    arguments=call.arguments,
                    raw_response=raw,
                    logprob_sum=logprob_sum,
                    token_count=max(len(values), 1),
                    seed=seed,
                    call_id=call.call_id,
                )
                diagnostic["tool"] = call.name
                records.append((proposal, diagnostic))
            except Exception as exc:
                diagnostic.update({"code": "PARSE_FAILED", "message": str(exc)})
                records.append((None, diagnostic))
        return records

    def generate(
        self,
        requests: Sequence[tuple[GroundedSearchNode, int]],
        tools: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[tuple[GroundedProposal | None, dict[str, Any]]]:
        output: list[tuple[GroundedProposal | None, dict[str, Any]]] = []
        for start in range(0, len(requests), self.microbatch_size):
            batch = requests[start : start + self.microbatch_size]
            try:
                output.extend(
                    self._generate_batch(
                        batch,
                        tools,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
            except torch_out_of_memory_error():
                if len(batch) == 1:
                    raise
                old = self.microbatch_size
                self.microbatch_size = max(1, len(batch) // 2)
                output.extend(
                    self.generate(
                        batch,
                        tools,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
                self.microbatch_size = old
        return output


def torch_out_of_memory_error():
    try:
        import torch

        return torch.cuda.OutOfMemoryError
    except (ImportError, AttributeError):
        return RuntimeError


def candidate_seed(base: int, identifier: str, method: str, index: int, turn: int) -> int:
    payload = f"{base}\0{identifier}\0{method}\0{index}\0{turn}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def terminal_record(item: GroundedTerminal) -> dict[str, Any]:
    return {
        "endpoint": item.result.get("derived_precursor"),
        "endpoint_exact": bool(item.result.get("endpoint_exact")),
        "formal_execute": bool(item.result.get("formal_execute")),
        "events": len(item.node.transitions),
        "generated_tokens": item.node.token_count,
        "normalized_logprob": item.node.normalized_logprob,
        "trace_digest": item.result.get("trace_digest"),
    }


def prefix_rank(
    selected: Sequence[GroundedSearchNode], gold_states: Sequence[str], depth: int
) -> int | None:
    if depth < 1 or depth > len(gold_states):
        return None
    gold = gold_states[depth - 1]
    for rank, node in enumerate(selected, start=1):
        if sides_equal(node.current_mapped_state, gold, ignore_maps=True):
            return rank
    return None


def run_search(
    row: Mapping[str, Any],
    gold_states: Sequence[str],
    generator: TransformersProposalGenerator,
    *,
    beam_width: int,
    branch_factor: int,
    max_depth: int,
    max_responses: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    root = make_root(row)
    tools = list(row.get("tools") or [])
    active = [root]
    terminals: list[GroundedTerminal] = []
    diagnostics: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    diversity: list[int] = []
    prefix_ranks: list[int | None] = []
    generated_tokens = responses = 0
    started = time.monotonic()
    identifier = stable_id(row)

    for depth in range(max_depth):
        requests: list[tuple[GroundedSearchNode, int]] = []
        owners: list[int] = []
        for owner, node in enumerate(active):
            for branch in range(branch_factor):
                if len(requests) + responses >= max_responses:
                    break
                requests.append(
                    (
                        node,
                        candidate_seed(seed, identifier, "search", owner * branch_factor + branch, depth),
                    )
                )
                owners.append(owner)
        if not requests:
            break
        generated = generator.generate(
            requests,
            tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        responses += len(generated)
        proposals: dict[int, list[GroundedProposal]] = defaultdict(list)
        for owner, (proposal, detail) in zip(owners, generated, strict=True):
            generated_tokens += int(detail["generated_tokens"])
            if proposal is None:
                counts["PARSE_FAILED"] += 1
                diagnostics.append({"depth": depth, **detail})
            else:
                proposals[owner].append(proposal)
        expanded = [
            (node, proposals.get(index, [])) for index, node in enumerate(active)
        ]
        step = advance_grounded_beam(
            expanded,
            beam_width=beam_width,
            expected_precursor=str(row.get("expected_precursor") or ""),
        )
        terminals.extend(step.terminals)
        for item in step.rejected:
            counts[item.code] += 1
        counts["DUPLICATE_SUCCESSOR"] += len(step.duplicate_pruned)
        counts["BEAM_CAPACITY"] += len(step.beam_pruned)
        active = list(step.selected)
        diversity.append(len({node.visited_visible_states[-1] for node in active}))
        prefix_ranks.append(prefix_rank(active, gold_states, depth + 1))
        if not active:
            break

    ranked = sorted(terminals, key=lambda item: item.node.normalized_logprob, reverse=True)
    return {
        "method": "executor_constrained_event_search",
        "responses": responses,
        "allocated_token_ceiling": responses * max_new_tokens,
        "generated_tokens": generated_tokens,
        "latency_seconds": time.monotonic() - started,
        "terminal_count": len(ranked),
        "explicit_finish": bool(ranked),
        "endpoint_pass_at_1": bool(ranked and ranked[0].result.get("endpoint_exact")),
        "endpoint_pass_oracle_at_budget": any(item.result.get("endpoint_exact") for item in ranked),
        "terminals": [terminal_record(item) for item in ranked],
        "prune_counts": dict(counts),
        "pruned_before_next_model_call": sum(
            value for key, value in counts.items() if key != "BEAM_CAPACITY"
        ),
        "surviving_state_diversity_by_depth": diversity,
        "gold_prefix_rank_by_depth": prefix_ranks,
        "parse_failures": [item for item in diagnostics if item.get("code") == "PARSE_FAILED"][:8],
    }


def run_independent(
    row: Mapping[str, Any],
    generator: TransformersProposalGenerator,
    *,
    candidates: int,
    max_depth: int,
    max_responses: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    tools = list(row.get("tools") or [])
    active: dict[int, GroundedSearchNode] = {
        index: make_root(row) for index in range(candidates)
    }
    terminals: list[GroundedTerminal] = []
    counts: Counter[str] = Counter()
    generated_tokens = responses = 0
    started = time.monotonic()
    identifier = stable_id(row)

    for turn in range(max_depth):
        indices = list(active)
        remaining = max_responses - responses
        indices = indices[:remaining]
        if not indices:
            break
        requests = [
            (
                active[index],
                candidate_seed(seed, identifier, "independent", index, turn),
            )
            for index in indices
        ]
        generated = generator.generate(
            requests,
            tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        responses += len(generated)
        for index, (proposal, detail) in zip(indices, generated, strict=True):
            generated_tokens += int(detail["generated_tokens"])
            if proposal is None:
                counts["PARSE_FAILED"] += 1
                active.pop(index, None)
                continue
            step = advance_grounded_beam(
                [(active[index], [proposal])],
                beam_width=1,
                expected_precursor=str(row.get("expected_precursor") or ""),
            )
            if step.terminals:
                terminals.extend(step.terminals)
                active.pop(index, None)
            elif step.selected:
                active[index] = step.selected[0]
            elif step.rejected:
                rejection = step.rejected[0]
                counts[rejection.code] += 1
                active[index] = append_rejection_feedback(
                    active[index], proposal, rejection
                )
            else:
                active.pop(index, None)

    ranked = sorted(terminals, key=lambda item: item.node.normalized_logprob, reverse=True)
    return {
        "method": "independent_stochastic_rollouts",
        "responses": responses,
        "allocated_token_ceiling": responses * max_new_tokens,
        "generated_tokens": generated_tokens,
        "latency_seconds": time.monotonic() - started,
        "terminal_count": len(ranked),
        "explicit_finish": bool(ranked),
        "endpoint_pass_at_1": bool(ranked and ranked[0].result.get("endpoint_exact")),
        "endpoint_pass_oracle_at_budget": any(item.result.get("endpoint_exact") for item in ranked),
        "terminals": [terminal_record(item) for item in ranked],
        "failure_counts": dict(counts),
    }


def aggregate(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    values = [dict(item[method]) for item in records]
    n = len(values)
    return {
        "n": n,
        "endpoint_pass_at_1": sum(item["endpoint_pass_at_1"] for item in values) / n,
        "endpoint_pass_oracle_at_budget": sum(item["endpoint_pass_oracle_at_budget"] for item in values) / n,
        "explicit_finish_rate": sum(item["explicit_finish"] for item in values) / n,
        "mean_generated_tokens": sum(item["generated_tokens"] for item in values) / n,
        "mean_responses": sum(item["responses"] for item in values) / n,
        "mean_latency_seconds": sum(item["latency_seconds"] for item in values) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gold-source-data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--independent-candidates", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-responses", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    if args.limit < 1 or args.max_responses < 1 or args.max_new_tokens < 1:
        raise ValueError("limits and token budgets must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("require shard_count >= 1 and 0 <= shard_index < shard_count")
    global_rows = frozen_subset(
        read_jsonl(args.data), count=args.limit, seed=args.selection_seed
    )
    global_subset_hash = subset_sha256(global_rows)
    rows = [
        row
        for index, row in enumerate(global_rows)
        if index % args.shard_count == args.shard_index
    ]
    if not rows:
        raise ValueError("selected shard is empty")
    gold_by_id = {stable_id(row): row for row in read_jsonl(args.gold_source_data)}
    missing = [stable_id(row) for row in rows if stable_id(row) not in gold_by_id]
    if missing:
        raise ValueError(f"selected IDs absent from gold source: {missing[:8]}")
    base = args.base_model or resolve_qwen_model_path() or "Qwen/Qwen3-8B"
    model, tokenizer, revision = load_model(base, args.adapter, load_in_4bit=not args.no_4bit)
    generator = TransformersProposalGenerator(
        model, tokenizer, microbatch_size=args.microbatch_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            gold = gold_by_id[stable_id(row)]
            runtime_row = dict(row)
            runtime_row["target_smiles"] = str(gold.get("target_smiles") or "")
            runtime_row["expected_precursor"] = str(
                gold.get("full_precursor_state")
                or gold.get("expected_precursor")
                or ""
            )
            if not runtime_row["target_smiles"] or not runtime_row["expected_precursor"]:
                raise ValueError(f"private executor state missing for {stable_id(row)}")
            steps = list((((gold.get("metadata") or {}).get("trace_plan") or {}).get("steps") or []))
            gold_states = [str(item.get("state_after") or "") for item in steps]
            record = {
                "id": stable_id(row),
                "independent": run_independent(
                    runtime_row,
                    generator,
                    candidates=args.independent_candidates,
                    max_depth=args.max_depth,
                    max_responses=args.max_responses,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                ),
                "search": run_search(
                    runtime_row,
                    gold_states,
                    generator,
                    beam_width=args.beam_width,
                    branch_factor=args.branch_factor,
                    max_depth=args.max_depth,
                    max_responses=args.max_responses,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                ),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[constrained-search] {index}/{len(rows)} id={record['id']} "
                f"independent_exact={int(record['independent']['endpoint_pass_oracle_at_budget'])} "
                f"search_exact={int(record['search']['endpoint_pass_oracle_at_budget'])}",
                flush=True,
            )

    summary = {
        "artifact_type": "grounded_executor_constrained_search_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.adapter),
        "adapter_model_sha256": path_sha256(args.adapter / "adapter_model.safetensors"),
        "base_model": base,
        "base_model_revision": revision,
        "data": str(args.data),
        "data_sha256": path_sha256(args.data),
        "gold_source_data": str(args.gold_source_data),
        "gold_source_data_sha256": path_sha256(args.gold_source_data),
        "global_subset_id_sha256": global_subset_hash,
        "global_subset_size": len(global_rows),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "subset_id_sha256": subset_sha256(rows),
        "subset_ids": [stable_id(row) for row in rows],
        "product_only_model_input": True,
        "gold_used_for_generation_pruning_or_ranking": False,
        "budget_contract": {
            "same_max_responses_per_target": args.max_responses,
            "same_max_new_tokens_per_response": args.max_new_tokens,
            "same_allocated_token_ceiling_per_target": args.max_responses * args.max_new_tokens,
            "independent_candidates": args.independent_candidates,
            "beam_width": args.beam_width,
            "branch_factor": args.branch_factor,
            "max_depth": args.max_depth,
        },
        "independent": aggregate(records, "independent"),
        "search": aggregate(records, "search"),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
