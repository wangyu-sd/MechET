#!/usr/bin/env python3
"""Gold-state local evaluation for natural-language electron-event SFT.

The model receives exactly the product/current-state prompt used by SFT and
generates one next tool decision.  Private atom maps are reconstructed only in
the evaluator so predicted natural-language aliases can be compiled and
strictly replayed.  This is an F-oracle local diagnostic, not a product-only
closed-loop endpoint result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from mechet.a7_rescue import canonical_event, mechanism_length_stratum, stratified_sample
from mechet.agent_inference import parse_tool_calls
from mechet.assistant_masking import render_chat
from mechet.forward_expert import ElectronMove, verify_electron_step
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    extract_import_fragments,
    mapped_atom_numbers,
    mapped_state_signature,
    retain_mapped_components,
    schedule_imports,
)
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.build_natural_language_event_sft import convert_row


MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distributed_coordinates() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not 0 <= rank < world:
        raise ValueError("invalid rank/world size")
    return rank, world, local_rank


def _private_states(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct executor-private state aligned with every public decision."""

    plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
    steps = [dict(item) for item in plan.get("steps") or []]
    fragments = extract_import_fragments(row)
    scheduled = schedule_imports(fragments, steps)
    present_maps = set(mapped_atom_numbers(str(row["target_smiles"])))
    current = retain_mapped_components(str(steps[0]["state_before"]), present_maps)
    output: list[dict[str, Any]] = []
    event_depth = 0
    for step, imports in zip(steps, scheduled, strict=True):
        if imports:
            output.append(
                {
                    "decision_type": "import",
                    "private_state": current,
                    "reference_successor": "",
                    "event_depth": event_depth,
                }
            )
            current_with_imports = append_mapped_fragments_verbatim(current, imports)
            for fragment in imports:
                present_maps.update(mapped_atom_numbers(fragment))
        else:
            current_with_imports = current
        successor = retain_mapped_components(str(step["state_after"]), present_maps)
        output.append(
            {
                "decision_type": "event",
                "private_state": current_with_imports,
                "reference_successor": successor,
                "event_depth": event_depth + 1,
            }
        )
        current = successor
        event_depth += 1
    output.append(
        {
            "decision_type": "finish",
            "private_state": current,
            "reference_successor": current,
            "event_depth": event_depth,
        }
    )
    return output


def collect_tasks(
    rows: Iterable[Mapping[str, Any]], *, sample_reactions: int, seed: int
) -> tuple[list[dict[str, Any]], list[str]]:
    selected = stratified_sample(rows, size=sample_reactions, seed=seed)
    tasks: list[dict[str, Any]] = []
    reaction_ids: list[str] = []
    for source in selected:
        public_rows = convert_row(source)
        private_rows = _private_states(source)
        if len(public_rows) != len(private_rows):
            raise ValueError(f"{source['id']}: public/private decision count mismatch")
        stratum = mechanism_length_stratum(
            len((source["metadata"]["trace_plan"] or {})["steps"])
        )
        reaction_id = str(source["source_id"])
        reaction_ids.append(reaction_id)
        for public, private in zip(public_rows, private_rows, strict=True):
            decision_type = str(public["metadata"]["decision_type"])
            if decision_type != private["decision_type"]:
                raise ValueError(f"{public['id']}: decision alignment changed")
            gold_call = public["messages"][2]["tool_calls"][0]["function"]
            tasks.append(
                {
                    "key": str(public["id"]),
                    "reaction_id": reaction_id,
                    "decision_type": decision_type,
                    "event_depth": int(private["event_depth"]),
                    "stratum": stratum,
                    "messages": public["messages"][:2],
                    "tools": public["tools"],
                    "gold_name": str(gold_call["name"]),
                    "gold_arguments": dict(gold_call["arguments"]),
                    "private_state": str(private["private_state"]),
                    "reference_successor": str(private["reference_successor"]),
                }
            )
    if len({task["key"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate evaluation decision key")
    return tasks, reaction_ids


def prediction_call(text: str, tokenizer: Any) -> tuple[str, dict[str, Any], str]:
    try:
        calls = parse_tool_calls(text, tokenizer=tokenizer)
    except Exception as exc:
        return "", {}, f"PARSE_ERROR:{type(exc).__name__}:{exc}"
    if len(calls) != 1:
        return "", {}, f"EXPECTED_ONE_TOOL_CALL:observed={len(calls)}"
    return str(calls[0].name), dict(calls[0].arguments), ""


def _normal_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ValueError(f"invalid imported SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _import_counter(
    arguments: Mapping[str, Any], *, include_purpose: bool
) -> Counter[tuple[str, str]]:
    output: Counter[tuple[str, str]] = Counter()
    fragments = arguments.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("fragments must be a nonempty list")
    for item in fragments:
        if not isinstance(item, Mapping):
            raise ValueError("fragment entry must be an object")
        count = int(item.get("count", 0))
        if count <= 0:
            raise ValueError("fragment count must be positive")
        purpose = str(item.get("purpose") or "") if include_purpose else ""
        output[(_normal_smiles(str(item.get("smiles") or "")), purpose)] += count
    return output


def _site_signature(moves: Sequence[Mapping[str, Any]], field: str) -> list[tuple[str, tuple[int, ...]]]:
    output = []
    for raw in moves:
        if raw.get("mode") == "BE_DELTA":
            continue
        move = ElectronMove.parse(raw)
        container = move.source if field == "source" else move.sink
        output.append((container.kind, tuple(container.atoms)))
    return sorted(output)


def score_prediction(
    task: Mapping[str, Any], *, predicted_name: str, predicted_arguments: Mapping[str, Any]
) -> dict[str, Any]:
    dtype = str(task["decision_type"])
    gold_name = str(task["gold_name"])
    gold_arguments = dict(task["gold_arguments"])
    correct_tool = predicted_name == gold_name
    base = {
        "correct_tool": correct_tool,
        "decision_exact": False,
        "argument_compile": False,
        "formal_execute": False,
        "event_exact": False,
        "source_site_exact": False,
        "destination_site_exact": False,
        "successor_map_exact": False,
        "successor_chemical_exact": False,
        "import_fragment_exact": False,
        "import_schedule_exact": False,
        "finish_exact": False,
        "execution_error": "",
    }
    if not correct_tool:
        base["execution_error"] = f"WRONG_TOOL:{predicted_name or 'NONE'}"
        return base
    try:
        if dtype == "finish":
            exact = not predicted_arguments
            base.update(decision_exact=exact, argument_compile=exact, finish_exact=exact)
            return base
        if dtype == "import":
            predicted_fragments = _import_counter(predicted_arguments, include_purpose=False)
            gold_fragments = _import_counter(gold_arguments, include_purpose=False)
            predicted_schedule = _import_counter(predicted_arguments, include_purpose=True)
            gold_schedule = _import_counter(gold_arguments, include_purpose=True)
            fragment_exact = predicted_fragments == gold_fragments
            schedule_exact = predicted_schedule == gold_schedule
            base.update(
                decision_exact=schedule_exact,
                argument_compile=True,
                import_fragment_exact=fragment_exact,
                import_schedule_exact=schedule_exact,
            )
            return base
        if dtype != "event":
            raise ValueError(f"unsupported decision type: {dtype}")
        predicted_moves = compile_event_arguments(
            str(task["private_state"]), predicted_arguments
        )
        gold_moves = compile_event_arguments(str(task["private_state"]), gold_arguments)
        execution = verify_electron_step(str(task["private_state"]), predicted_moves)
        formal = bool(execution.get("ok"))
        reference = str(task["reference_successor"])
        predicted_state = str(execution.get("state_smiles") or "")
        map_exact = bool(
            formal
            and mapped_state_signature(predicted_state)
            == mapped_state_signature(reference)
        )
        chemical_exact = bool(
            formal
            and deterministic_unmapped_state(predicted_state).text
            == deterministic_unmapped_state(reference).text
        )
        event_exact = canonical_event(predicted_moves) == canonical_event(gold_moves)
        base.update(
            decision_exact=event_exact,
            argument_compile=True,
            formal_execute=formal,
            event_exact=event_exact,
            source_site_exact=_site_signature(predicted_moves, "source")
            == _site_signature(gold_moves, "source"),
            destination_site_exact=_site_signature(predicted_moves, "destination")
            == _site_signature(gold_moves, "destination"),
            successor_map_exact=map_exact,
            successor_chemical_exact=chemical_exact,
            execution_error=(
                ""
                if formal
                else str(execution.get("code") or execution.get("message") or "EXECUTION_FAILED")
            ),
        )
        return base
    except Exception as exc:
        base["execution_error"] = f"{type(exc).__name__}:{exc}"
        return base


def _trim_completion(ids: list[int], tokenizer: Any) -> list[int]:
    stop = {value for value in (tokenizer.eos_token_id,) if value is not None}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop.add(im_end)
    for index, token_id in enumerate(ids):
        if token_id in stop:
            return ids[: index + 1]
    return ids


def _batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rank, world, local_rank = distributed_coordinates()
    torch.cuda.set_device(local_rank)
    rows = read_jsonl(args.data)
    tasks, reaction_ids = collect_tasks(
        rows, sample_reactions=args.sample_reactions, seed=args.seed
    )
    selected = [task for index, task in enumerate(tasks) if index % world == rank]
    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.output / f"decisions.shard-{rank:02d}-of-{world:02d}.jsonl"
    completed = {row["key"] for row in read_jsonl(shard)} if shard.exists() else set()
    selected = [task for task in selected if task["key"] not in completed]
    if rank == 0:
        (args.output / "selection.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "sample_reactions": args.sample_reactions,
                    "reaction_ids": reaction_ids,
                    "reaction_ids_sha256": hashlib.sha256(
                        "\n".join(reaction_ids).encode()
                    ).hexdigest(),
                    "planned_decisions": len(tasks),
                },
                indent=2,
            )
            + "\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=MODEL_REVISION, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.float16
    try:
        bnb_version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("bitsandbytes is required for the matched NF4 evaluation") from exc
    print(
        f"[meteor-nl-event-eval] rank={rank}/{world} gpu={torch.cuda.get_device_name(local_rank)} "
        f"tasks={len(selected)} bnb={bnb_version}",
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": local_rank},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    device = next(model.parameters()).device
    with shard.open("a", encoding="utf-8") as sink:
        done = 0
        for batch in _batches(selected, args.batch_size):
            started = time.time()
            prompts = [
                render_chat(
                    tokenizer,
                    list(task["messages"]),
                    tools=list(task["tools"]),
                    add_generation_prompt=True,
                )
                for task in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            max_input = int(encoded["input_ids"].shape[1])
            if max_input + args.max_new_tokens > args.max_context:
                raise ValueError(
                    f"context budget exceeded: input={max_input} new={args.max_new_tokens}"
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.time() - started
            input_width = int(encoded["input_ids"].shape[1])
            for task, sequence in zip(batch, output, strict=True):
                completion_ids = _trim_completion(
                    [int(value) for value in sequence[input_width:].tolist()], tokenizer
                )
                generated = tokenizer.decode(completion_ids, skip_special_tokens=False)
                name, arguments, generation_error = prediction_call(generated, tokenizer)
                metrics = score_prediction(
                    task,
                    predicted_name=name,
                    predicted_arguments=arguments,
                )
                record = {
                    "key": task["key"],
                    "reaction_id": task["reaction_id"],
                    "decision_type": task["decision_type"],
                    "event_depth": task["event_depth"],
                    "stratum": task["stratum"],
                    "gold_name": task["gold_name"],
                    "predicted_name": name,
                    "predicted_arguments": arguments,
                    "generated_text": generated,
                    "generation_error": generation_error,
                    "generated_tokens": len(completion_ids),
                    "batch_seconds": elapsed,
                    **metrics,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                done += 1
            counts = Counter(task["decision_type"] for task in batch)
            print(
                f"[meteor-nl-event-eval] rank={rank} done={done}/{len(selected)} "
                f"types={dict(counts)} seconds={elapsed:.2f}",
                flush=True,
            )
    return 0


METRICS = (
    "correct_tool",
    "decision_exact",
    "argument_compile",
    "formal_execute",
    "event_exact",
    "source_site_exact",
    "destination_site_exact",
    "successor_map_exact",
    "successor_chemical_exact",
    "import_fragment_exact",
    "import_schedule_exact",
    "finish_exact",
)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    for name in METRICS:
        count = sum(bool(row.get(name)) for row in rows)
        result[name] = count
        result[f"{name}_rate"] = count / max(len(rows), 1)
    result["mean_generated_tokens"] = sum(
        int(row.get("generated_tokens", 0)) for row in rows
    ) / max(len(rows), 1)
    return result


def aggregate(args: argparse.Namespace) -> int:
    source = read_jsonl(args.data)
    tasks, reaction_ids = collect_tasks(
        source, sample_reactions=args.sample_reactions, seed=args.seed
    )
    expected = {task["key"] for task in tasks}
    rows: list[dict[str, Any]] = []
    for path in sorted(args.output.glob("decisions.shard-*-of-*.jsonl")):
        rows.extend(read_jsonl(path))
    observed = [str(row["key"]) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate prediction keys")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    by_type = {
        name: _summary([row for row in rows if row["decision_type"] == name])
        for name in ("import", "event", "finish")
    }
    report = {
        "artifact_type": "natural_language_event_gold_state_local_k1_v1",
        "claim_boundary": (
            "Fixed validation F-oracle one-decision diagnostic; not product-only "
            "closed-loop rollout and not test endpoint accuracy."
        ),
        "seed": args.seed,
        "planned_reactions": args.sample_reactions,
        "reaction_ids_sha256": hashlib.sha256("\n".join(reaction_ids).encode()).hexdigest(),
        "planned_decisions": len(expected),
        "completed_decisions": len(rows),
        "missing_decisions": len(missing),
        "extra_decisions": len(extra),
        "complete": not missing and not extra,
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "adapter": str(args.adapter),
        "adapter_model_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "overall": _summary(rows),
        "by_type": by_type,
        "event_by_stratum": {
            name: _summary(
                [
                    row
                    for row in rows
                    if row["decision_type"] == "event" and row["stratum"] == name
                ]
            )
            for name in ("short", "medium", "long")
        },
        "generation_errors": dict(
            Counter(row["generation_error"] for row in rows if row["generation_error"])
        ),
        "execution_errors": dict(
            Counter(row["execution_error"] for row in rows if row["execution_error"])
        ),
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sample-reactions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    return run(args) if args.command == "run" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
