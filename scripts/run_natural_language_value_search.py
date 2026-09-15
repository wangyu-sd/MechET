#!/usr/bin/env python3
"""Product-only executable beam search with a learned state-value critic.

The policy sees only the target, executor-owned current state, and temporary
atom aliases.  The reference precursor is used after search for evaluation and
for selecting successful training trajectories; it is never in a model prompt
or search score.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.assistant_masking import render_chat
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import (
    deterministic_unmapped_state,
    map_unmapped_fragment,
    mapped_atom_numbers,
    merge_mapped_fragments,
)
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS, _decision_row, _prompt
from scripts.build_natural_language_state_value import VALUE_SYSTEM, value_prompt
from scripts.eval_natural_language_event_local import MODEL_REVISION, prediction_call


def read_selected(path: Path, size: int, seed: int) -> list[dict[str, Any]]:
    """Streaming deterministic bottom-k selection."""
    import heapq

    heap: list[tuple[int, int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            digest = hashlib.sha256(f"{seed}:{row['id']}".encode()).digest()
            key = int.from_bytes(digest, "big")
            item = (-key, -line_number, row)
            if len(heap) < size:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda value: (-value[0], -value[1]))]


def visible(state: str) -> str:
    return deterministic_unmapped_state(state).text


def normal_smiles(value: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(value or ""))
    if mol is None:
        raise ValueError("invalid SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@dataclass
class Action:
    name: str
    arguments: dict[str, Any]
    raw: str
    logprob: float
    tokens: int


@dataclass
class Node:
    target: str
    state: str
    next_map: int
    actions: list[dict[str, Any]] = field(default_factory=list)
    visited: set[str] = field(default_factory=set)
    imported: dict[str, int] = field(default_factory=dict)
    logprob: float = 0.0
    tokens: int = 0
    value: float = 0.0
    terminal: bool = False

    @property
    def policy_score(self) -> float:
        return self.logprob / max(self.tokens, 1)

    def score(self, weight: float) -> float:
        return self.policy_score + weight * self.value


class Runtime:
    def __init__(self, args: argparse.Namespace, local_rank: int):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        torch.cuda.set_device(local_rank)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=MODEL_REVISION, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        self.model = PeftModel.from_pretrained(
            base, args.policy_adapter, adapter_name="policy", is_trainable=False
        )
        self.model.load_adapter(args.value_adapter, adapter_name="value", is_trainable=False)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.label_ids = {
            label: self.tokenizer(label, add_special_tokens=False)["input_ids"]
            for label in "ABC"
        }
        if any(len(ids) != 1 for ids in self.label_ids.values()):
            raise RuntimeError(f"value labels must each be one token: {self.label_ids}")

    def proposals(
        self, target: str, state: str, *, candidates: int, max_new_tokens: int
    ) -> list[Action]:
        torch = self.torch
        self.model.set_adapter("policy")
        prompts = [
            render_chat(
                self.tokenizer,
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt(target, state, include_inventory=False)},
                ],
                tools=TOOLS,
                add_generation_prompt=True,
            ),
            render_chat(
                self.tokenizer,
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt(target, state, include_inventory=True)},
                ],
                tools=TOOLS,
                add_generation_prompt=True,
            ),
        ]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        width = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=candidates,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        actions: list[Action] = []
        seen: set[str] = set()
        for index, sequence in enumerate(output.sequences):
            ids = sequence[width:].tolist()
            stop = len(ids)
            if self.tokenizer.eos_token_id in ids:
                stop = ids.index(self.tokenizer.eos_token_id) + 1
            ids = ids[:stop]
            raw = self.tokenizer.decode(ids, skip_special_tokens=False)
            name, arguments, error = prediction_call(raw, self.tokenizer)
            prompt_kind = index // candidates
            if error or (prompt_kind == 0 and name not in {"import_fragments", "finish_trace"}):
                continue
            if prompt_kind == 1 and name != "apply_electron_flow":
                continue
            signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
            if signature in seen:
                continue
            seen.add(signature)
            token_count = max(stop, 1)
            actions.append(
                Action(
                    name=name,
                    arguments=arguments,
                    raw=raw,
                    logprob=float(scores[index, :stop].sum().item()),
                    tokens=token_count,
                )
            )
        return actions

    def values(self, target: str, states: list[str], batch_size: int = 16) -> list[float]:
        torch = self.torch
        self.model.set_adapter("value")
        output_values: list[float] = []
        label_tokens = [self.label_ids[label][0] for label in "ABC"]
        for start in range(0, len(states), batch_size):
            prompts = [
                render_chat(
                    self.tokenizer,
                    [
                        {"role": "system", "content": VALUE_SYSTEM},
                        {"role": "user", "content": value_prompt(target, state)},
                    ],
                    tools=[],
                    add_generation_prompt=True,
                )
                for state in states[start : start + batch_size]
            ]
            encoded = self.tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, label_tokens].float()
                logp = torch.log_softmax(logits, dim=-1)
                useful = torch.logsumexp(logp[:, :2], dim=-1) - logp[:, 2]
            output_values.extend(float(value) for value in useful.cpu().tolist())
        return output_values


def execute(node: Node, action: Action, *, max_imports: int) -> tuple[Node | None, str]:
    state_before = node.state
    state = node.state
    next_map = node.next_map
    imported = dict(node.imported)
    result: dict[str, Any]
    terminal = False
    try:
        if action.name == "import_fragments":
            fragments = action.arguments.get("fragments")
            if not isinstance(fragments, list) or not fragments:
                raise ValueError("IMPORT_SCHEMA_INVALID")
            expanded: list[str] = []
            for item in fragments:
                if not isinstance(item, Mapping):
                    raise ValueError("IMPORT_SCHEMA_INVALID")
                count = int(item.get("count", 0))
                if count < 1:
                    raise ValueError("IMPORT_COUNT_INVALID")
                expanded.extend([normal_smiles(str(item.get("smiles") or ""))] * count)
            if sum(imported.values()) + len(expanded) > max_imports:
                raise ValueError("IMPORT_BUDGET_EXCEEDED")
            mapped: list[str] = []
            for fragment in expanded:
                # Repeated stoichiometric imports are allowed only when explicitly counted.
                mapped_fragment, next_map = map_unmapped_fragment(fragment, first_map=next_map)
                mapped.append(mapped_fragment)
                imported[fragment] = imported.get(fragment, 0) + 1
            state = merge_mapped_fragments(state, mapped)
            result = {"ok": True, "code": "PASS", "current_state": visible(state)}
        elif action.name == "apply_electron_flow":
            moves = compile_event_arguments(state, action.arguments)
            replay = verify_electron_step(state, moves)
            if not replay.get("ok"):
                raise ValueError(str(replay.get("code") or "EXECUTION_FAILED"))
            state = str(replay["state_smiles"])
            if visible(state) in node.visited:
                raise ValueError("STATE_CYCLE")
            result = {"ok": True, "code": "PASS", "current_state": visible(state)}
        elif action.name == "finish_trace":
            if action.arguments:
                raise ValueError("FINISH_ARGUMENTS_NOT_EMPTY")
            terminal = True
            result = {"ok": True, "code": "PASS", "derived_precursor": visible(state)}
        else:
            raise ValueError("UNKNOWN_TOOL")
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"
    record = {
        "state_before": state_before,
        "name": action.name,
        "arguments": action.arguments,
        "result": result,
    }
    return Node(
        target=node.target,
        state=state,
        next_map=next_map,
        actions=node.actions + [record],
        visited=set(node.visited) | {visible(state)},
        imported=imported,
        logprob=node.logprob + action.logprob,
        tokens=node.tokens + action.tokens,
        terminal=terminal,
    ), ""


def rollout(runtime: Runtime, row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    target_mapped = str(row["target_smiles"])
    target = visible(target_mapped)
    expected = visible(str(row.get("full_precursor_state") or row["expected_precursor"]))
    root = Node(
        target=target,
        state=target_mapped,
        next_map=max(mapped_atom_numbers(target_mapped), default=0) + 1,
        visited={target},
    )
    beam = [root]
    terminals: list[Node] = []
    rejected: dict[str, int] = {}
    for depth in range(args.max_decisions):
        children: list[Node] = []
        for node in beam:
            for action in runtime.proposals(
                target, node.state, candidates=args.branching, max_new_tokens=args.max_new_tokens
            ):
                child, error = execute(node, action, max_imports=args.max_imports)
                if child is None:
                    rejected[error] = rejected.get(error, 0) + 1
                elif child.terminal:
                    terminals.append(child)
                else:
                    children.append(child)
        if not children:
            break
        # Collapse graph-equivalent visible states before paying for critic scores.
        unique: dict[str, Node] = {}
        for child in children:
            key = visible(child.state)
            incumbent = unique.get(key)
            if incumbent is None or child.policy_score > incumbent.policy_score:
                unique[key] = child
        children = list(unique.values())
        values = runtime.values(target, [child.state for child in children])
        for child, value in zip(children, values, strict=True):
            child.value = value
        width = args.early_beam if depth < args.early_depth else args.late_beam
        beam = sorted(
            children,
            key=lambda child: child.score(args.value_weight),
            reverse=True,
        )[:width]
    terminals.sort(key=lambda node: node.score(args.value_weight), reverse=True)
    top = terminals[0] if terminals else (beam[0] if beam else root)
    any_exact = any(visible(node.state) == expected for node in terminals)
    top_exact = bool(top.terminal and visible(top.state) == expected)
    successful = next((node for node in terminals if visible(node.state) == expected), None)
    return {
        "id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "target": target,
        "expected_precursor": expected,
        "top_precursor": visible(top.state),
        "top_terminal": top.terminal,
        "top1_exact": top_exact,
        "pass_at_beam": any_exact,
        "n_terminals": len(terminals),
        "top_policy_score": top.policy_score,
        "top_value": top.value,
        "n_actions": len(top.actions),
        "rejected": rejected,
        "top_actions": top.actions,
        "successful_actions": successful.actions if successful is not None else [],
    }


def distill_rows(result: Mapping[str, Any], source: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = list(result.get("successful_actions") or [])
    rows: list[dict[str, Any]] = []
    public_source = dict(
        source,
        target_smiles=str(result["target"]),
        expected_precursor=str(result["expected_precursor"]),
    )
    for index, record in enumerate(actions):
        rows.append(
            _decision_row(
                row=public_source,
                sequence_index=index,
                decision_type=(
                    "import" if record["name"] == "import_fragments" else
                    "finish" if record["name"] == "finish_trace" else "event"
                ),
                mapped_state=str(record["state_before"]),
                name=str(record["name"]),
                arguments=dict(record["arguments"]),
                result=dict(record["result"]),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--policy-adapter", required=True)
    parser.add_argument("--value-adapter", required=True)
    parser.add_argument("--sample-reactions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--branching", type=int, default=4)
    parser.add_argument("--early-beam", type=int, default=4)
    parser.add_argument("--late-beam", type=int, default=2)
    parser.add_argument("--early-depth", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=10)
    parser.add_argument("--max-imports", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--value-weight", type=float, default=0.20)
    parser.add_argument("--write-distill", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    rows = read_selected(args.data, args.sample_reactions, args.seed)
    selected = [row for index, row in enumerate(rows) if index % world == rank]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"results.shard-{rank:02d}-of-{world:02d}.jsonl"
    distill = args.output / f"distill.shard-{rank:02d}-of-{world:02d}.jsonl"
    runtime = Runtime(args, local_rank)
    print(
        f"[meteor-value-search] rank={rank}/{world} reactions={len(selected)} "
        f"beam={args.early_beam}->{args.late_beam} branching={args.branching}",
        flush=True,
    )
    started = time.time()
    with output.open("w", encoding="utf-8") as sink, distill.open("w", encoding="utf-8") as dsink:
        for number, row in enumerate(selected, 1):
            result = rollout(runtime, row, args)
            sink.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            if args.write_distill and result["pass_at_beam"]:
                for item in distill_rows(result, row):
                    dsink.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            sink.flush()
            dsink.flush()
            print(
                json.dumps(
                    {
                        "stage": "value-search",
                        "rank": rank,
                        "done": number,
                        "total": len(selected),
                        "top1_exact": result["top1_exact"],
                        "pass_at_beam": result["pass_at_beam"],
                        "elapsed_s": round(time.time() - started),
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
