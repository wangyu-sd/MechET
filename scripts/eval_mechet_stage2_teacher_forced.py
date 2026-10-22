#!/usr/bin/env python3
"""Gold-state, next-tool-action accuracy for a frozen MechET Stage-II policy.

The policy receives the product, an executor-authoritative reference state,
and the accepted reference history *preceding* this decision. No target action,
future state, reference precursor, or ground-truth horizon enters the prompt.

This is an oracle-state diagnostic, NOT autonomous product-start Top-1.
The same runner supports FlowER and the current mech-USPTO-31k trace view.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rdkit import Chem
from mechet.a7_rescue import mechanism_length_stratum, stratified_sample
from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import deterministic_unmapped_state
from mechet.natural_language_electron_flow import build_inventory, compile_event_arguments
from scripts.build_natural_language_event_sft import convert_row
from scripts.build_natural_language_history_sft import transform_rows
from scripts.eval_natural_language_event_local import (
    _import_counter, _trim_completion, prediction_call, score_prediction,
)

DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TYPES = ("import", "event", "finish")
METRICS = (
    "correct_tool", "decision_exact", "argument_compile", "formal_execute",
    "event_exact", "source_site_exact", "destination_site_exact",
    "successor_map_exact", "successor_chemical_exact",
    "import_fragment_exact", "import_schedule_exact", "finish_exact",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for part in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def private_state_from_prompt(public_prompt: str) -> str:
    """Reconstruct TEMPORARY alias maps from the actually displayed molecule.

    Each prompt independently numbers atoms A01,... in SMILES input order.
    These maps are local evaluator addresses, NOT reference/global atom maps.
    No reference successor or private reference map is used in reconstruction.
    """
    match = re.search(r"^ANNOTATED CURRENT STATE: ([^\n]+)$", public_prompt, re.M)
    if match is None:
        raise ValueError("missing annotated current state in model-visible prompt")
    annotated = match.group(1).strip()
    ids = [int(s) for s in re.findall(r"<A(\d+)>", annotated)]
    visible = re.sub(r"<A\d+>", "", annotated)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(visible, params)
    if mol is None or len(ids) != mol.GetNumAtoms() or len(ids) != len(set(ids)):
        raise ValueError("annotated-state atom/alias reconstruction mismatch")
    for atom, map_id in zip(mol.GetAtoms(), ids, strict=True):
        atom.SetAtomMapNum(map_id)
    mapped = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    reproduced = build_inventory(mapped).prompt.split("ANNOTATED CURRENT STATE: ", 1)[-1]
    if reproduced != annotated:
        raise ValueError("round-trip annotated-state/alias mismatch")
    return mapped


def _gold_call(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    calls = row["messages"][2].get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError("reference decision has no unique gold tool call")
    function = calls[0]["function"]
    return str(function["name"]), dict(function["arguments"])


def prepare_task_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build both original and Stage-II prompts, replaying all gold FLOW labels."""
    baseline = convert_row(source)
    history = list(transform_rows(baseline))
    if len(baseline) != len(history):
        raise ValueError("history/decision count mismatch")
    n_events = len(source["metadata"]["trace_plan"]["steps"])
    stratum = mechanism_length_stratum(n_events)
    tasks = []
    event_depth = 0
    for public, stage2 in zip(baseline, history, strict=True):
        dtype = str(public["metadata"]["decision_type"])
        gold_name, gold_args = _gold_call(public)
        reference_result = json.loads(public["messages"][3]["content"])
        private_state = private_state_from_prompt(public["messages"][1]["content"])
        gold_be_delta = bool(gold_args.get("bond_order_changes") or gold_args.get("charge_changes"))
        n_arrows = len(gold_args.get("electron_flow") or [])
        if dtype == "event":
            gold_moves = compile_event_arguments(private_state, gold_args)
            execution = verify_electron_step(private_state, gold_moves)
            if not execution.get("ok"):
                raise ValueError(
                    f"{public['id']}: canonical gold FLOW did not execute: {execution}"
                )
            actual = deterministic_unmapped_state(str(execution["state_smiles"])).text
            if actual != reference_result.get("current_state"):
                raise ValueError(
                    f"{public['id']}: gold FLOW successor differs from recorded tool result"
                )
            event_depth += 1
        if dtype not in TYPES:
            raise ValueError(f"unknown decision type {dtype}")
        # Only these two messages and tool schemas are sent to the actor.
        model_messages = stage2["messages"][:2]
        visible_json = json.dumps(model_messages, ensure_ascii=False)
        if any(key in visible_json for key in (
            '"expected_precursor"', '"reference_successor"', '"remaining_events"'
        )):
            raise ValueError("private target or horizon appears in actor prompt")
        tasks.append({
            "key": stage2["id"],
            "reaction_id": str(source["source_id"]),
            "decision_type": dtype,
            "event_depth": event_depth,
            "stratum": stratum,
            "messages": model_messages,
            "tools": stage2["tools"],
            "gold_name": gold_name,
            "gold_arguments": gold_args,
            "private_state": private_state,
            "reference_successor": (
                str(execution["state_smiles"]) if dtype == "event" else ""
            ),
            "gold_be_delta": gold_be_delta if dtype == "event" else False,
            "gold_flow_arrows": n_arrows if dtype == "event" else 0,
            "source_data": str(source.get("metadata", {}).get("source_dataset") or ""),
        })
    if sum(task["decision_type"] == "event" for task in tasks) != n_events:
        raise ValueError("source FLOW count changed in conversion")
    return tasks


def prepare(args: argparse.Namespace) -> int:
    sources = read_jsonl(args.data)
    if len(sources) != args.expected_source_rows:
        raise ValueError(
            f"source denominator {len(sources)} != {args.expected_source_rows}"
        )
    source_names = {
        str((row.get("metadata") or {}).get("source_dataset") or "")
        for row in sources
    }
    if source_names != {args.corpus}:
        raise ValueError(f"source corpus mismatch: {source_names} vs {args.corpus}")
    for row in sources:
        split = str((row.get("metadata") or {}).get("source_split") or "").lower()
        if split in {"test", "train"}:
            raise ValueError(f"TRAIN/TEST row refused in validation: {row.get('id')}")
    selected = (
        stratified_sample(sources, size=args.sample_reactions, seed=args.seed)
        if args.sample_reactions else sorted(sources, key=lambda item: str(item["id"]))
    )
    tasks: list[dict[str, Any]] = []
    for source in selected:
        try:
            tasks.extend(prepare_task_rows(source))
        except Exception as exc:
            raise ValueError(f"failed full gold parity for {source['id']}: {exc}") from exc
    if len({task["key"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate decision identifiers")
    args.output.mkdir(parents=True, exist_ok=False)
    task_path = args.output / "tasks.jsonl"
    with task_path.open("w", encoding="utf-8") as writer:
        for task in tasks:
            writer.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
    ids = [str(row["source_id"]) for row in selected]
    manifest = {
        "schema": "mechet_stage2_gold_state_next_action_v1",
        "corpus": args.corpus,
        "source": str(args.data),
        "source_sha256": sha256(args.data),
        "source_rows": len(sources),
        "validation_reactions": len(selected),
        "validation_decisions": len(tasks),
        "reaction_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "tasks_sha256": sha256(task_path),
        "seed": args.seed,
        "same_cohort_for_all_tools": True,
        "stage2_compressed_gold_accepted_history": True,
        "gold_executor_parity": "passed_all_selected_FLOW",
        "model_input": "product + gold current state + gold accepted prefix; no future",
        "claim_boundary": "oracle-state next-action evaluation, not E2E retrosynthesis",
    }
    (args.output / "selection.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for begin in range(0, len(rows), size):
        yield rows[begin:begin + size]


def run(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = read_jsonl(args.input / "tasks.jsonl")
    meta = json.loads((args.input / "selection.json").read_text())
    if sha256(args.input / "tasks.jsonl") != meta["tasks_sha256"]:
        raise ValueError("prepared tasks SHA changed")
    weight = args.adapter / "adapter_model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    weight_sha = sha256(weight)
    if args.expected_adapter_sha and weight_sha != args.expected_adapter_sha:
        raise ValueError("frozen Stage-II adapter SHA mismatch")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for frozen Qwen Stage-II evaluation")
    torch.cuda.set_device(local_rank)
    selected = [task for i, task in enumerate(tasks) if i % world == rank]
    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.output / f"decisions.shard-{rank:02d}-of-{world:02d}.jsonl"
    previous = read_jsonl(shard) if shard.exists() else []
    existing = {row["key"] for row in previous}
    if len(existing) != len(previous):
        raise ValueError("duplicate rows in resumed prediction shard")
    if existing - {task["key"] for task in selected}:
        raise ValueError("resumed shard does not belong to this rank/task universe")
    selected = [task for task in selected if task["key"] not in existing]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=dtype, device_map={"": local_rank}, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = list(dict.fromkeys(
        token for token in (tokenizer.eos_token_id, im_end)
        if isinstance(token, int) and token >= 0
    ))
    device = next(model.parameters()).device
    with shard.open("a", encoding="utf-8") as writer:
        for group in _chunks(selected, args.batch_size):
            prompts = [
                render_qwen_sft_tool_prefix(tokenizer, task["messages"], tools=task["tools"])
                for task in group
            ]
            encoded = tokenizer(
                prompts, padding=True, return_tensors="pt", add_special_tokens=False
            )
            input_width = encoded["input_ids"].shape[1]
            if input_width + args.max_new_tokens > args.max_context:
                raise ValueError("model prompt exceeds frozen context budget")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                    eos_token_id=ends, pad_token_id=tokenizer.pad_token_id
                )
            for task, sequence in zip(group, generated, strict=True):
                completion = _trim_completion(
                    sequence[input_width:].tolist(), tokenizer
                )
                raw = tokenizer.decode(completion, skip_special_tokens=False)
                name, arguments, generation_error = prediction_call(raw, tokenizer)
                outcome = score_prediction(
                    task, predicted_name=name, predicted_arguments=arguments
                )
                record = {
                    "key": task["key"], "reaction_id": task["reaction_id"],
                    "decision_type": task["decision_type"],
                    "event_depth": task["event_depth"],
                    "stratum": task["stratum"], "gold_name": task["gold_name"],
                    "predicted_name": name, "predicted_arguments": arguments,
                    "generation_error": generation_error,
                    "generated_text": raw, "generated_tokens": len(completion),
                    "gold_be_delta": task["gold_be_delta"],
                    "gold_flow_arrows": task["gold_flow_arrows"],
                    **outcome,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
    provenance = {
        "model": args.model, "revision": args.revision, "adapter_sha256": weight_sha,
        "input_tasks_sha256": meta["tasks_sha256"], "dtype": args.dtype,
        "greedy_k1": True, "gold_state_conditioned": True,
    }
    (args.output / f"provenance.rank-{rank:02d}.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return 0


def counts(rows: list[dict[str, Any]], keys: tuple[str, ...] = METRICS) -> dict[str, Any]:
    n = len(rows)
    result: dict[str, Any] = {"n": n}
    for key in keys:
        count = sum(bool(row.get(key)) for row in rows)
        result[key] = count
        result[key + "_rate"] = count / n if n else None
    return result


def _purpose_fragments(arguments: Mapping[str, Any], purpose: str) -> Counter:
    all_fragments = _import_counter(arguments, include_purpose=True)
    return Counter({key: amount for key, amount in all_fragments.items()
                    if key[1] == purpose})


def aggregate(args: argparse.Namespace) -> int:
    tasks = read_jsonl(args.input / "tasks.jsonl")
    meta = json.loads((args.input / "selection.json").read_text())
    if sha256(args.input / "tasks.jsonl") != meta["tasks_sha256"]:
        raise ValueError("tasks changed after preparation")
    task_index = {row["key"]: row for row in tasks}
    predictions: list[dict[str, Any]] = []
    for path in sorted(args.output.glob("decisions.shard-*-of-*.jsonl")):
        predictions.extend(read_jsonl(path))
    keys = [row["key"] for row in predictions]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate predictions in evaluation")
    missing = sorted(set(task_index) - set(keys))
    extra = sorted(set(keys) - set(task_index))
    if extra:
        raise ValueError(f"out-of-universe prediction keys: {extra[:5]}")
    by_type = {
        dtype: counts([r for r in predictions if r["decision_type"] == dtype])
        for dtype in TYPES
    }
    event = [r for r in predictions if r["decision_type"] == "event"]
    imports = [r for r in predictions if r["decision_type"] == "import"]
    non_delta = [r for r in event if not r["gold_be_delta"]]
    delta = [r for r in event if r["gold_be_delta"]]
    confusion = {gold: dict(Counter(
        r["predicted_name"] or "UNPARSEABLE" for r in predictions
        if r["gold_name"] == gold
    )) for gold in ("import_fragments", "apply_electron_flow", "finish_trace")}
    import_purpose: dict[str, Any] = {}
    for purpose in ("electron_participant", "endpoint_context"):
        cases = []
        for record in imports:
            task = task_index[record["key"]]
            reference = _purpose_fragments(task["gold_arguments"], purpose)
            if not reference:
                continue
            try:
                proposed = (
                    _purpose_fragments(record["predicted_arguments"], purpose)
                    if record["predicted_name"] == "import_fragments" else Counter()
                )
            except Exception:
                proposed = Counter()
            cases.append(proposed == reference)
        import_purpose[purpose] = {
            "decisions_containing_role": len(cases),
            "exact_decisions": sum(cases),
            "exact_rate": sum(cases) / len(cases) if cases else None,
        }
    by_reaction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_reaction[row["reaction_id"]].append(row)
    all_exact = sum(
        len(rows) == sum(task["reaction_id"] == rid for task in tasks)
        and all(row["decision_exact"] for row in rows)
        for rid, rows in by_reaction.items()
    )
    all_successors = sum(
        len(rows) == sum(task["reaction_id"] == rid for task in tasks)
        and all(
            row["successor_chemical_exact"] if row["decision_type"] == "event"
            else row["import_fragment_exact"] if row["decision_type"] == "import"
            else row["finish_exact"]
            for row in rows
        )
        for rid, rows in by_reaction.items()
    )
    n_reactions = meta["validation_reactions"]
    report = {
        "schema": "mechet_stage2_gold_state_next_action_accuracy_v1",
        "corpus": meta["corpus"],
        "claim_boundary": "teacher-forced gold-state action accuracy; not product-only E2E Top-1",
        "n_reactions": n_reactions,
        "planned_decisions": len(tasks), "completed_decisions": len(predictions),
        "missing_decisions": len(missing), "complete": not missing,
        "selection": meta,
        "overall": counts(predictions), "by_type": by_type,
        "tool_confusion_matrix": confusion,
        "import_by_purpose": import_purpose,
        "flow_by_reference_encoding": {
            "electron_arrows": counts(non_delta),
            "bond_charge_delta_fallback": counts(delta),
        },
        "flow_arrows_only_by_coupling": {
            str(n): counts([r for r in non_delta if r["gold_flow_arrows"] == n])
            for n in sorted({r["gold_flow_arrows"] for r in non_delta})
        },
        "event_by_depth": {
            str(depth): counts([r for r in event if r["event_depth"] == depth])
            for depth in sorted({r["event_depth"] for r in event})
        },
        "reaction_all_reference_actions_exact": all_exact,
        "reaction_all_local_successors_or_imports_exact": all_successors,
        "premature_finish_on_nonfinish_decisions": sum(
            r["predicted_name"] == "finish_trace"
            for r in predictions if r["decision_type"] != "finish"
        ),
        "nonfinish_decisions": sum(r["decision_type"] != "finish" for r in predictions),
        "generation_errors": dict(Counter(
            r["generation_error"] for r in predictions if r["generation_error"]
        )),
        "execution_errors": dict(Counter(
            r["execution_error"] for r in predictions if r["execution_error"]
        )),
        "note": (
            "FLOW source/sink-site exact is meaningful only in electron_arrows "
            "stratum; BE_DELTA events have no source/sink arrows. Independent "
            "local temporary atom IDs are not upstream reaction atom maps."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({
        "corpus": report["corpus"], "complete": report["complete"],
        "by_type": by_type, "reaction_all_reference_actions_exact": all_exact,
        "reaction_all_local_successors_or_imports_exact": all_successors,
    }, ensure_ascii=False), flush=True)
    return 0 if report["complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--corpus", choices=("flower_new_dataset", "mech_uspto_31k"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--expected-source-rows", type=int)
    parser.add_argument("--sample-reactions", type=int, default=256,
                        help="0 selects all source validation rows")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--input", type=Path, help="prepared dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--expected-adapter-sha", default="")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=16384)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.data is None or args.corpus is None or args.expected_source_rows is None:
            parser.error("prepare needs --data, --corpus and --expected-source-rows")
        return prepare(args)
    if args.input is None:
        parser.error("run/aggregate require --input")
    if args.command == "run":
        if args.adapter is None:
            parser.error("run requires --adapter")
        return run(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
