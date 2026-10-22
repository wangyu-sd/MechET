#!/usr/bin/env python3
"""Small, fixed-ID, real-model Stage-I/II main-product validation.

Both gold-state next-action and autonomous product-start K1 use the *same*
checkpoint and source reactions. No reference is injected during autonomous
rollout. Only an explicit, inspected report can authorize Stage-II.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.a7_rescue import stratified_sample
from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.in_place_grounded_flow import mapped_atom_numbers
from mechet.qwen35_text_training import MODEL_ID, MODEL_REVISION, load_text_only_qwen35
from scripts.build_natural_language_event_sft import convert_row
from scripts.build_natural_language_history_sft import transform_rows
from scripts.eval_natural_language_event_local import score_prediction
from scripts.run_natural_language_value_search import (
    Action, Node, execute, policy_prompt, prediction_call,
    product_only_private_state, visible,
)


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_input(source: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(source)
    if n <= 0:
        raise ValueError("Lightweight validation requires fixed positive cohort")
    chosen = stratified_sample(rows, size=n, seed=seed)
    return chosen


class Actor:
    def __init__(self, model_path: str, adapter: Path, batch: int = 1):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Frozen Qwen3.5 inference needs one CUDA device")
        if batch != 1:
            raise ValueError("This audit uses batch=1 for reproducible prompts")
        weight = adapter / "adapter_model.safetensors"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        self.adapter_hash = sha(weight)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=MODEL_REVISION, trust_remote_code=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = load_text_only_qwen35(
            revision=MODEL_REVISION, dtype=torch.bfloat16,
            quantization_config=None, device_map={"": 0},
            attn_implementation="sdpa",
        )
        self.model = PeftModel.from_pretrained(
            base, str(adapter), is_trainable=False
        ).eval()
        self.device = next(self.model.parameters()).device
        self.torch = torch
        imend = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.stop = [x for x in (self.tokenizer.eos_token_id, imend)
                     if isinstance(x, int) and x >= 0]

    def choose(self, row: dict[str, Any]) -> tuple[str, dict[str, Any], str, int]:
        messages = row["messages"][:2]
        tools = row["tools"]
        prompt = render_qwen_sft_tool_prefix(self.tokenizer, messages, tools=tools)
        encoded = self.tokenizer(prompt, add_special_tokens=False,
                                 return_tensors="pt")
        if encoded["input_ids"].shape[1] + 512 > 4096:
            raise ValueError("Qwen3.5 evaluator context exceeded training limit")
        encoded = {k:v.to(self.device) for k,v in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=512,
                eos_token_id=self.stop, pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = output[0, encoded["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(completion, skip_special_tokens=False)
        name, args, error = prediction_call(raw, self.tokenizer)
        return name, args, error, completion.shape[0]


def _node(target: str) -> Node:
    state = product_only_private_state(target)
    return Node(
        target=target, state=state,
        next_map=max(mapped_atom_numbers(state)) + 1,
        visited={visible(state)},
    )


def _model_row(target: str, node: Node, tools: list, system: str,
               compact_history: bool) -> dict[str, Any]:
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": policy_prompt(
            target, node.state, include_inventory=True,
            actions=node.actions, compact_history=compact_history,
        )},
    ], "tools": tools}


def predict_one(own: Actor, row: dict) -> tuple[str, dict, str, int]:
    return own.choose(row)


def assess_reaction(own: Actor, source: dict[str, Any],
                    compact_history: bool) -> dict[str, Any]:
    gold_public = convert_row(source)
    gold = list(transform_rows(gold_public)) if compact_history else gold_public
    target = str(gold[0]["target_smiles"])
    if "." in target:
        raise ValueError("Validation input is not main-product-only")
    node = _node(target)
    rows: list[dict[str, Any]] = []
    exact_decisions = 0
    for idx, reference in enumerate(gold):
        expected_prompt = reference["messages"][1]["content"]
        current_row = _model_row(
            target, node, reference["tools"],
            reference["messages"][0]["content"], compact_history,
        )
        if current_row["messages"][1]["content"] != expected_prompt:
            raise ValueError(
                f"gold-state prompt drift at {source['source_id']} step {idx}"
            )
        function = reference["messages"][2]["tool_calls"][0]["function"]
        gold_name = function["name"]
        gold_args = dict(function["arguments"])
        private_state = node.state
        child, ref_err = execute(
            node, Action(gold_name, gold_args, "", 0, 0),
            max_imports=32, reject_target_retained_finish=False,
        )
        if child is None:
            raise ValueError(f"GT reference did not execute: {ref_err}")
        name, args, error, tokens = predict_one(own, current_row)
        decision_type = str(reference["metadata"]["decision_type"])
        task = {
            "decision_type": decision_type,
            "gold_name": gold_name,
            "gold_arguments": gold_args,
            "private_state": private_state,
            "reference_successor": child.state if decision_type == "event" else "",
        }
        scores = score_prediction(task, predicted_name=name, predicted_arguments=args)
        rows.append({
            "step": idx, "gold_tool": gold_name, "predicted_tool": name,
            "decision_type": decision_type,
            "root": idx == 0, "generation_error": error,
            "generation_tokens": int(tokens), **scores,
        })
        exact_decisions += int(bool(scores["decision_exact"]))
        node = child
    if not node.terminal:
        raise ValueError("GT trajectory failed to finish")
    if not structural_exact(
        split_precursor_endpoints(node.state, _node(target).state).structural,
        split_precursor_endpoints(source["full_precursor_state"],
                                  _node(target).state).structural,
    ):
        raise ValueError("GT reference endpoint differs in main-derived precursors")
    # Independent, autonomous, product-only rollout. No GT action/state is used.
    node = _node(target)
    attempts = []
    for step in range(40):
        generated_row = _model_row(
            target, node, gold[0]["tools"],
            gold[0]["messages"][0]["content"], compact_history,
        )
        name, args, error, tokens = predict_one(own, generated_row)
        if error:
            attempts.append({"step": step, "error": error})
            break
        next_node, error = execute(
            node, Action(name, args, "", 0, int(tokens)),
            max_imports=32, reject_target_retained_finish=False,
        )
        attempts.append({
            "step": step, "tool": name, "error": error,
            "accepted": next_node is not None,
        })
        if next_node is None:
            break
        node = next_node
        if node.terminal:
            break
    expected = split_precursor_endpoints(
        source["full_precursor_state"], _node(target).state,
    ).structural
    pred = split_precursor_endpoints(node.state, _node(target).state).structural
    product_hit = bool(node.terminal and structural_exact(pred, expected))
    return {
        "reaction_id": str(source["source_id"]),
        "reference_decisions": len(gold),
        "next_action_exact": exact_decisions,
        "decision_scores": rows,
        "terminal": bool(node.terminal),
        "product_start_structural_hit": product_hit,
        "attempts": attempts,
        "product_start_reference_scope": "main-product-derived precursor",
    }


def aggregate(records: list[dict[str, Any]], *, stage: str,
              model_hash: str, selected_hash: str) -> dict[str, Any]:
    decisions = [row for rxn in records for row in rxn["decision_scores"]]
    roots = [r for r in decisions if r["root"]]
    root_imports = [r for r in roots if r["decision_type"] == "import"]
    flow = [r for r in decisions if r["decision_type"] == "event"]
    finish = [r for r in decisions if r["decision_type"] == "finish"]
    ratio = lambda num, den: num / den if den else None
    return {
        "status": "complete",
        "evaluation": "greedy_k1_gold_state_and_product_start",
        "stage": stage,
        "reactions": len(records),
        "decisions": len(decisions),
        "checkpoint_weight_sha256": model_hash,
        "selected_reactions_sha256": selected_hash,
        "root_import_denominator": len(root_imports),
        "root_import_tool_acc": ratio(
            sum(r["correct_tool"] for r in root_imports), len(root_imports)),
        "root_import_exact": ratio(
            sum(r["import_fragment_exact"] for r in root_imports), len(root_imports)),
        "flow_denominator": len(flow),
        "flow_chemical_successor_exact": ratio(
            sum(r["successor_chemical_exact"] for r in flow), len(flow)),
        "finish_accuracy": ratio(sum(r["finish_exact"] for r in finish), len(finish)),
        "overall_tool_accuracy": ratio(
            sum(r["correct_tool"] for r in decisions), len(decisions)),
        "product_start_structural_top1": ratio(
            sum(r["product_start_structural_hit"] for r in records), len(records)),
        "claim_boundary": "fixed sampled VALID of replayable trace view, not complete benchmark or GT-free chemistry validity",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-valid", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--stage", choices=("state", "history"), required=True)
    ap.add_argument("--sample-reactions", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite validation report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = select_input(args.source_valid, args.sample_reactions, args.seed)
    ids = "\n".join(str(row["source_id"]) for row in sources)
    digest = hashlib.sha256(ids.encode()).hexdigest()
    actor = Actor(MODEL_ID, args.adapter)
    records = []
    with args.output.with_suffix(".reactions.jsonl").open("x", encoding="utf-8") as sink:
        for index, source in enumerate(sources, 1):
            item = assess_reaction(actor, source, args.stage == "history")
            records.append(item)
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")
            sink.flush()
            if index % 8 == 0 or index == len(sources):
                print(json.dumps({
                    "phase": "QWEN35_MAIN_ONLY_LIGHT_VALIDATION",
                    "stage": args.stage, "done": index, "planned": len(sources),
                    "root_import_exact": sum(
                        r["import_fragment_exact"]
                        for x in records for r in x["decision_scores"]
                        if r["root"] and r["decision_type"] == "import"),
                    "product_start_hits": sum(
                        x["product_start_structural_hit"] for x in records),
                }), flush=True)
    report = aggregate(records, stage=args.stage,
                       model_hash=actor.adapter_hash, selected_hash=digest)
    report["source_valid_sha256"] = sha(args.source_valid)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
