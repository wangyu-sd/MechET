#!/usr/bin/env python3
"""Pair saved Stage-I smoke outputs with their frozen model-visible inputs.

The original gold-state shard saved raw completions but omitted prompts. The
product-only shard saved parsed proposals/feedback but omitted raw completion
text. This export reconstructs prompts; it never fabricates missing completions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.in_place_grounded_flow import mapped_atom_numbers
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from scripts.run_natural_language_value_search import (
    Action,
    Node,
    execute,
    policy_prompt,
    product_only_private_state,
    visible,
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_local(
    *, data: Path, predictions: Path, tokenizer: Any, output: Path,
) -> tuple[int, dict[str, dict[str, Any]]]:
    saved = list(read_jsonl(predictions))
    by_key = {str(row["key"]): row for row in saved}
    if len(by_key) != len(saved):
        raise ValueError("duplicate local prediction key")
    frozen: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(data):
        key = str(row["id"])
        if key in by_key:
            frozen[key] = row
    if set(frozen) != set(by_key):
        raise ValueError("local predictions do not match frozen validation rows")
    first_by_reaction: dict[str, dict[str, Any]] = {}
    with output.open("w", encoding="utf-8") as handle:
        for prediction in saved:
            key = str(prediction["key"])
            row = frozen[key]
            messages = row["messages"][:2]
            tools = row["tools"]
            prompt = render_qwen_sft_tool_prefix(tokenizer, messages, tools=tools)
            reference = row["messages"][2]["tool_calls"][0]["function"]
            if str(reference["name"]) != str(prediction["gold_name"]):
                raise ValueError(f"{key}: reference tool mismatch")
            record = {
                "key": key,
                "reaction_id": prediction["reaction_id"],
                "input_source": "frozen_v2_validation_decision_row",
                "input_messages": messages,
                "input_tool_schema": tools,
                "rendered_model_prompt": prompt,
                "rendered_prompt_provenance": "re-rendered_from_frozen_messages_with_saved_tokenizer",
                "raw_model_completion": prediction["generated_text"],
                "parsed_model_output": {
                    "name": prediction["predicted_name"],
                    "arguments": prediction["predicted_arguments"],
                    "generation_error": prediction["generation_error"],
                },
                "reference_tool_call_evaluator_only": reference,
                "saved_prediction_record": prediction,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            first_by_reaction.setdefault(str(prediction["reaction_id"]), record)
    return len(saved), first_by_reaction


def export_product(
    *, episodes: Path, source_data: Path, tokenizer: Any,
    first_local: dict[str, dict[str, Any]],
    output: Path,
) -> tuple[int, int]:
    saved = list(read_jsonl(episodes))
    selected_ids = {str(row["id"]) for row in saved}
    sources = {
        str(row["id"]): row for row in read_jsonl(source_data)
        if str(row["id"]) in selected_ids
    }
    if set(sources) != selected_ids:
        raise ValueError("product episodes do not match frozen source reactions")
    reactions = decisions = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in saved:
            if len(row["episodes"]) != 1:
                raise ValueError("this export requires K=1")
            source_id = str(row["source_id"])
            # The saved display target is unmapped; the actual run started from
            # the frozen mapped source product, retaining explicit hydrogens.
            source = sources[str(row["id"])]
            target_mapped = product_only_private_state(str(source["target_smiles"]))
            root_visible = visible(target_mapped)
            node = Node(
                target=root_visible,
                state=target_mapped,
                next_map=max(mapped_atom_numbers(target_mapped), default=0) + 1,
                visited={root_visible},
            )
            attempts = row["episodes"][0]["attempts"]
            if not attempts:
                raise ValueError(f"{source_id}: missing attempt records")
            accepted: list[dict[str, Any]] = []
            for depth, attempt in enumerate(attempts):
                if int(attempt["depth"]) != depth:
                    raise ValueError(f"{source_id}: unexpected attempt order")
                state_before = visible(node.state)
                if state_before != str(attempt["state_before"]):
                    raise ValueError(f"{source_id}: replay state disagrees at depth {depth}")
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": policy_prompt(
                        node.target, node.state,
                        include_inventory=True,
                        actions=node.actions,
                        compact_history=False,
                    )},
                ]
                prompt = render_qwen_sft_tool_prefix(tokenizer, messages, tools=TOOLS)
                if depth == 0:
                    local = first_local.get(source_id)
                    if local is None or messages != local["input_messages"]:
                        raise ValueError(f"{source_id}: product-start prompt differs from frozen local input")
                    if prompt != local["rendered_model_prompt"]:
                        raise ValueError(f"{source_id}: rendered product-start prompt differs")
                name = str(attempt.get("name") or "")
                arguments = dict(attempt.get("arguments") or {})
                record = {
                    "reaction_id": source_id,
                    "episode_id": row["id"],
                    "decision_index": depth,
                    "input_source": "product_only_executor_replay_of_saved_attempts",
                    "input_messages": messages,
                    "input_tool_schema": TOOLS,
                    "rendered_model_prompt": prompt,
                    "rendered_prompt_provenance": "reconstructed_from_saved_actions_and_frozen_runtime",
                    "raw_model_completion": None,
                    "raw_completion_status": "not_saved_by_original_product_only_evaluator",
                    "parsed_model_output": {"name": name, "arguments": arguments},
                    "executor_feedback": {
                        "accepted": bool(attempt["accepted"]),
                        "error": str(attempt.get("error") or ""),
                        "state_before": state_before,
                        "state_after": str(attempt.get("state_after") or ""),
                        "terminal": bool(attempt.get("terminal")),
                    },
                    "saved_attempt_record": attempt,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                decisions += 1
                if bool(attempt["accepted"]):
                    child, error = execute(
                        node,
                        Action(name, arguments, "", 0.0, 0),
                        max_imports=32,
                        reject_target_retained_finish=True,
                    )
                    if child is None or error:
                        raise ValueError(f"{source_id}: saved accepted action failed replay: {error}")
                    result_state = str(
                        child.actions[-1]["result"].get("current_state")
                        or child.actions[-1]["result"].get("derived_precursor")
                        or ""
                    )
                    if result_state != str(attempt["state_after"]):
                        raise ValueError(f"{source_id}: saved successor differs at depth {depth}")
                    accepted.append({"name": name, "arguments": arguments})
                    node = child
            if accepted != row["episodes"][0]["actions"]:
                raise ValueError(f"{source_id}: saved accepted-action list differs")
            reactions += 1
    return reactions, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-decisions", type=Path, required=True)
    parser.add_argument("--local-predictions", type=Path, required=True)
    parser.add_argument("--product-episodes", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_path = args.output_dir / "gold_state_bs1" / "input_output.jsonl"
    product_path = args.output_dir / "product_only_k1" / "input_output.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    product_path.parent.mkdir(parents=True, exist_ok=True)
    local_n, first = export_local(
        data=args.valid_decisions,
        predictions=args.local_predictions,
        tokenizer=tokenizer,
        output=local_path,
    )
    product_n, product_decisions = export_product(
        episodes=args.product_episodes,
        source_data=args.source_data,
        tokenizer=tokenizer,
        first_local=first,
        output=product_path,
    )
    print(json.dumps({
        "local_decisions": local_n,
        "product_reactions": product_n,
        "product_decisions": product_decisions,
        "local_io_sha256": sha256(local_path),
        "product_io_sha256": sha256(product_path),
        "tokenizer_json_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "chat_template_sha256": sha256(args.tokenizer / "chat_template.jinja"),
    }, indent=2))


if __name__ == "__main__":
    main()
