#!/usr/bin/env python3
"""Fail-closed protocol-v2 train/inference parity audit for Stage-I/II policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import (
    render_chat,
    render_qwen_sft_tool_prefix,
    tokenize_text,
)
from mechet.trajectory_history import TrajectoryHistory
from scripts.build_natural_language_event_sft import TOOLS, convert_row
from scripts.build_natural_language_history_sft import transform_rows
from scripts.run_natural_language_value_search import (
    PROMPT_SUFFIX,
    read_selected,
    validate_v2_adapter_manifest,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exchange(row: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    messages = list(row.get("messages") or [])
    assistant = next(message for message in messages if message.get("role") == "assistant")
    tool = next(message for message in messages if message.get("role") == "tool")
    call = dict((assistant.get("tool_calls") or [])[0])
    function = dict(call.get("function") or {})
    return (
        str(function.get("name") or ""),
        dict(function.get("arguments") or {}),
        dict(json.loads(str(tool.get("content") or "{}"))),
    )


def reconstruct_history_user_prompt(
    base_user_prompt: str,
    accepted_actions: Sequence[Mapping[str, Any]],
) -> str:
    if not base_user_prompt.endswith(PROMPT_SUFFIX):
        raise ValueError("protocol-v2 base prompt suffix changed")
    history = TrajectoryHistory()
    for record in accepted_actions:
        history = history.accept(
            str(record["name"]),
            dict(record["arguments"]),
            dict(record["result"]),
        )
    return (
        base_user_prompt[: -len(PROMPT_SUFFIX)]
        + "\n\n"
        + history.render()
        + PROMPT_SUFFIX
    )


def assert_tool_prefix_matches_sft(tokenizer: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    messages = [dict(message) for message in row.get("messages") or []]
    tools = [dict(tool) for tool in row.get("tools") or []]
    assistant_index = next(
        index for index, message in enumerate(messages)
        if str(message.get("role") or "") == "assistant"
    )
    history = messages[:assistant_index]
    full = render_chat(tokenizer, messages, tools=tools, add_generation_prompt=False)
    prefix = render_qwen_sft_tool_prefix(tokenizer, history, tools=tools)
    generation = render_chat(
        tokenizer, history, tools=tools, add_generation_prompt=True
    )
    full_ids = tokenize_text(tokenizer, full)
    prefix_ids = tokenize_text(tokenizer, prefix)
    generation_ids = tokenize_text(tokenizer, generation)
    prefix_match = full_ids[: len(prefix_ids)] == prefix_ids
    generation_match = full_ids[: len(generation_ids)] == generation_ids
    if not prefix_match:
        raise ValueError("SFT-aligned inference prefix does not match training tokens")
    return {
        "prefix_match": prefix_match,
        "generation_template_match": generation_match,
        "prefix_tokens": len(prefix_ids),
        "full_tokens": len(full_ids),
    }


def audit_reaction(tokenizer: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    state_rows = list(convert_row(source))
    history_rows = list(transform_rows(state_rows))
    if len(state_rows) != len(history_rows):
        raise ValueError("State/Trajectory decision count mismatch")
    accepted: list[dict[str, Any]] = []
    generation_template_matches = 0
    for state_row, history_row in zip(state_rows, history_rows, strict=True):
        if state_row.get("tools") != TOOLS or history_row.get("tools") != TOOLS:
            raise ValueError("tool schema drift")
        if (state_row.get("metadata") or {}).get("decision_contract") != (
            "unified_inventory_tool_decision_v2"
        ):
            raise ValueError("State-SFT decision contract drift")
        if (history_row.get("metadata") or {}).get("decision_contract") != (
            "unified_inventory_compressed_history_tool_decision_v2"
        ):
            raise ValueError("Trajectory-SFT decision contract drift")
        state_user = next(
            message for message in state_row["messages"] if message.get("role") == "user"
        )["content"]
        history_user = next(
            message for message in history_row["messages"] if message.get("role") == "user"
        )["content"]
        rebuilt = reconstruct_history_user_prompt(str(state_user), accepted)
        if rebuilt != history_user:
            raise ValueError("runtime compact-history prompt differs from Stage-II training")
        for row in (state_row, history_row):
            report = assert_tool_prefix_matches_sft(tokenizer, row)
            generation_template_matches += int(report["generation_template_match"])
        name, arguments, result = _exchange(state_row)
        accepted.append({"name": name, "arguments": arguments, "result": result})
    return {
        "source_id": str(source.get("source_id") or source.get("id") or ""),
        "decisions": len(state_rows),
        "generation_template_matches": generation_template_matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state-adapter", type=Path, required=True)
    parser.add_argument("--history-adapter", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-reactions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    import rdkit
    from transformers import AutoTokenizer

    if rdkit.__version__ != "2026.03.4":
        raise RuntimeError(
            f"protocol-v2 audit requires RDKit 2026.03.4, got {rdkit.__version__}"
        )
    validate_v2_adapter_manifest(args.state_adapter, compact_history=False)
    validate_v2_adapter_manifest(args.history_adapter, compact_history=True)
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(f"local tokenizer snapshot missing: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )

    selected = read_selected(args.source, args.sample_reactions, args.seed)
    rows = [audit_reaction(tokenizer, source) for source in selected]
    decisions = sum(int(row["decisions"]) for row in rows)
    generation_template_matches = sum(
        int(row["generation_template_matches"]) for row in rows
    )
    report = {
        "artifact_type": "stage2_protocol_v2_train_inference_parity_audit",
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "sample_reactions": len(rows),
        "seed": args.seed,
        "decision_rows": decisions,
        "state_adapter": str(args.state_adapter),
        "state_adapter_manifest_sha256": sha256(
            args.state_adapter / "adapter_manifest.json"
        ),
        "history_adapter": str(args.history_adapter),
        "history_adapter_manifest_sha256": sha256(
            args.history_adapter / "adapter_manifest.json"
        ),
        "rdkit_version": rdkit.__version__,
        "unified_prompt": True,
        "runtime_history_exact": True,
        "sft_tool_prefix_token_exact": True,
        "ordinary_generation_template_matches": generation_template_matches,
        "matched_v2_contract": {
            "max_decisions": 40,
            "max_imports": 32,
            "branching": 1,
            "beam": 1,
            "greedy": True,
            "value_critic": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
