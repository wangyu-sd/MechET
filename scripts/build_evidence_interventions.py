#!/usr/bin/env python3
"""Build frozen H3 evidence-content interventions without changing chemistry traces."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import read_jsonl, row_id, write_jsonl


TEXTBOOK_TOOL = "retrieve_textbook_guidance"
ANCHOR_TOOL = "retrieve_primitives"
_WARNING_RE = re.compile(r"(?im)^.*\bwarning(?:s)?\b.*$")
_COMPETITOR_RE = re.compile(
    r"(?im)^.*\b(competing|competitor|alternative pathway|side pathway)\b.*$"
)
_REMOVE_KEYS = {
    "warnings",
    "warning",
    "competitors",
    "competing_primitives",
    "competing_pathways",
    "possible_followups",
}


def _tool_result_location(
    row: Mapping[str, Any], name: str
) -> tuple[int, dict[str, Any]]:
    for index, message in enumerate(row.get("messages") or []):
        if message.get("role") == "tool" and message.get("name") == name:
            value = json.loads(str(message.get("content") or "{}"))
            if not isinstance(value, dict):
                raise ValueError(f"{row_id(row)}: {name} result is not an object")
            return index, dict(value)
    raise ValueError(f"{row_id(row)}: missing {name} result")


def _fit(text: str, length: int) -> str:
    value = str(text or "")
    if len(value) >= length:
        return value[:length]
    return value + " " * (length - len(value))


def _set_context(
    row: Mapping[str, Any], result: dict[str, Any], text: str, *, intervention: str
) -> dict[str, Any]:
    value = deepcopy(dict(row))
    index, _ = _tool_result_location(value, TEXTBOOK_TOOL)
    context = dict(result.get("context") or {})
    target_length = int(context.get("n_characters") or len(str(context.get("text") or "")))
    bounded = _fit(text, target_length)
    context.update(
        {
            "text": bounded,
            "n_characters": len(bounded),
            "context_sha256": hashlib.sha256(bounded.encode()).hexdigest(),
            "evidence_intervention": intervention,
        }
    )
    result = deepcopy(result)
    result["context"] = context
    result["matches"] = [] if intervention in {"passage_shuffle", "same_topic_wrong"} else result.get("matches", [])
    result["evidence_intervention"] = intervention
    result["direct_reward"] = False
    value["messages"][index]["content"] = json.dumps(result, ensure_ascii=False)
    metadata = dict(value.get("metadata") or {})
    metadata.update(
        {
            "evidence_intervention": intervention,
            "textbook_context_sha256": context["context_sha256"],
            "textbook_context_characters": context["n_characters"],
        }
    )
    value["metadata"] = metadata
    return value


def _terms(result: Mapping[str, Any]) -> set[str]:
    output: set[str] = set()
    for match in result.get("matches") or []:
        for key in ("matched_terms", "state_terms"):
            output.update(str(item).lower() for item in match.get(key) or [])
    return {item for item in output if item}


def passage_shuffle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("passage shuffle requires at least two rows")
    prepared = [(row, _tool_result_location(row, TEXTBOOK_TOOL)[1]) for row in rows]
    output: list[dict[str, Any]] = []
    for index, (row, original) in enumerate(prepared):
        donor = prepared[(index + 1) % len(prepared)]
        donor_text = str((donor[1].get("context") or {}).get("text") or "")
        value = _set_context(
            row, original, donor_text, intervention="passage_shuffle"
        )
        metadata = dict(value.get("metadata") or {})
        metadata["evidence_donor_id"] = row_id(donor[0])
        value["metadata"] = metadata
        output.append(value)
    return output


def same_topic_wrong(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = [(row, _tool_result_location(row, TEXTBOOK_TOOL)[1]) for row in rows]
    output: list[dict[str, Any]] = []
    for row, original in prepared:
        source_terms = _terms(original)
        source_ids = set((original.get("context") or {}).get("passage_ids") or [])
        candidates = []
        for donor_row, donor_result in prepared:
            if row_id(donor_row) == row_id(row):
                continue
            donor_ids = set((donor_result.get("context") or {}).get("passage_ids") or [])
            overlap = len(source_terms & _terms(donor_result))
            if overlap and source_ids.isdisjoint(donor_ids):
                candidates.append((overlap, row_id(donor_row), donor_row, donor_result))
        if not candidates:
            raise ValueError(
                f"{row_id(row)}: no same-topic wrong-passage donor; add reviewed topic labels"
            )
        _, _, donor_row, donor_result = max(candidates, key=lambda item: (item[0], item[1]))
        donor_text = str((donor_result.get("context") or {}).get("text") or "")
        value = _set_context(
            row, original, donor_text, intervention="same_topic_wrong"
        )
        metadata = dict(value.get("metadata") or {})
        metadata["evidence_donor_id"] = row_id(donor_row)
        metadata["shared_retrieval_terms"] = sorted(source_terms & _terms(donor_result))
        value["metadata"] = metadata
        output.append(value)
    return output


def _remove_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_keys(item)
            for key, item in value.items()
            if str(key).lower() not in _REMOVE_KEYS
        }
    if isinstance(value, list):
        return [_remove_keys(item) for item in value]
    return value


def remove_warnings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        value = deepcopy(row)
        try:
            index, result = _tool_result_location(value, ANCHOR_TOOL)
            value["messages"][index]["content"] = json.dumps(
                _remove_keys(result), ensure_ascii=False
            )
        except ValueError:
            pass
        try:
            _, result = _tool_result_location(value, TEXTBOOK_TOOL)
            text = str((result.get("context") or {}).get("text") or "")
            redacted = _WARNING_RE.sub(lambda match: " " * len(match.group(0)), text)
            value = _set_context(
                value, result, redacted, intervention="remove_warnings"
            )
        except ValueError:
            metadata = dict(value.get("metadata") or {})
            metadata["evidence_intervention"] = "remove_warnings"
            value["metadata"] = metadata
        output.append(value)
    return output


def remove_competing_pathways(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        value = deepcopy(row)
        try:
            index, result = _tool_result_location(value, ANCHOR_TOOL)
            value["messages"][index]["content"] = json.dumps(
                _remove_keys(result), ensure_ascii=False
            )
        except ValueError:
            pass
        try:
            _, result = _tool_result_location(value, TEXTBOOK_TOOL)
            text = str((result.get("context") or {}).get("text") or "")
            redacted = _COMPETITOR_RE.sub(lambda match: " " * len(match.group(0)), text)
            value = _set_context(
                value,
                result,
                redacted,
                intervention="remove_competing_pathways",
            )
        except ValueError:
            metadata = dict(value.get("metadata") or {})
            metadata["evidence_intervention"] = "remove_competing_pathways"
            value["metadata"] = metadata
        output.append(value)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--intervention",
        action="append",
        choices=[
            "passage_shuffle",
            "same_topic_wrong",
            "remove_warnings",
            "remove_competing_pathways",
        ],
        required=True,
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    transforms = {
        "passage_shuffle": passage_shuffle,
        "same_topic_wrong": same_topic_wrong,
        "remove_warnings": remove_warnings,
        "remove_competing_pathways": remove_competing_pathways,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in args.intervention:
        transformed = transforms[name](rows)
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, transformed)
        outputs[name] = {
            "path": str(path),
            "n_rows": len(transformed),
            "stable_ids_sha256": hashlib.sha256(
                "\n".join(row_id(item) for item in transformed).encode()
            ).hexdigest(),
            "characters_preserved": all(
                int((left.get("metadata") or {}).get("textbook_context_characters") or 0)
                == int((right.get("metadata") or {}).get("textbook_context_characters") or 0)
                for left, right in zip(rows, transformed)
            ),
        }
    manifest = {
        "artifact_type": "evidence_intervention_suite",
        "input": str(args.input),
        "n_rows": len(rows),
        "outputs": outputs,
        "contract": {
            "same_ids_targets_endpoints": True,
            "same_chemistry_trace": True,
            "same_context_character_budget": True,
            "direct_reward": False,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
