#!/usr/bin/env python3
"""Build frozen H3 evidence-content interventions without changing chemistry traces."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **_kwargs):
        return iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import read_jsonl, row_id, write_jsonl

TEXTBOOK_TOOL = "retrieve_textbook_guidance"
ANCHOR_TOOL = "retrieve_primitives"
_WARNING_RE = re.compile(r"(?im)^.*\bwarning(?:s)?\b.*$")
_COMPETITOR_RE = re.compile(
    r"(?im)^.*\b(competing|competitor|alternative pathway|side pathway)\b.*$"
)
_WARNING_KEYS = {"warnings", "warning"}
_COMPETITOR_KEYS = {
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
    row: Mapping[str, Any],
    result: dict[str, Any],
    text: str,
    *,
    intervention: str,
) -> dict[str, Any]:
    value = deepcopy(dict(row))
    index, _ = _tool_result_location(value, TEXTBOOK_TOOL)
    context = dict(result.get("context") or {})
    target_length = int(
        context.get("n_characters") or len(str(context.get("text") or ""))
    )
    bounded = _fit(text, target_length)
    context.update(
        {
            "text": bounded,
            "n_characters": len(bounded),
            "context_sha256": hashlib.sha256(bounded.encode()).hexdigest(),
            "evidence_intervention": intervention,
        }
    )
    changed_result = deepcopy(result)
    changed_result["context"] = context
    changed_result["matches"] = (
        []
        if intervention in {"passage_shuffle", "same_topic_wrong"}
        else changed_result.get("matches", [])
    )
    changed_result["evidence_intervention"] = intervention
    changed_result["direct_reward"] = False
    value["messages"][index]["content"] = json.dumps(
        changed_result, ensure_ascii=False
    )
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
    for index, (row, original) in enumerate(
        tqdm(prepared, desc="passage_shuffle", unit="row")
    ):
        donor = prepared[(index + 1) % len(prepared)]
        if row_id(donor[0]) == row_id(row):
            raise ValueError(f"{row_id(row)}: passage shuffle selected itself")
        donor_text = str((donor[1].get("context") or {}).get("text") or "")
        value = _set_context(
            row, original, donor_text, intervention="passage_shuffle"
        )
        metadata = dict(value.get("metadata") or {})
        metadata["evidence_donor_id"] = row_id(donor[0])
        value["metadata"] = metadata
        output.append(value)
    return output


def same_topic_wrong(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = []
    term_index: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        original = _tool_result_location(row, TEXTBOOK_TOOL)[1]
        terms = _terms(original)
        passage_ids = set((original.get("context") or {}).get("passage_ids") or [])
        row_key = row_id(row)
        prepared.append((row, original, row_key, terms, passage_ids))
        for term in terms:
            term_index[term].append(len(prepared) - 1)
    output: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for index, (row, original, row_key, source_terms, source_ids) in enumerate(
        tqdm(prepared, desc="same_topic_wrong", unit="row")
    ):
        candidates = []
        candidate_indices: set[int] = set()
        for term in source_terms:
            candidate_indices.update(term_index.get(term, ()))
        candidate_indices.discard(index)
        for donor_index in candidate_indices:
            donor_row, donor_result, donor_key, donor_terms, donor_ids = prepared[
                donor_index
            ]
            shared = source_terms & donor_terms
            if shared and source_ids.isdisjoint(donor_ids):
                candidates.append(
                    (len(shared), donor_key, donor_row, donor_result, shared)
                )
        if not candidates:
            quarantined.append(
                {
                    "id": row_key,
                    "error_code": "NO_SAME_TOPIC_WRONG_PASSAGE_DONOR",
                    "error": (
                        f"{row_key}: no same-topic wrong-passage donor; "
                        "add reviewed topic labels"
                    ),
                }
            )
            continue
        _, _, donor_row, donor_result, shared = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        donor_text = str((donor_result.get("context") or {}).get("text") or "")
        value = _set_context(
            row, original, donor_text, intervention="same_topic_wrong"
        )
        metadata = dict(value.get("metadata") or {})
        metadata["evidence_donor_id"] = row_id(donor_row)
        metadata["shared_retrieval_terms"] = sorted(shared)
        value["metadata"] = metadata
        output.append(value)
    return output, quarantined


def _remove_selected_keys(value: Any, blocked: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_selected_keys(item, blocked)
            for key, item in value.items()
            if str(key).lower() not in blocked
        }
    if isinstance(value, list):
        return [_remove_selected_keys(item, blocked) for item in value]
    return value


def _rewrite_anchor_result(
    row: dict[str, Any], blocked: set[str], intervention: str
) -> dict[str, Any]:
    value = deepcopy(row)
    try:
        index, result = _tool_result_location(value, ANCHOR_TOOL)
    except ValueError:
        return value
    changed = _remove_selected_keys(result, blocked)
    changed["evidence_intervention"] = intervention
    changed["direct_reward"] = False
    value["messages"][index]["content"] = json.dumps(changed, ensure_ascii=False)
    return value


def remove_warnings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="remove_warnings", unit="row"):
        value = _rewrite_anchor_result(row, _WARNING_KEYS, "remove_warnings")
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


def remove_competing_pathways(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="remove_competing_pathways", unit="row"):
        value = _rewrite_anchor_result(
            row, _COMPETITOR_KEYS, "remove_competing_pathways"
        )
        try:
            _, result = _tool_result_location(value, TEXTBOOK_TOOL)
            text = str((result.get("context") or {}).get("text") or "")
            redacted = _COMPETITOR_RE.sub(
                lambda match: " " * len(match.group(0)), text
            )
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
    rows_by_id = {row_id(row): row for row in rows}
    transforms = {
        "passage_shuffle": passage_shuffle,
        "same_topic_wrong": same_topic_wrong,
        "remove_warnings": remove_warnings,
        "remove_competing_pathways": remove_competing_pathways,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in args.intervention:
        quarantined: list[dict[str, Any]] = []
        transformed = transforms[name](rows)
        if name == "same_topic_wrong":
            transformed, quarantined = transformed
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, transformed)
        quarantine_path = None
        if quarantined:
            quarantine_path = args.output_dir / f"{name}.quarantine.jsonl"
            write_jsonl(quarantine_path, quarantined)
        outputs[name] = {
            "path": str(path),
            "n_rows": len(transformed),
            "n_quarantined": len(quarantined),
            "quarantine": str(quarantine_path) if quarantine_path else None,
            "quarantine_reasons": dict(
                Counter(item["error_code"] for item in quarantined)
            ),
            "stable_ids_sha256": hashlib.sha256(
                "\n".join(row_id(item) for item in transformed).encode()
            ).hexdigest(),
            "characters_preserved": all(
                int((rows_by_id[row_id(item)].get("metadata") or {}).get("textbook_context_characters") or 0)
                == int((item.get("metadata") or {}).get("textbook_context_characters") or 0)
                for item in transformed
                if row_id(item) in rows_by_id
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
            "intervention_fields_are_isolated": True,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
