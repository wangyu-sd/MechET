#!/usr/bin/env python3
"""Gate compact-full-state rows against the frozen legacy full-state rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.assistant_masking import encode_assistant_only_conversation  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loads(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("row must be a JSON object")
    return value


def _tool_calls(row: dict[str, Any]) -> list[str]:
    return [
        json.dumps(message["tool_calls"], ensure_ascii=False, separators=(",", ":"))
        for message in row["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]


def _tool_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        json.loads(message["content"])
        for message in row["messages"]
        if message.get("role") == "tool"
    ]


def _nonterminal_states(row: dict[str, Any], key: str) -> list[str]:
    return [
        str(result[key])
        for result in _tool_results(row)[:-1]
        if key in result
    ]


def _terminal(row: dict[str, Any]) -> dict[str, Any]:
    results = _tool_results(row)
    if not results:
        raise ValueError("row has no tool result")
    return results[-1]


def _product_occurs_once(row: dict[str, Any]) -> bool:
    user = next(
        message["content"]
        for message in row["messages"]
        if message.get("role") == "user"
    )
    first_line = str(user).splitlines()[0]
    if not first_line.startswith("TARGET: "):
        return False
    product = first_line.removeprefix("TARGET: ")
    return str(user).count(product) == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-token-ratio", type=float, default=0.60)
    args = parser.parse_args()

    config = dict(yaml.safe_load(args.config.read_text(encoding="utf-8")) or {})
    training = dict(config.get("training") or {})
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name_or_path"]),
        revision=str(training["model_revision"]),
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )

    failures: list[dict[str, Any]] = []
    compact_tokens = legacy_tokens = rows = 0
    compact_max_tokens = legacy_max_tokens = 0
    compact_ids: list[str] = []
    legacy_ids: list[str] = []
    forbidden = {
        "state_smiles",
        "state_before",
        "state_after",
        "trace_step",
        "trace_digest",
        "move_sequence_digest",
        "compiled_proof",
        "full_precursor_state",
        "structural_precursor",
        "pending_imports",
        "imported_fragment",
    }
    with args.compact.open(encoding="utf-8") as compact_handle, args.legacy.open(
        encoding="utf-8"
    ) as legacy_handle:
        for compact_line in compact_handle:
            if not compact_line.strip():
                continue
            legacy_line = legacy_handle.readline()
            while legacy_line and not legacy_line.strip():
                legacy_line = legacy_handle.readline()
            if not legacy_line:
                failures.append({"code": "LEGACY_ROWS_MISSING", "row": rows})
                break
            compact = _loads(compact_line)
            legacy = _loads(legacy_line)
            rows += 1
            identifier = str(compact.get("id") or "")
            compact_ids.append(identifier)
            legacy_ids.append(str(legacy.get("id") or ""))

            checks = {
                "STABLE_ID_MISMATCH": compact.get("id") == legacy.get("id")
                and compact.get("source_id") == legacy.get("source_id"),
                "ACTION_TARGET_MISMATCH": _tool_calls(compact)
                == _tool_calls(legacy),
                "STATE_SEQUENCE_MISMATCH": _nonterminal_states(
                    compact, "current_state_smiles"
                )
                == _nonterminal_states(legacy, "state_smiles"),
                "ENDPOINT_MISMATCH": _terminal(compact).get("derived_precursor")
                == _terminal(legacy).get("derived_precursor"),
                "EXECUTION_MISMATCH": (
                    _terminal(compact).get("formal_execute"),
                    _terminal(compact).get("endpoint_exact"),
                )
                == (
                    _terminal(legacy).get("formal_execute"),
                    _terminal(legacy).get("endpoint_exact"),
                ),
                "PRODUCT_DUPLICATED": _product_occurs_once(compact),
            }
            for code, passed in checks.items():
                if not passed:
                    failures.append({"code": code, "id": identifier})
            for result in _tool_results(compact)[:-1]:
                leaked = sorted(forbidden.intersection(result))
                if leaked:
                    failures.append(
                        {"code": "VISIBLE_AUDIT_FIELD", "id": identifier, "keys": leaked}
                    )
                if "current_state_smiles" not in result:
                    failures.append(
                        {"code": "CURRENT_STATE_MISSING", "id": identifier}
                    )
            terminal_leaked = sorted(forbidden.intersection(_terminal(compact)))
            if terminal_leaked:
                failures.append(
                    {
                        "code": "VISIBLE_TERMINAL_AUDIT_FIELD",
                        "id": identifier,
                        "keys": terminal_leaked,
                    }
                )

            _, compact_audit = encode_assistant_only_conversation(
                tokenizer, compact, max_length=1_000_000
            )
            _, legacy_audit = encode_assistant_only_conversation(
                tokenizer, legacy, max_length=1_000_000
            )
            compact_tokens += int(compact_audit["raw_length"])
            legacy_tokens += int(legacy_audit["raw_length"])
            compact_max_tokens = max(
                compact_max_tokens, int(compact_audit["raw_length"])
            )
            legacy_max_tokens = max(
                legacy_max_tokens, int(legacy_audit["raw_length"])
            )

    ratio = compact_tokens / max(legacy_tokens, 1)
    if compact_ids != legacy_ids:
        failures.append({"code": "ID_ORDER_OR_COVERAGE_MISMATCH"})
    if ratio > args.max_token_ratio:
        failures.append(
            {
                "code": "TOKEN_RATIO_EXCEEDED",
                "actual": ratio,
                "maximum": args.max_token_ratio,
            }
        )
    report = {
        "artifact_type": "compact_full_state_equivalence_gate_v1",
        "compact": str(args.compact),
        "compact_sha256": _sha256(args.compact),
        "legacy": str(args.legacy),
        "legacy_prefix_rows": rows,
        "rows": rows,
        "stable_ids_identical": compact_ids == legacy_ids,
        "compact_tokens": compact_tokens,
        "legacy_tokens": legacy_tokens,
        "compact_max_tokens": compact_max_tokens,
        "legacy_max_tokens": legacy_max_tokens,
        "serialization_truncation_count": 0,
        "token_ratio": ratio,
        "max_token_ratio": args.max_token_ratio,
        "failure_count": len(failures),
        "failures": failures[:100],
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
