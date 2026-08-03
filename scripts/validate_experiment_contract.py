#!/usr/bin/env python3
"""Validate matched scientific conditions before training or evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mechet.knowledge_ablation import (
    condition_contract_summary,
    read_jsonl,
    validate_alignment,
)
from train_tool_sft import tokenizer_audit, validate_rows


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    return name, Path(path)


def _is_direct(name: str) -> bool:
    return name in {"direct", "direct_textbook_rag"}


def _has_textbook(name: str) -> bool:
    return name in {
        "textbook",
        "trace_textbook_rag",
        "irrelevant",
        "trace_length_matched_irrelevant",
        "combined",
        "trace_text_plus_anchors",
        "direct",
        "direct_textbook_rag",
    }


def _assistant_mask(payload: dict[str, Any]) -> list[int]:
    for key, value in payload.items():
        if "assistant" in str(key).lower() and "mask" in str(key).lower():
            return [int(item) for item in value]
    return []


def _tokenizer_condition_summary(
    rows: list[dict[str, Any]], tokenizer, max_length: int
) -> dict[str, Any]:
    audit = tokenizer_audit(rows, tokenizer, max_length=max_length)
    input_tokens = int(audit["total_input_tokens"])
    supervised_tokens = int(audit["total_supervised_tokens"])
    return {
        **audit,
        "mean_input_tokens": input_tokens / max(len(rows), 1),
        "mean_supervised_tokens": supervised_tokens / max(len(rows), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, type=parse_condition)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--skip-tokenizer-audit",
        action="store_true",
        help="CI-only escape hatch for network-free synthetic contract tests",
    )
    parser.add_argument(
        "--strict-evidence-character-budget",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    conditions: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for name, path in args.condition:
        if name in conditions:
            raise ValueError(f"duplicate condition name: {name}")
        if not path.exists():
            raise FileNotFoundError(path)
        conditions[name] = read_jsonl(path)
        sources[name] = str(path)

    validate_alignment(conditions)
    summaries = {
        name: {
            **condition_contract_summary(rows),
            **validate_rows(rows, require_trace_owned=not _is_direct(name)),
        }
        for name, rows in conditions.items()
    }
    violations: list[dict[str, Any]] = []

    if len({value["stable_ids_sha256"] for value in summaries.values()}) != 1:
        violations.append({"code": "STABLE_ID_MISMATCH"})

    for name, rows in conditions.items():
        if name not in {"oracle", "label_oracle"}:
            leaked = [
                str(row.get("id") or "")
                for row in rows
                if bool((row.get("metadata") or {}).get("gold_label_query_used"))
            ]
            if leaked:
                violations.append(
                    {
                        "code": "GOLD_QUERY_LEAKAGE",
                        "condition": name,
                        "example_ids": leaked[:10],
                    }
                )

    evidence_values = {
        name: int(summary["textbook_context_characters"])
        for name, summary in summaries.items()
        if _has_textbook(name)
    }
    if args.strict_evidence_character_budget and evidence_values:
        nonzero = [value for value in evidence_values.values() if value > 0]
        if nonzero and any(value != nonzero[0] for value in nonzero):
            violations.append(
                {
                    "code": "EVIDENCE_CHARACTER_BUDGET_MISMATCH",
                    "values": evidence_values,
                }
            )

    tokenizer_summaries: dict[str, dict[str, Any]] = {}
    if not args.skip_tokenizer_audit:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("tokenizer audit requires mechet[agent]") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            revision=args.model_revision or None,
            trust_remote_code=True,
        )
        for name, rows in conditions.items():
            try:
                tokenizer_summaries[name] = _tokenizer_condition_summary(
                    rows, tokenizer, args.max_length
                )
            except Exception as exc:
                violations.append(
                    {
                        "code": "TOKENIZER_AUDIT_FAILED",
                        "condition": name,
                        "message": str(exc),
                    }
                )
        for name, summary in tokenizer_summaries.items():
            if int(summary.get("truncation_count", 0)):
                violations.append(
                    {
                        "code": "CONTEXT_TRUNCATION",
                        "condition": name,
                        "count": summary["truncation_count"],
                    }
                )
            if not bool(summary.get("assistant_mask_valid")):
                violations.append(
                    {"code": "ASSISTANT_MASK_INVALID", "condition": name}
                )

    supervised_totals = {
        name: int(value.get("total_supervised_tokens", 0))
        for name, value in tokenizer_summaries.items()
    }
    target_supervised = max(supervised_totals.values(), default=0)
    normalization = {
        name: (
            target_supervised / max(tokens, 1)
            if target_supervised
            else None
        )
        for name, tokens in supervised_totals.items()
    }

    report = {
        "artifact_type": "matched_experiment_contract",
        "scientific_contract": "causal_compositional_electron_flow_v2",
        "sources": sources,
        "conditions": summaries,
        "tokenizer_audit_skipped": args.skip_tokenizer_audit,
        "tokenizer_conditions": tokenizer_summaries,
        "supervised_token_totals": supervised_totals,
        "suggested_sampling_or_update_multiplier_to_match_supervised_tokens": normalization,
        "budget_policy": (
            "Do not claim raw token equality across direct and tool syntaxes. Freeze examples, "
            "optimizer updates, and report supervised-token-normalized compute; use the listed "
            "multipliers only if exact cumulative supervised-token matching is required."
        ),
        "violations": violations,
        "passed": not violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
