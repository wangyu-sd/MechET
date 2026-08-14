#!/usr/bin/env python3
"""Stream a Tool-SFT file and audit exact chat-template token lengths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.assistant_masking import (  # noqa: E402
    encode_assistant_only_conversation,
    percentile_nearest_rank,
)
from mechet.model_revision import (  # noqa: E402
    is_immutable_revision,
    resolve_loaded_model_revision,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = dict(yaml.safe_load(args.config.read_text(encoding="utf-8")) or {})
    training = dict(config.get("training") or {})
    input_path = args.input or Path(str(config["train_file"]))
    output_path = args.output or input_path.with_suffix(".tokenizer_audit.json")
    model_name = str(config["model_name_or_path"])
    revision = str(training.get("model_revision") or "").strip() or None
    if not revision or not is_immutable_revision(revision):
        raise ValueError("model_revision must be an immutable commit before audit")
    max_length = int(training.get("max_length", 12288))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )
    revision_info = resolve_loaded_model_revision(
        model_name_or_path=model_name,
        requested_revision=revision,
        tokenizer=tokenizer,
    )
    if revision_info.get("resolved_model_revision") != revision:
        raise ValueError("loaded tokenizer revision differs from pinned model revision")

    lengths: list[int] = []
    supervised: list[int] = []
    assistant_turns: list[int] = []
    mask_methods: set[str] = set()
    over_budget_ids: list[str] = []
    zero_supervision_ids: list[str] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if args.limit and len(lengths) >= args.limit:
                break
            row = json.loads(line)
            _, audit = encode_assistant_only_conversation(
                tokenizer,
                row,
                max_length=max_length,
            )
            identifier = str(row.get("id") or "")
            raw_length = int(audit["raw_length"])
            supervised_tokens = int(audit["supervised_tokens"])
            lengths.append(raw_length)
            supervised.append(supervised_tokens)
            assistant_turns.append(int(audit["assistant_turns"]))
            mask_methods.add(str(audit["mask_method"]))
            if bool(audit["exceeds_max_length"]):
                over_budget_ids.append(identifier)
            if supervised_tokens <= 0:
                zero_supervision_ids.append(identifier)

    if not lengths:
        raise ValueError("input contains no rows")
    report = {
        "artifact_type": "tool_sft_tokenizer_length_audit",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "config": str(args.config),
        "model_name_or_path": model_name,
        **revision_info,
        "rows": len(lengths),
        "total_input_tokens": sum(lengths),
        "total_supervised_tokens": sum(supervised),
        "max_input_tokens": max(lengths),
        "p50_input_tokens": percentile_nearest_rank(lengths, 0.50),
        "p95_input_tokens": percentile_nearest_rank(lengths, 0.95),
        "p99_input_tokens": percentile_nearest_rank(lengths, 0.99),
        "min_supervised_tokens": min(supervised),
        "max_assistant_turns": max(assistant_turns),
        "configured_max_length": max_length,
        "truncation_count": len(over_budget_ids),
        "over_budget_ids": over_budget_ids[:50],
        "zero_supervision_count": len(zero_supervision_ids),
        "zero_supervision_ids": zero_supervision_ids[:50],
        "assistant_mask_methods": sorted(mask_methods),
        "passed": not over_budget_ids and not zero_supervision_ids,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
