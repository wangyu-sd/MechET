#!/usr/bin/env python3
"""Create executor-labelled controlled proof corruptions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.metrics import extract_product_from_user
from mechet.proof_curriculum import (
    DEFAULT_CORRUPTIONS,
    build_corruption_set,
    proof_text_from_row,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_messages(row: dict) -> list[dict[str, str]]:
    return [
        {"role": str(item.get("role")), "content": str(item.get("content") or "")}
        for item in row.get("messages") or []
        if item.get("role") != "assistant"
    ]


def product_from_row(row: dict) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            value = extract_product_from_user(str(message.get("content") or ""))
            if value:
                return value
    return str((row.get("metadata") or {}).get("product") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--types",
        nargs="*",
        default=list(DEFAULT_CORRUPTIONS),
    )
    parser.add_argument("--include-valid-controls", action="store_true")
    args = parser.parse_args()

    corruption_types = list(args.types)
    if args.include_valid_controls:
        corruption_types.extend(["COMMUTING_ORDER_CONTROL", "STATE_RENAME_CONTROL"])
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    written = sources = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            proof = proof_text_from_row(row)
            if not proof:
                continue
            items = build_corruption_set(
                proof,
                source_id=str(row.get("id", row_index)),
                corruption_types=corruption_types,
                seed=args.seed + 1009 * row_index,
            )
            if items:
                sources += 1
            for item in items:
                payload = item.to_dict()
                payload["product"] = product_from_row(row)
                payload["prompt_messages"] = prompt_messages(row)
                payload["source_metadata"] = dict(row.get("metadata") or {})
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                counts[item.corruption_type] += 1
                written += 1
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "n_input": len(rows),
        "sources_with_corruptions": sources,
        "corruptions_written": written,
        "counts": dict(sorted(counts.items())),
        "seed": args.seed,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
