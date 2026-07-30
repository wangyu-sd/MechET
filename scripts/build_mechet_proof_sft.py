#!/usr/bin/env python3
"""Compile MECH_ET v3 SFT JSONL into executable MECH_PROOF v1 SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_sft import convert_mech_et_row_to_proof_sft


def build_split(src: Path, dst: Path, *, limit: int = 0) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    skipped = 0
    reasons: Counter[str] = Counter()
    with src.open("r", encoding="utf-8") as reader, dst.open(
        "w", encoding="utf-8"
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            if limit and accepted >= limit:
                break
            try:
                row = json.loads(line)
                converted = convert_mech_et_row_to_proof_sft(row)
                writer.write(json.dumps(converted, ensure_ascii=False) + "\n")
                accepted += 1
            except Exception as exc:
                skipped += 1
                reasons[type(exc).__name__] += 1
                if skipped <= 10:
                    print(
                        f"[{src.name}:{line_number}] skipped: {exc}",
                        file=sys.stderr,
                    )
    return {
        "source": str(src),
        "output": str(dst),
        "accepted": accepted,
        "skipped": skipped,
        "skip_reasons": dict(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO / "data/mechet_sft",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/mechet_proof_sft",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    reports = []
    for split in args.splits:
        src = args.input_dir / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        reports.append(
            build_split(
                src,
                args.output_dir / f"{split}.jsonl",
                limit=args.limit,
            )
        )

    manifest = {
        "version": "mech_proof_sft_v1",
        "task_type": "mech_proof_retro",
        "answer_channel": "executor_derived",
        "splits": {
            Path(report["output"]).stem: report for report in reports
        },
        "accepted_total": sum(report["accepted"] for report in reports),
        "skipped_total": sum(report["skipped"] for report in reports),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
