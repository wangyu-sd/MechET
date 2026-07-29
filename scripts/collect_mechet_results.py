#!/usr/bin/env python3
"""Export MechET eval summary to ORBIT single-step table TSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.metrics import single_step_table_row

COLUMNS = [
    "method",
    "backbone",
    "train_examples",
    "test_examples",
    "top1",
    "top5",
    "valid_precursors",
    "parse_rate",
    "compile_rate",
    "execute_rate",
    "endpoint_rate",
    "repair_success",
    "status",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/mechet_eval/main_table.tsv")
    parser.add_argument("--method", default="MechET v3 (MECH_ET CoT)")
    parser.add_argument("--backbone", default="Qwen3-8B")
    parser.add_argument("--train-n", default="258233")
    parser.add_argument("--test-n", default="")
    parser.add_argument("--status", default="COMPLETED")
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    test_n = args.test_n or str((payload.get("totals") or {}).get("n", 0))
    row = single_step_table_row(
        args.method,
        backbone=args.backbone,
        train_n=args.train_n,
        test_n=test_n,
        agg=payload,
        status=args.status,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps({"wrote": str(args.out), "row": row}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
