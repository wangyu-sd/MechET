#!/usr/bin/env python3
"""Evaluate MechET model generations against gold reactants (real benchmark)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.metrics import build_eval_report, score_mech_et_prediction


def _load_data(path: Path, limit: int) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("id") or len(rows))
            rows[rid] = row
            if limit and len(rows) >= limit:
                break
    return rows


def _load_predictions(path: Path) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            preds[str(rec.get("id"))] = rec
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/mechet_sft/valid.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/mechet_eval/model_eval_summary.json")
    parser.add_argument("--cases-out", type=Path, default=None)
    args = parser.parse_args()

    data = _load_data(args.data, args.limit)
    preds = _load_predictions(args.predictions)

    cases = []
    missing = 0
    for rid, row in data.items():
        if rid not in preds:
            missing += 1
            continue
        rec = preds[rid]
        raw_candidates = rec.get("raw_generations") or rec.get("candidates") or []
        cases.append(
            score_mech_et_prediction(
                row,
                str(rec.get("prediction") or ""),
                mode="model",
                candidates=list(raw_candidates) if isinstance(raw_candidates, list) else None,
            )
        )

    manifest_path = args.predictions.with_suffix(args.predictions.suffix + ".manifest.json")
    extra = {}
    if manifest_path.exists():
        extra["inference_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

    report = build_eval_report(
        cases,
        mode="model",
        data_path=str(args.data),
        predictions_path=str(args.predictions),
        missing_predictions=missing,
        extra=extra,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    case_path = args.cases_out or args.out.parent / "model_eval_cases.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
