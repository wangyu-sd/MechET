#!/usr/bin/env python3
"""Apply one deterministic verifier-guided repair to proof generations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_diagnostics import diagnose_proof, repair_proof_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            prediction = str(row.get("prediction") or "")
            certificate = diagnose_proof(prediction)
            repair = repair_proof_once(prediction, certificate)
            if certificate is None:
                counts["already_valid"] += 1
            else:
                counts[f"failure:{certificate.code}"] += 1
            if repair.changed:
                counts["changed"] += 1
            if repair.execute_ok:
                counts["execute_ok_after"] += 1
            output = dict(row)
            output.update(
                {
                    "prediction_before_repair": prediction,
                    "prediction": repair.repaired_text,
                    "repair_changed": repair.changed,
                    "repair_execute_ok": repair.execute_ok,
                    "failure_certificate": (
                        None
                        if certificate is None
                        else {
                            "code": certificate.code,
                            "stage": certificate.stage,
                            "edge": certificate.edge,
                            "message": certificate.message,
                            "repairable": certificate.repairable,
                            "repair_lines": list(certificate.repair_lines),
                        }
                    ),
                }
            )
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    summary = {"n": len(rows), **dict(sorted(counts.items()))}
    args.out.with_suffix(args.out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
