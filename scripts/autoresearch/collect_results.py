#!/usr/bin/env python3
"""Collect independent R1–R5 packages; missing results remain visible."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.evaluate_gate import promotion_evidence
from scripts.autoresearch.stratified_manifest import digest


def collect(output: Path) -> dict:
    packages = {}
    for index in range(1, 6):
        path = output / f"r{index}/result.json"
        if not path.is_file():
            packages[f"r{index}"] = {"status": "missing", "path": str(path)}
            continue
        payload = json.loads(path.read_text())
        if payload.get("package") != f"r{index}" or payload.get("status") not in {"complete", "failed"}:
            raise ValueError(f"invalid result package: {path}")
        packages[f"r{index}"] = {**payload, "path": str(path), "sha256": digest(path)}
    scientific = output / "scientific_freeze/manifests/freeze.json"
    freeze = json.loads(scientific.read_text()) if scientific.is_file() else None
    training_path = output / "jobs/scientific_training_metrics.json"
    training = json.loads(training_path.read_text()) if training_path.is_file() else None
    overlap_path = output / "manifests/train_eval_overlap_audit.json"
    overlap = json.loads(overlap_path.read_text()) if overlap_path.is_file() else None
    scorecard = {
        "artifact_type": "autoresearch_smoke_scorecard_v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "scientific_manifest_sha256": digest(scientific) if freeze else None,
        "packages": packages,
        "training": training,
        "train_eval_overlap_count": overlap.get("overlap_count") if overlap else None,
    }
    scorecard["promotion"] = promotion_evidence(scorecard)
    return scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorecard = collect(args.output)
    path = args.output / "scorecard.json"
    if path.exists():
        previous = json.loads(path.read_text())
        old = {key: value for key, value in previous.items() if key != "collected_at"}
        new = {key: value for key, value in scorecard.items() if key != "collected_at"}
        if old != new:
            raise ValueError("existing scorecard differs; preserve it and create a new campaign revision")
        scorecard = previous
    else:
        path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scorecard": str(path), "recommendation":
                      scorecard["promotion"]["recommendation"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
