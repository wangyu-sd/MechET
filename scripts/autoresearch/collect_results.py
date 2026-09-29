#!/usr/bin/env python3
"""Collect independent R1–R5 packages; missing results remain visible."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.evaluate_gate import promotion_evidence
from scripts.autoresearch.stratified_manifest import digest


PACKAGE_SOURCES = {
    "r1": ("r1_multi_reference",),
    "r2": ("r2_plausibility",),
    "r3": ("r3_corruptions",),
    "r4": ("r4_pmechdb_challenging", "r4_pmechrp_pathways", "r4_literature_cycles"),
    "r5": ("r5_external_predictions",),
}


def _validate_complete(package: str, payload: dict, scientific: dict | None,
                       scientific_sha: str | None) -> None:
    if scientific is None or scientific.get("engineering_only") is not False:
        raise ValueError(f"{package} cannot be complete without a scientific freeze")
    expected = {name: scientific["evaluation_hashes"][name]
                for name in PACKAGE_SOURCES[package]}
    if payload.get("evaluation_source_hashes") != expected:
        raise ValueError(f"{package} result does not match frozen evaluation sources")
    if payload.get("scientific_freeze_sha256") != scientific_sha:
        raise ValueError(f"{package} result does not match the scientific freeze")
    errors = payload.get("data_contract_errors")
    if not isinstance(errors, int) or isinstance(errors, bool) or errors < 0:
        raise ValueError(f"{package} result lacks a valid data-contract error count")
    denominators = payload.get("denominators")
    if (not isinstance(denominators, dict) or not denominators
            or any(not isinstance(count, int) or isinstance(count, bool) or count <= 0
                   for count in denominators.values())):
        raise ValueError(f"{package} result lacks positive integer denominators")
    if not isinstance(payload.get("metrics"), dict) or not payload["metrics"]:
        raise ValueError(f"{package} result lacks metrics")
    checkpoints = payload.get("model_checkpoint_sha256")
    if (not isinstance(checkpoints, dict) or set(checkpoints) != {"base", "mech"}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in checkpoints.values())):
        raise ValueError(f"{package} result lacks paired model provenance")


def collect(output: Path) -> dict:
    scientific = output / "scientific_freeze/manifests/freeze.json"
    freeze = json.loads(scientific.read_text()) if scientific.is_file() else None
    scientific_sha = digest(scientific) if freeze else None
    packages = {}
    for index in range(1, 6):
        path = output / f"r{index}/result.json"
        if not path.is_file():
            packages[f"r{index}"] = {"status": "missing", "path": str(path)}
            continue
        payload = json.loads(path.read_text())
        if payload.get("package") != f"r{index}" or payload.get("status") not in {"complete", "failed"}:
            raise ValueError(f"invalid result package: {path}")
        if payload["status"] == "complete":
            _validate_complete(f"r{index}", payload, freeze, scientific_sha)
        elif not isinstance(payload.get("failure_reason"), str) or not payload["failure_reason"].strip():
            raise ValueError(f"failed result package lacks a reason: {path}")
        packages[f"r{index}"] = {**payload, "path": str(path), "sha256": digest(path)}
    training_path = output / "jobs/scientific_training_metrics.json"
    training = json.loads(training_path.read_text()) if training_path.is_file() else None
    overlap_path = output / "manifests/train_eval_overlap_audit.json"
    overlap = json.loads(overlap_path.read_text()) if overlap_path.is_file() else None
    if overlap is not None:
        if (overlap.get("scientific_freeze_sha256") != scientific_sha
                or not isinstance(overlap.get("overlap_count"), int)
                or overlap.get("passed") is not (overlap["overlap_count"] == 0)):
            raise ValueError("train/evaluation overlap audit is invalid or drifted")
    scorecard = {
        "artifact_type": "autoresearch_smoke_scorecard_v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "scientific_manifest_sha256": scientific_sha,
        "packages": packages,
        "training": training,
        "train_eval_overlap_count": overlap.get("overlap_count") if overlap else None,
        "train_eval_overlap_passed": overlap.get("passed") if overlap else None,
    }
    scorecard["promotion"] = promotion_evidence(scorecard)
    return scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorecard = collect(args.output)
    semantic = {key: value for key, value in scorecard.items() if key != "collected_at"}
    semantic_hash = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    snapshot = args.output / "scorecards" / f"{semantic_hash}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        previous = json.loads(snapshot.read_text())
        if {key: value for key, value in previous.items() if key != "collected_at"} != semantic:
            raise ValueError("scorecard snapshot hash collided with different evidence")
        scorecard = previous
    else:
        with snapshot.open("x", encoding="utf-8") as stream:
            json.dump(scorecard, stream, indent=2, sort_keys=True)
            stream.write("\n")
    final = args.output / "scorecard.json"
    recommendation = scorecard["promotion"]["recommendation"]
    if recommendation != "INCOMPLETE":
        if final.exists():
            previous = json.loads(final.read_text())
            if {key: value for key, value in previous.items() if key != "collected_at"} != semantic:
                raise ValueError("final scorecard differs; create a new campaign revision")
        else:
            with final.open("x", encoding="utf-8") as stream:
                json.dump(scorecard, stream, indent=2, sort_keys=True)
                stream.write("\n")
    print(json.dumps({"scorecard_snapshot": str(snapshot),
                      "final_scorecard": str(final) if final.is_file() else None,
                      "recommendation": recommendation}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
