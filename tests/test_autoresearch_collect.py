"""A result file cannot promote itself by merely claiming `complete`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.collect_results import collect, main
from scripts.autoresearch.stratified_manifest import digest


def _frozen(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "campaign"
    scientific = output / "scientific_freeze/manifests/freeze.json"
    scientific.parent.mkdir(parents=True)
    scientific.write_text(json.dumps({
        "engineering_only": False,
        "evaluation_hashes": {"r2_plausibility": "a" * 64},
    }))
    result = output / "r2/result.json"
    result.parent.mkdir()
    result.write_text(json.dumps({
        "package": "r2", "status": "complete", "data_contract_errors": 0,
        "scientific_freeze_sha256": digest(scientific),
        "evaluation_source_hashes": {"r2_plausibility": "a" * 64},
        "denominators": {"proposals": 800},
        "metrics": {"auroc": 0.5},
        "model_checkpoint_sha256": {"base": "b" * 64, "mech": "c" * 64},
    }))
    return output, result


def test_missing_packages_are_incomplete_not_failed(tmp_path: Path) -> None:
    scorecard = collect(tmp_path)
    assert all(item["status"] == "missing" for item in scorecard["packages"].values())
    assert scorecard["promotion"]["recommendation"] == "INCOMPLETE"


def test_complete_package_must_match_frozen_science(tmp_path: Path) -> None:
    output, result = _frozen(tmp_path)
    scorecard = collect(output)
    assert scorecard["packages"]["r2"]["status"] == "complete"
    assert scorecard["promotion"]["recommendation"] == "INCOMPLETE"
    payload = json.loads(result.read_text())
    payload["evaluation_source_hashes"]["r2_plausibility"] = "d" * 64
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen evaluation sources"):
        collect(output)
    payload["evaluation_source_hashes"]["r2_plausibility"] = "a" * 64
    payload["denominators"] = {"proposals": 0}
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="positive integer denominators"):
        collect(output)


def test_complete_package_without_scientific_freeze_is_rejected(tmp_path: Path) -> None:
    output, _ = _frozen(tmp_path)
    (output / "scientific_freeze/manifests/freeze.json").unlink()
    with pytest.raises(ValueError, match="without a scientific freeze"):
        collect(output)


def test_overlap_audit_must_be_bound_and_consistent(tmp_path: Path) -> None:
    output, _ = _frozen(tmp_path)
    overlap = output / "manifests/train_eval_overlap_audit.json"
    overlap.parent.mkdir()
    overlap.write_text(json.dumps({"scientific_freeze_sha256": "e" * 64,
                                   "overlap_count": 0, "passed": True}))
    with pytest.raises(ValueError, match="overlap audit"):
        collect(output)


def test_incomplete_snapshot_does_not_lock_out_final_scorecard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = iter([
        {"collected_at": "first", "packages": {},
         "promotion": {"recommendation": "INCOMPLETE"}},
        {"collected_at": "second", "packages": {"r1": {"status": "complete"}},
         "promotion": {"recommendation": "HUMAN_REVIEW_FOR_SCALE"}},
    ])
    monkeypatch.setattr("scripts.autoresearch.collect_results.collect", lambda _: next(reports))
    monkeypatch.setattr("sys.argv", ["collect_results.py", "--output", str(tmp_path)])
    assert main() == 0
    assert not (tmp_path / "scorecard.json").exists()
    assert len(list((tmp_path / "scorecards").glob("*.json"))) == 1
    assert main() == 0
    assert (tmp_path / "scorecard.json").is_file()
    assert len(list((tmp_path / "scorecards").glob("*.json"))) == 2
