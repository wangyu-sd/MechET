"""R2 scoring uses the whole frozen denominator, including failed executions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.freeze_r2_plausibility import NEGATIVE_CLASSES
from scripts.autoresearch.score_r2_plausibility import _auroc, _average_precision, score
from scripts.autoresearch.stratified_manifest import digest


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    cohort = cohort_dir / "r2_plausibility.jsonl"
    rows = []
    for index in range(400):
        rows.append({"proposal_id": f"p{index}", "product_smiles": f"[{index+1}CH4]",
                     "private_label": {"known_recorded_positive": True}})
    for class_index, name in enumerate(NEGATIVE_CLASSES):
        for index in range(50):
            rows.append({"proposal_id": f"n{class_index}:{index}",
                         "product_smiles": f"[{index+1}CH4]",
                         "private_label": {"negative_class": name}})
    _jsonl(cohort, rows)
    (cohort_dir / "manifest.json").write_text(json.dumps({
        "artifact_type": "r2_plausibility_manifest_v1",
        "cohort_sha256": digest(cohort), "positive_proposals": 400,
        "negative_proposals": 400}))
    (cohort_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "cohort_sha256": digest(cohort), "evaluation_allowed": True}))
    paths = []
    for condition in ("base", "mech"):
        path = tmp_path / f"{condition}.jsonl"
        scores = []
        for index, row in enumerate(rows):
            positive = index < 400
            if condition == "base":
                probability = 0.6 if positive else 0.4
                if row["private_label"].get("negative_class") == "executor_valid_wrong_successor":
                    probability = 0.8
            else:
                probability = 0.9 if positive else 0.1
            scores.append({"proposal_id": row["proposal_id"],
                           "support_probability": probability,
                           "compile_status": "failed" if index == 0 else "success",
                           "execute_status": "not_run" if index == 0 else "success"})
        _jsonl(path, scores)
        path.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
            "scores_sha256": digest(path), "cohort_sha256": digest(cohort),
            "condition": condition,
            "input_fields": ["product_smiles", "proposed_precursors"],
            "score_semantics": "probability_known_valid_proposal",
            "checkpoint_identifier": f"fixture-{condition}",
            "checkpoint_sha256": ("a" if condition == "base" else "b") * 64,
        }))
        paths.append(path)
    scientific = tmp_path / "scientific_freeze.json"
    scientific.write_text(json.dumps({
        "engineering_only": False,
        "mech_comparison_identifiable": True,
        "evaluation_hashes": {"r2_plausibility": digest(cohort)},
    }))
    return cohort, paths[0], paths[1], scientific


def test_auc_ties_and_average_precision() -> None:
    assert _auroc([1, 0], [0.5, 0.5]) == 0.5
    assert _average_precision([1, 0], [0.5, 0.5]) == 0.5
    assert _auroc([1, 0], [0.9, 0.1]) == 1
    assert _average_precision([1, 0], [0.9, 0.1]) == 1


def test_r2_scores_complete_pair_and_retains_failed_executor(tmp_path: Path) -> None:
    cohort, base, mech, scientific = _inputs(tmp_path)
    output = tmp_path / "result"
    result = score(cohort, base, mech, output, scientific, bootstrap_replicates=20)
    assert result["package"] == "r2" and result["status"] == "complete"
    assert result["base"]["rows"] == result["mech"]["rows"] == 800
    assert result["base"]["compile_fraction"] == 799 / 800
    assert result["base"]["auroc"] < result["mech"]["auroc"] == 1
    assert result["base"]["executor_valid_hard_negatives"][
        "false_positive_fraction_at_0_5"] == 1
    assert result["mech"]["executor_valid_hard_negatives"][
        "false_positive_fraction_at_0_5"] == 0
    assert result["paired_delta"]["auroc_product_cluster_bootstrap_95ci"][0] > 0
    assert (output / "result.json").is_file()
    with pytest.raises(FileExistsError):
        score(cohort, base, mech, output, scientific, bootstrap_replicates=20)


def test_r2_rejects_missing_score_and_hash_drift(tmp_path: Path) -> None:
    cohort, base, mech, scientific = _inputs(tmp_path)
    base.write_text("\n".join(base.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="provenance/hash mismatch"):
        score(cohort, base, mech, tmp_path / "bad", scientific, bootstrap_replicates=2)
    metadata = base.with_suffix(".jsonl.manifest.json")
    data = json.loads(metadata.read_text())
    data["scores_sha256"] = digest(base)
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="all 800"):
        score(cohort, base, mech, tmp_path / "bad", scientific, bootstrap_replicates=2)
    assert not (tmp_path / "bad").exists()
