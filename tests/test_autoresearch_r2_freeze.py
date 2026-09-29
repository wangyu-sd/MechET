"""R2 intake is fail-closed; unit fixtures do not assert chemical validity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.freeze_r2_plausibility import NEGATIVE_CLASSES, freeze
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source


def _source(directory: Path, filename: str, rows: list[dict], *, positive: bool,
            negative_class: str | None = None, audited: bool = True) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / filename
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {"cohort_sha256": digest(path)}
    status = {"cohort_sha256": digest(path), "evaluation_allowed": False,
              "training_allowed": False}
    if positive:
        manifest["positive_proposals"] = len(rows)
        status["positive_source_allowed"] = True
    else:
        manifest.update({"negative_proposals": len(rows),
                         "negative_class": negative_class})
        status["evidence_audited"] = audited
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "ARTIFACT_STATUS.json").write_text(json.dumps(status))
    return path


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    positives = []
    for index in range(1, 401):
        product = f"[{index}CH4]"
        kind = "recorded_precursor" if index <= 200 else "documented_alternative"
        positives.append({
            "proposal_id": f"positive:{index}", "product_smiles": product,
            "source_split": "test", "model_input": {
                "product_smiles": product, "proposed_precursors": "O"},
            "private_label": {
                "known_recorded_positive": True,
                "proposal_record_ids": [f"record:{index}"],
                "evidence_kind": ("heldout_recorded_precursor" if index <= 200
                                  else "independent_heldout_recorded_alternative"),
                **({} if index <= 200 else {
                    "primary_record_ids": [f"primary:{index}"],
                    "primary_precursors": "N",
                }),
            },
            "strata": {"positive_type": kind},
        })
    positive_path = _source(tmp_path / "positive", "r2_positives.jsonl",
                            positives, positive=True)
    negatives = {}
    for class_index, name in enumerate(NEGATIVE_CLASSES):
        rows = []
        review_rows = []
        for index in range(1, 51):
            product = f"[{index}CH4]"
            label = {"negative_class": name, "evidence_kind": "mechanistic_contradiction"}
            if name == "missing_necessary_fragment":
                label.update({
                    "evidence_kind": "product_element_inventory_deficit_after_fragment_omission",
                    "scope": "closed_stated_precursor_inventory_atom_conservation",
                    "missing_element_counts": {"C": 1},
                })
            else:
                label["independent_negative_evidence"] = {
                    "kind": "literature_mechanistic_constraint",
                    "locator": f"fixture-record:{name}:{index}",
                    "sha256": "", "reviewer_id": "fixture-reviewer",
                    "artifact_path": "review.jsonl",
                    "record_id": f"review:{name}:{index}",
                }
            if name == "executor_valid_wrong_successor":
                label["executor_replay"] = {"accepted": True, "successor_smiles": "CC"}
            rows.append({
                "proposal_id": f"negative:{name}:{index}",
                "product_smiles": product, "source_split": "test",
                "model_input": {"product_smiles": product,
                                "proposed_precursors": "N" * (class_index + 1)},
                "private_label": label,
                "strata": {"negative_class": name},
            })
            if name != "missing_necessary_fragment":
                review_rows.append({
                    "record_id": f"review:{name}:{index}",
                    "proposal_id": f"negative:{name}:{index}",
                    "negative_class": name,
                    "product_smiles": product,
                    "proposed_precursors": "N" * (class_index + 1),
                    "evidence_kind": "literature_mechanistic_constraint",
                    "evidence_locator": f"fixture-record:{name}:{index}",
                    "reviewer_id": "fixture-reviewer",
                    "finding": "inconsistent_under_stated_conditions",
                    "rationale": "Unit fixture only; not chemical evidence.",
                    **({"executor_successor": "CC"}
                       if name == "executor_valid_wrong_successor" else {}),
                })
        if review_rows:
            directory = tmp_path / name
            directory.mkdir()
            review = directory / "review.jsonl"
            review.write_text("".join(json.dumps(item, sort_keys=True) + "\n"
                                      for item in review_rows))
            review_hash = digest(review)
            for row in rows:
                row["private_label"]["independent_negative_evidence"]["sha256"] = review_hash
        negatives[name] = _source(tmp_path / name, f"{name}.jsonl", rows,
                                  positive=False, negative_class=name)
    return positive_path, negatives


def test_r2_freeze_requires_all_eight_classes_and_frozen_source_hashes(tmp_path: Path) -> None:
    positives, negatives = _fixture(tmp_path)
    with pytest.raises(ValueError, match="eight distinct"):
        freeze(positives, {"missing_necessary_fragment": negatives["missing_necessary_fragment"]},
               tmp_path / "partial")
    assert not (tmp_path / "partial").exists()
    negatives["wrong_nucleophile"].write_text(
        negatives["wrong_nucleophile"].read_text() + "\n")
    with pytest.raises(ValueError, match="hash drifted"):
        freeze(positives, negatives, tmp_path / "drifted")
    assert not (tmp_path / "drifted").exists()


def test_r2_freeze_rejects_unsubstantiated_executor_valid_negative(tmp_path: Path) -> None:
    positives, negatives = _fixture(tmp_path)
    path = negatives["executor_valid_wrong_successor"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[0]["private_label"]["independent_negative_evidence"]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    metadata = path.parent / "manifest.json"
    status = path.parent / "ARTIFACT_STATUS.json"
    for source in (metadata, status):
        value = json.loads(source.read_text())
        value["cohort_sha256"] = digest(path)
        source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="independent evidence"):
        freeze(positives, negatives, tmp_path / "unsubstantiated")
    assert not (tmp_path / "unsubstantiated").exists()


def test_r2_freeze_rejects_drifted_review_bundle(tmp_path: Path) -> None:
    positives, negatives = _fixture(tmp_path)
    review = negatives["wrong_nucleophile"].parent / "review.jsonl"
    review.write_text(review.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash drifted"):
        freeze(positives, negatives, tmp_path / "drifted_review")
    assert not (tmp_path / "drifted_review").exists()


def test_r2_freeze_rejects_review_record_for_other_proposal(tmp_path: Path) -> None:
    positives, negatives = _fixture(tmp_path)
    path = negatives["wrong_nucleophile"]
    review = path.parent / "review.jsonl"
    review_rows = [json.loads(line) for line in review.read_text().splitlines()]
    review_rows[0]["proposed_precursors"] = "C"
    review.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in review_rows))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["private_label"]["independent_negative_evidence"]["sha256"] = digest(review)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    for filename in ("manifest.json", "ARTIFACT_STATUS.json"):
        metadata = path.parent / filename
        value = json.loads(metadata.read_text())
        value["cohort_sha256"] = digest(path)
        metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not substantiate proposal"):
        freeze(positives, negatives, tmp_path / "wrong_review")
    assert not (tmp_path / "wrong_review").exists()


def test_r2_complete_fixture_freezes_800_without_exposing_labels_to_model(tmp_path: Path) -> None:
    positives, negatives = _fixture(tmp_path)
    output = tmp_path / "r2"
    report = freeze(positives, negatives, output)
    cohort = output / "r2_plausibility.jsonl"
    assert report["rows"] == 800
    assert report["positive_proposals"] == report["negative_proposals"] == 400
    assert set(report["negative_strata"]) == set(NEGATIVE_CLASSES)
    assert verify_evaluation_source(cohort, name="r2_plausibility") == digest(cohort)
    rows = [json.loads(line) for line in cohort.read_text().splitlines()]
    assert len(rows) == len({row["proposal_id"] for row in rows}) == 800
    assert all(set(row["model_input"]) == {"product_smiles", "proposed_precursors"}
               for row in rows)
    assert all("private_label" not in row["model_input"] for row in rows)
    with pytest.raises(FileExistsError):
        freeze(positives, negatives, output)
