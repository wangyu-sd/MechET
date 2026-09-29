"""R3 model-facing export omits the private repair/reference target."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.export_r3_queries import export
from scripts.autoresearch.stratified_manifest import digest


def _source(tmp_path: Path) -> Path:
    directory = tmp_path / "frozen"
    directory.mkdir()
    path = directory / "r3_corruptions.jsonl"
    with path.open("w") as stream:
        for index in range(288):
            stream.write(json.dumps({
                "reaction_id": f"test:{index}", "source_split": "test",
                "target_smiles": "CCO",
                "model_visible": {
                    "target_smiles": "CCO", "prefix_actions": [],
                    "corrupted_action": {"name": "apply_electron_flow", "arguments": {}},
                    "executor_result": {"ok": False, "code": "INVALID_MOVE"},
                },
                "private_reference": {
                    "correct_action": {"name": "apply_electron_flow"},
                    "expected_precursor": "CC.O",
                    "first_failure_index": 1,
                },
            }, sort_keys=True) + "\n")
    (directory / "manifest.json").write_text(json.dumps({"cohort_sha256": digest(path)}))
    (directory / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "cohort_sha256": digest(path), "evaluation_allowed": True}))
    return path


def test_r3_queries_preserve_all_cases_without_reference_leak(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "queries"
    report = export(source, output)
    rows = [json.loads(line) for line in (output / "r3_queries.jsonl").read_text().splitlines()]
    assert report["cases"] == len({row["case_id"] for row in rows}) == 288
    assert all(set(row) == {"artifact_type", "case_id", "model_input"} for row in rows)
    assert all("private_reference" not in json.dumps(row) for row in rows)
    assert all("expected_precursor" not in json.dumps(row) for row in rows)
    status = json.loads((output / "ARTIFACT_STATUS.json").read_text())
    assert status["inference_allowed"] and not status["training_allowed"]
    with pytest.raises(FileExistsError):
        export(source, output)


def test_r3_queries_reject_nested_reference_leak(tmp_path: Path) -> None:
    source = _source(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["model_visible"]["corrupted_action"]["expected_successor"] = "CO"
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    for suffix in ("manifest.json", "ARTIFACT_STATUS.json"):
        sidecar = source.parent / suffix
        metadata = json.loads(sidecar.read_text())
        metadata["cohort_sha256"] = digest(source)
        sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="leaks reference fields"):
        export(source, tmp_path / "leaked")
    assert not (tmp_path / "leaked").exists()
