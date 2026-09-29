"""Unmarked R3 localization queries must not expose the private cut point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.export_r3_unmarked_localization import export
from scripts.autoresearch.stratified_manifest import digest


def _action(label: int, *, with_result: bool = False) -> dict:
    row = {"name": "apply_electron_flow", "arguments": {"site": f"A{label:02d}"}}
    if with_result:
        row["result"] = {"ok": True, "current_state": "CC"}
    return row


def _source(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    source = directory / "r3_corruptions.jsonl"
    with source.open("w") as stream:
        for number in range(288):
            index = number % 3
            stream.write(json.dumps({
                "reaction_id": f"test:{number}", "source_split": "test",
                "target_smiles": "CCO",
                "model_visible": {"target_smiles": "CCO",
                                  "prefix_actions": [_action(step, with_result=True)
                                                     for step in range(index)],
                                  "corrupted_action": _action(100 + number),
                                  "executor_result": {"ok": False, "code": "INVALID_MOVE"}},
                "private_reference": {"first_failure_index": index,
                                      "corruption_kind": "rejected",
                                      "correct_action": _action(index),
                                      "suffix_actions": [_action(200 + number + step)
                                                         for step in range(2 - index)],
                                      "expected_successor": "CO",
                                      "expected_precursor": "C.O"},
            }, sort_keys=True) + "\n")
    (directory / "manifest.json").write_text(json.dumps({"cohort_sha256": digest(source)}))
    (directory / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "cohort_sha256": digest(source), "evaluation_allowed": True}))
    return source


def test_unmarked_r3_queries_flatten_all_actions_and_remove_feedback(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "queries"
    report = export(source, output)
    rows = [json.loads(line) for line in (output / "r3_unmarked_queries.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["case_id"] for row in rows}) == 288
    assert len(report["private_first_divergence_index_histogram"]) == 3
    assert all(set(row["model_input"]) == {"target_smiles", "candidate_actions"}
               for row in rows)
    assert all(len(row["model_input"]["candidate_actions"]) == 3 for row in rows)
    assert all(set(action) == {"name", "arguments"}
               for row in rows for action in row["model_input"]["candidate_actions"])
    assert all("first_failure_index" not in json.dumps(row) and
               "expected_precursor" not in json.dumps(row) and
               "executor_result" not in json.dumps(row) for row in rows)
    status = json.loads((output / "ARTIFACT_STATUS.json").read_text())
    assert status["localization_inference_allowed"] is True
    assert status["evaluation_allowed"] is False and status["training_allowed"] is False
    with pytest.raises(FileExistsError):
        export(source, output)


def test_unmarked_r3_rejects_inconsistent_private_index(tmp_path: Path) -> None:
    source = _source(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["private_reference"]["first_failure_index"] = 2
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    for name in ("manifest.json", "ARTIFACT_STATUS.json"):
        sidecar = source.parent / name
        payload = json.loads(sidecar.read_text())
        payload["cohort_sha256"] = digest(source)
        sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent private failure index"):
        export(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
