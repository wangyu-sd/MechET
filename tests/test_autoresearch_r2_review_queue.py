"""The R2 executor-valid pool is a review queue, never a negative source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch import prepare_r2_executor_valid_review_queue as queue
from scripts.autoresearch.stratified_manifest import digest


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("counterfactual,expected_count", [("O", 1), ("N", 0)])
def test_review_queue_does_not_label_or_reuse_recorded_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    counterfactual: str, expected_count: int,
) -> None:
    positives = tmp_path / "positives.jsonl"
    positives.write_text("{}\n")
    official = tmp_path / "official.jsonl"
    _jsonl(official, [{"target_smiles": "C", "structural_precursor": "N"}])
    official_manifest = tmp_path / "official_manifest.json"
    official_manifest.write_text(json.dumps({
        "splits": {"test": {"output_sha256": digest(official)}}
    }))
    traces = tmp_path / "traces.jsonl"
    _jsonl(traces, [{"source_id": "r1", "target_smiles": "C",
                     "structural_precursor": "N"}])
    histories = tmp_path / "histories.jsonl"
    _jsonl(histories, [{"source_id": "r1"}])
    r3_manifest = tmp_path / "r3_manifest.json"
    r3_manifest.write_text(json.dumps({
        "trace_sha256": digest(traces), "history_sha256": digest(histories)
    }))
    monkeypatch.setattr(queue, "_load_positives", lambda _: {
        "C": {"proposal_id": "positive:r1",
              "model_input": {"proposed_precursors": "N"}}
    })
    monkeypatch.setattr(queue, "_terminal_counterfactual", lambda *_: ({
        "proposed_precursors": counterfactual,
        "counterfactual_successor": "[CH4:1]",
    }, "candidate"))

    output = tmp_path / "review_queue"
    report = queue.build(positives, official, official_manifest,
                         traces, histories, r3_manifest, output)
    rows = [json.loads(line) for line in
            (output / "review_candidates.jsonl").read_text().splitlines()]
    status = json.loads((output / "ARTIFACT_STATUS.json").read_text())
    assert report["candidate_count"] == expected_count == len(rows)
    assert report["evaluation_allowed"] is False
    assert report["training_allowed"] is False
    assert status["evidence_audited"] is False
    assert status["human_review_required"] is True
    if rows:
        assert rows[0]["chemical_negative_label"] is None
        assert rows[0]["review_status"] == "pending_independent_chemistry_review"
        assert rows[0]["model_input"]["proposed_precursors"] == "O"
    with pytest.raises(FileExistsError):
        queue.build(positives, official, official_manifest,
                    traces, histories, r3_manifest, output)
