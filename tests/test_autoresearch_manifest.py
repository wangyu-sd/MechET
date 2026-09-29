import json
from pathlib import Path

import pytest

from scripts.autoresearch.stratified_manifest import (
    allocate, digest, freeze, grouped_decisions, verify_evaluation_source,
)


def _row(reaction: str, decision: int, kind: str) -> dict:
    call = ({"function": {"name": "apply_electron_flow", "arguments": {
        "electron_flow": [{"source": "A", "destination": "B"}]}}}
            if kind == "event" else {"function": {"name": "finish_trace", "arguments": {}}})
    return {
        "id": f"{reaction}:{decision}", "source_id": reaction,
        "target_smiles": "CCO" if reaction.endswith("0") else "CCN",
        "metadata": {"reaction_id": reaction, "decision_type": kind,
                     "executor_replayed": True},
        "messages": [{"role": "assistant", "tool_calls": [call]}],
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_allocation_minimum_then_proportional() -> None:
    counts = {("Q1",): 7, ("Q2",): 3, ("Q3",): 1}
    result = allocate(counts, 8)
    assert sum(result.values()) == 8
    assert all(result[cell] >= 1 for cell in counts)
    assert all(result[cell] <= counts[cell] for cell in counts)


def test_noncontiguous_reaction_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write(path, [_row("r0", 0, "finish"), _row("r1", 0, "finish"),
                  _row("r0", 1, "finish")])
    with pytest.raises(ValueError, match="noncontiguous"):
        list(grouped_decisions(path))


def test_engineering_freeze_deterministic_and_immutable(tmp_path: Path) -> None:
    flower = tmp_path / "flower.jsonl"
    mech = tmp_path / "mech.jsonl"
    _write(flower, [_row(f"f{i}", 0, "event") for i in range(20)])
    _write(mech, [_row(f"m{i}", 0, "event") for i in range(20)])
    config = {
        "campaign_id": "toy", "seed": 17,
        "sources": {"flower": {"train": str(flower)},
                    "mech_uspto_31k": {"train": str(mech)},
                    "curated": {"train": None}},
        "evaluation_sources": {"r1": None},
    }
    first = freeze(config, tmp_path, tmp_path / "first", engineering=True)
    second = freeze(config, tmp_path, tmp_path / "second", engineering=True)
    assert first["files"]["engineering"]["rows"] == 32
    assert first["files"]["engineering"]["train_sha256"] == second["files"]["engineering"]["train_sha256"]
    with pytest.raises(FileExistsError):
        freeze(config, tmp_path, tmp_path / "first", engineering=True)
    with pytest.raises(FileNotFoundError, match="evaluation source"):
        freeze(config, tmp_path, tmp_path / "scientific", engineering=False)


def test_deprecated_source_is_rejected(tmp_path: Path) -> None:
    flower = tmp_path / "flower" / "train.jsonl"
    mech = tmp_path / "mech" / "train.jsonl"
    _write(flower, [_row(f"f{i}", 0, "event") for i in range(20)])
    _write(mech, [_row(f"m{i}", 0, "event") for i in range(20)])
    (mech.parent / "ARTIFACT_STATUS.json").write_text(json.dumps({"training_allowed": False}))
    config = {"campaign_id": "toy", "seed": 17,
              "sources": {"flower": {"train": str(flower)},
                          "mech_uspto_31k": {"train": str(mech)},
                          "curated": {"train": None}},
              "evaluation_sources": {}}
    with pytest.raises(ValueError, match="forbids training"):
        freeze(config, tmp_path, tmp_path / "out", engineering=True)


def test_eval_source_rejects_superseded_or_drifted_cohort(tmp_path: Path) -> None:
    cohort = tmp_path / "r3_corruptions.jsonl"
    _write(cohort, [{"reaction_id": "r0", "target_smiles": "CCO"}])
    with pytest.raises(ValueError, match="frozen cohort manifest"):
        verify_evaluation_source(cohort)
    (tmp_path / "manifest.json").write_text(json.dumps({"cohort_sha256": digest(cohort)}))
    status = tmp_path / "ARTIFACT_STATUS.json"
    status.write_text(json.dumps({"evaluation_allowed": False}))
    with pytest.raises(ValueError, match="forbidden"):
        verify_evaluation_source(cohort)
    status.write_text(json.dumps({"cohort_sha256": digest(cohort)}))
    with pytest.raises(ValueError, match="forbidden"):
        verify_evaluation_source(cohort)
    status.write_text(json.dumps({"evaluation_allowed": True,
                                  "cohort_sha256": digest(cohort)}))
    assert verify_evaluation_source(cohort) == digest(cohort)
    _write(cohort, [{"reaction_id": "r1", "target_smiles": "CCN"}])
    with pytest.raises(ValueError, match="hash differs"):
        verify_evaluation_source(cohort)


def test_eval_source_rejects_partial_scientific_cohort(tmp_path: Path) -> None:
    cohort = tmp_path / "r3_corruptions.jsonl"
    _write(cohort, [{"reaction_id": "r0", "target_smiles": "CCO"}])
    (tmp_path / "manifest.json").write_text(json.dumps({"cohort_sha256": digest(cohort)}))
    with pytest.raises(ValueError, match="expected 288"):
        verify_evaluation_source(cohort, name="r3_corruptions")
