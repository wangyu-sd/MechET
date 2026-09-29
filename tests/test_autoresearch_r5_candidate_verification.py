"""The R5 producer preserves frozen external slots and does not leak labels."""

from __future__ import annotations

import json

import pytest

from scripts.autoresearch.run_r5_candidate_verification import (
    PolicyContextExceeded, read_public_slots, rollout_candidate,
    verification_rows,
)
from scripts.autoresearch.score_r5_external import _read_verifications
from scripts.autoresearch.stratified_manifest import digest, product_key


def _fixture_source(tmp_path, monkeypatch):
    from scripts.autoresearch import run_r5_candidate_verification as producer

    product = product_key("CO")
    candidate = product_key("CBr.[OH-]")
    source = tmp_path / "r5_external_predictions.jsonl"
    row = {
        "product_smiles": product,
        "model_input": {"product_smiles": product},
        "candidates": [
            {"rank": rank,
             "canonical_precursors": candidate if rank == 1 else None,
             "smiles_status": "valid_smiles" if rank == 1 else "missing",
             "recorded_reference_status": "recorded_reference" if rank == 1 else "unrecorded"}
            for rank in range(1, 6)],
    }
    source.write_text(json.dumps(row) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "products": 1, "ranks_per_product": 5,
        "target_semantics": "retrosynthetic_precursor_set"}))
    (tmp_path / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "evaluation_allowed": True, "training_allowed": False}))
    monkeypatch.setattr(producer, "verify_evaluation_source",
                        lambda path, name: digest(path))
    return source, product, candidate


def test_r5_producer_reads_only_public_slots_and_scorer_accepts_shape(tmp_path, monkeypatch):
    source, product, candidate = _fixture_source(tmp_path, monkeypatch)
    products, cohort_sha = read_public_slots(source, expected_products=1)
    assert len(products) == 1 and products[0]["slots"][0]["valid"]
    assert "recorded_reference_status" not in str(products)

    def policy(prompt, decision):
        assert decision == 0
        assert product in prompt and candidate in prompt
        assert "recorded_reference" not in prompt
        return "finish_trace", {}, "", "<tool_call>finish_trace</tool_call>"

    rows, raw = verification_rows(products, policy, max_attempts=2,
                                  max_decisions=1, max_tool_calls=4, max_imports=2)
    assert len(rows) == 5 and [row["rank"] for row in rows] == list(range(1, 6))
    assert len(raw) == 2
    assert rows[0]["model_input"] == {
        "product_smiles": product, "proposed_precursors": candidate}
    assert rows[0]["verification_status"] == "completed"
    assert all(row["verification_status"] == "skipped_invalid_or_missing"
               and row["attempts"] == [] for row in rows[1:])

    output = tmp_path / "verification.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
        "verification_sha256": digest(output), "cohort_sha256": cohort_sha,
        "condition": "base",
        "input_fields": ["product_smiles", "proposed_precursors"],
        "verification_semantics": "candidate_conditioned_executor_trace_v1",
        "checkpoint_identifier": "test-adapter", "checkpoint_sha256": "a" * 64,
        "max_attempts": 2,
    }))
    slots = {(product, rank): {"smiles_status": "valid_smiles" if rank == 1 else "missing",
                              "canonical_precursors": candidate if rank == 1 else None}
             for rank in range(1, 6)}
    scored, _ = _read_verifications(output, condition="base",
                                    cohort_sha256=cohort_sha, slots=slots)
    assert len(scored) == 5


def test_r5_context_limit_is_recorded_without_truncating_history():
    product = product_key("CO")
    candidate = product_key("CBr.[OH-]")

    def too_long(prompt, decision):
        raise PolicyContextExceeded("over frozen budget")

    attempt, raw = rollout_candidate(product, candidate, too_long,
                                     max_decisions=2, max_tool_calls=4,
                                     max_imports=1)
    assert attempt["termination_reason"] == "context_budget"
    assert attempt["final_result"] is None
    assert raw[0]["parse_error"] == "CONTEXT_BUDGET_EXCEEDED"


def test_r5_rejects_changed_slot_denominator(tmp_path, monkeypatch):
    source, _, _ = _fixture_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="frozen evaluation cohort"):
        read_public_slots(source)
