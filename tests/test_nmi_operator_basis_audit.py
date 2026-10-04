import hashlib
import json

import pytest

from scripts.audit_nmi_operator_basis import audit_basis


def test_streaming_operator_audit_counts_and_verifies_source(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"source_id": "r1", "metadata": {"executor_replayed": True, "endpoint_source": "environment_owned_trace", "execution_composition_signature": "a" * 64, "execution_primitive_signatures": ["p1"], "n_trace_steps": 1, "n_trace_moves": 1}},
        {"source_id": "r2", "metadata": {"executor_replayed": True, "endpoint_source": "environment_owned_trace", "execution_composition_signature": "b" * 64, "execution_primitive_signatures": ["p1", "p2"], "n_trace_steps": 2, "n_trace_moves": 2}},
        {"source_id": "r3", "metadata": {"executor_replayed": True, "endpoint_source": "environment_owned_trace", "execution_composition_signature": "a" * 64, "execution_primitive_signatures": ["p1"], "n_trace_steps": 1, "n_trace_moves": 1}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    report = audit_basis(path, expected_sha256=sha, expected_rows=3, verify_first=0)
    assert report["rows"] == report["unique_source_ids"] == 3
    assert report["total_execution_steps"] == report["total_electron_moves"] == 4
    assert report["unique_local_primitives"] == 2
    assert report["unique_complete_move_compositions"] == 2
    assert report["primitive_reaction_frequency_bins"]["one"] == 1
    assert report["vocabulary_growth"][-1]["unique_primitives"] == 2
    with pytest.raises(ValueError, match="source contract mismatch"):
        audit_basis(path, expected_sha256="0" * 64, expected_rows=3, verify_first=0)


def test_operator_audit_rejects_unreplayed_and_duplicate_ids(tmp_path):
    path = tmp_path / "train.jsonl"
    bad = {"source_id": "r1", "metadata": {"executor_replayed": False}}
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="non-replayed"):
        audit_basis(path, expected_sha256="0" * 64, expected_rows=1, verify_first=0)
