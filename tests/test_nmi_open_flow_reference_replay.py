import hashlib
import json

import pytest

from scripts.audit_nmi_open_flow_reference_replay import audit_split


MOVES = [
    {"source": {"kind": "BOND", "atoms": [1, 2]},
     "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
    {"source": {"kind": "LP", "atoms": [3]},
     "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
]


def _fixture(tmp_path, *, reference="[CH3:1][Br:3].[OH-:2]"):
    program = "<flow>\nOPEN_FLOW v1\nIMPORT [Br-:3]\nSTEP 0 " + json.dumps(MOVES) + "\nEXECUTE\n</flow>"
    row = {
        "source_id": "one", "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": reference,
        "messages": [{"role": "assistant", "content": program}],
    }
    data = (json.dumps(row) + "\n").encode()
    (tmp_path / "test.jsonl").write_bytes(data)
    manifest = {
        "artifact_type": "nmi_h2_open_flow_all_step_imports_v2",
        "rows": {"test": 1},
        "output_sha256": {"test": hashlib.sha256(data).hexdigest()},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_full_reference_replay_requires_strict_execution_and_endpoint(tmp_path):
    path = _fixture(tmp_path)
    result = audit_split(path, "test", progress_every=0)
    assert result["passed"] is True
    assert result["strict_execution_and_structural_endpoint_exact"] == 1


def test_full_reference_replay_reports_endpoint_mismatch_without_excluding_row(tmp_path):
    path = _fixture(tmp_path, reference="[CH3:1][Cl:3].[OH-:2]")
    result = audit_split(path, "test", progress_every=0)
    assert result["rows"] == 1
    assert result["failure_counts"] == {"ENDPOINT_MISMATCH": 1}
    assert result["passed"] is False


def test_full_reference_replay_rejects_file_hash_drift(tmp_path):
    path = _fixture(tmp_path)
    data_path = tmp_path / "test.jsonl"
    data_path.write_text(data_path.read_text().replace('"one"', '"two"'))
    with pytest.raises(ValueError, match="row count/SHA mismatch"):
        audit_split(path, "test", progress_every=0)
