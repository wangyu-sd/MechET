import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_reliable_import_supervision import audit


def _row(identifier: str, source: str, tool: str, fragments=None):
    return {
        "id": identifier, "source_id": source,
        "messages": [{}, {}, {"tool_calls": [{"function": {
            "name": tool, "arguments": {"fragments": fragments or []},
        }}]}],
    }


def _fixture(tmp_path: Path):
    decisions = tmp_path / "valid.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "manifest.json"
    rows = [
        _row("a::0", "a", "import_fragments", [
            {"smiles": "O", "count": 1, "purpose": "electron_participant"},
        ]),
        _row("a::1", "a", "import_fragments", [
            {"smiles": "CCO", "count": 2, "purpose": "endpoint_context"},
        ]),
        _row("a::2", "a", "finish_trace"),
    ]
    decisions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    predictions.write_text("".join(json.dumps(row) + "\n" for row in [
        {"key": "a::0", "decision_type": "import", "correct_tool": True,
         "import_fragment_exact": True},
        {"key": "a::1", "decision_type": "import", "correct_tool": True,
         "import_fragment_exact": False},
        {"key": "a::2", "decision_type": "finish", "correct_tool": True},
    ]), encoding="utf-8")
    manifest.write_text(json.dumps({
        "splits": {"valid": {"output_sha256": hashlib.sha256(decisions.read_bytes()).hexdigest()}},
        "decision_rows": {"valid": 3}, "reaction_denominator": {"valid": 1},
    }), encoding="utf-8")
    return decisions, manifest, predictions


def test_import_supervision_separates_electron_and_context_roles(tmp_path: Path):
    decisions, manifest, predictions = _fixture(tmp_path)
    report = audit(
        decisions=decisions, manifest=manifest, split="valid",
        predictions=predictions,
    )
    counts = report["counts"]
    assert counts["import_decisions"] == 2
    assert counts["fragment_copies_electron_participant"] == 1
    assert counts["fragment_copies_endpoint_context"] == 2
    assert counts["reactions_with_endpoint_context"] == 1
    assert report["prediction_by_gold_role"] == {
        "electron_participant_n": 1,
        "electron_participant_correct_tool": 1,
        "electron_participant_fragment_exact": 1,
        "endpoint_context_n": 1,
        "endpoint_context_correct_tool": 1,
        "endpoint_context_fragment_exact": 0,
    }


def test_import_supervision_refuses_wrong_manifest_hash(tmp_path: Path):
    decisions, manifest, _ = _fixture(tmp_path)
    decisions.write_text(decisions.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        audit(decisions=decisions, manifest=manifest, split="valid")
