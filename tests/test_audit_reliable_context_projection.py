import json
from types import SimpleNamespace

import pytest

from mechet.endpoints import split_precursor_endpoints
from mechet.forward_expert import verify_electron_step
from mechet.natural_language_electron_flow import render_event_arguments
from scripts import audit_reliable_context_projection as audit


def test_context_projection_preserves_retained_component_serialization():
    assert audit.strip_context_components("[CH4:1].[Na+:2]", {2}) == "[CH4:1]"
    with pytest.raises(audit.ContextChemicallyActive, match=r"decision 4.*\[2\].*\[1\]"):
        audit.strip_context_components("[CH3:1][Na:2]", {2}, decision_index=4)


def test_counterfactual_keeps_real_electron_move_after_context_removal(monkeypatch):
    root = "[O:1]=[C:2]([OH:3])[CH3:4].[O-:5][CH2:6][CH3:7]"
    with_context = root + ".[Na+:8]"
    moves = [
        {"source": {"kind": "BOND", "atoms": [1, 2]},
         "sink": {"kind": "ATOM", "atoms": [1]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [5]},
         "sink": {"kind": "BOND", "atoms": [2, 5]}, "electrons": 2},
    ]
    event_args = render_event_arguments(with_context, moves)
    event_result = verify_electron_step(with_context, moves)
    assert event_result["ok"]
    after_event = event_result["state_smiles"]
    decisions = [
        {"name": "import_fragments", "arguments": {"fragments": [
            {"smiles": "[Na+]", "count": 1, "purpose": "endpoint_context"},
        ]}},
        {"name": "apply_electron_flow", "arguments": event_args},
        {"name": "finish_trace", "arguments": {}},
    ]
    states = [root, with_context, after_event, after_event]
    monkeypatch.setattr(
        audit, "replay_reference",
        lambda *_args, **_kwargs: SimpleNamespace(
            nodes=[SimpleNamespace(state=state) for state in states],
        ),
    )
    monkeypatch.setattr(
        audit, "decision_action",
        lambda row: (row["name"], row["arguments"], {}),
    )
    structural = split_precursor_endpoints(
        audit.strip_context_components(after_event, {8}), root,
    ).structural
    counts = audit.project_reference(
        {"target_smiles": root, "structural_precursor": structural}, decisions,
    )
    assert counts["context_copies_removed"] == 1
    assert counts["projected_events_ok"] == 1
    assert counts["projected_structural_exact"] == 1


def test_audit_separates_chemically_active_context_from_projection_failure(
    monkeypatch, tmp_path,
):
    source = tmp_path / "source.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    source_manifest = tmp_path / "source_manifest.json"
    decision_manifest = tmp_path / "decision_manifest.json"
    source.write_text('{"source_id":"case_1"}\n', encoding="utf-8")
    decisions.write_text(
        '{"source_id":"case_1","metadata":{"decision_index":0}}\n',
        encoding="utf-8",
    )
    source_manifest.write_text(
        json.dumps({"splits": {"valid": {"rows": 1, "sha256": "abc"}}}),
        encoding="utf-8",
    )
    decision_manifest.write_text(
        json.dumps({"splits": {"valid": {"output_sha256": "abc"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "sha256", lambda _path: "abc")
    monkeypatch.setattr(audit, "read_selected", lambda *_args: [{"source_id": "case_1"}])
    monkeypatch.setattr(
        audit, "project_reference",
        lambda *_args: (_ for _ in ()).throw(
            audit.ContextChemicallyActive("context merged at decision 2")
        ),
    )
    report = audit.audit(
        source=source, source_manifest=source_manifest,
        decisions=decisions, decision_manifest=decision_manifest,
        n_reactions=1,
    )
    assert report["counts"] == {"context_chemically_active": 1, "reactions": 1}
    assert report["failures"][0]["source_id"] == "case_1"
