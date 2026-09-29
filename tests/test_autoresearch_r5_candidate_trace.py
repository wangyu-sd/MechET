"""R5 proposed endpoints never enter the executor as a private reference."""

from __future__ import annotations

from rdkit import Chem
import pytest

from mechet.natural_language_electron_flow import render_event_arguments
from scripts.autoresearch.r5_candidate_trace import CandidateTraceSession
from scripts.autoresearch.score_r5_external import _trace_support
from scripts.autoresearch.stratified_manifest import product_key


def _sn2_moves(mapped_state: str) -> list[dict]:
    molecule = Chem.MolFromSmiles(mapped_state)
    assert molecule is not None
    by_symbol = {atom.GetSymbol(): atom.GetAtomMapNum() for atom in molecule.GetAtoms()}
    carbon, oxygen, bromine = (by_symbol[name] for name in ("C", "O", "Br"))
    return [
        {"source": {"kind": "BOND", "atoms": [carbon, oxygen]},
         "sink": {"kind": "ATOM", "atoms": [oxygen]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [bromine]},
         "sink": {"kind": "BOND", "atoms": [carbon, bromine]}, "electrons": 2},
    ]


def test_r5_candidate_trace_finishes_with_real_compiled_proof() -> None:
    product = product_key("[Br-].CO")
    candidate = product_key("CBr.[OH-]")
    session = CandidateTraceSession(product, candidate)
    assert session.env.expected_precursor == ""
    prompt = session.prompt()
    assert product in prompt and candidate in prompt
    assert "ANNOTATED CURRENT STATE" in prompt
    arguments = render_event_arguments(session.env.current_state,
                                       _sn2_moves(session.env.current_state))
    changed = session.step("apply_electron_flow", arguments)
    assert changed["ok"]
    finished = session.step("finish_trace", {})
    assert finished["ok"] and session.termination_reason == "terminal_tool"
    record = session.attempt_record()
    final = record["final_result"]
    assert final["formal_execute"] and final["trace_bound"]
    assert final["endpoint_source"] == "environment_owned_trace"
    assert final["endpoint_exact"] is False  # no reference endpoint was given
    assert "compiled_proof" in final
    assert _trace_support([record], candidate) == (True, 1)


def test_r5_endpoint_only_import_is_not_falsely_formalized() -> None:
    product = product_key("CO")
    candidate = product_key("CBr.[OH-]")
    session = CandidateTraceSession(product, candidate)
    state = session.env.current_state
    result = session.step("import_fragments", {"fragments": [
        {"smiles": "[Br-]", "count": 1, "purpose": "endpoint_context"}]})
    assert not result["ok"]
    assert session.termination_reason == "tool_rejected"
    assert session.env.current_state == state
    assert session.attempt_record()["final_result"] is None
    with pytest.raises(RuntimeError, match="terminated"):
        session.prompt()


def test_r5_candidate_is_model_input_not_executor_expected_precursor() -> None:
    product = product_key("CO")
    candidate = product_key("CBr.[OH-]")
    session = CandidateTraceSession(product, candidate)
    assert session.model_input == {"product_smiles": product,
                                   "proposed_precursors": candidate}
    assert session.env.expected_precursor == ""
    result = session.step("finish_trace", {})
    assert not result["ok"]
    assert session.attempt_record()["termination_reason"] == "tool_rejected"
