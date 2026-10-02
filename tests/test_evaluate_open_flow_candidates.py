import json

from scripts.evaluate_open_flow_candidates import score_candidate


def test_open_flow_candidate_scores_endpoint_only_after_execution():
    moves = [
        {"source": {"kind": "BOND", "atoms": [1, 2]},
         "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [3]},
         "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
    ]
    program = f"<flow>\nOPEN_FLOW v1\nIMPORT [Br-:3]\nSTEP 0 {json.dumps(moves)}\nEXECUTE\n</flow>"
    reference = {
        "id": "one",
        "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
        "full_precursor_state": "[CH3:1][Br:3].[OH-:2]",
    }
    result = score_candidate({"prediction": program, "sample_index": 0}, reference, 0)
    assert result["execute_ok"] is True
    assert result["structural_exact"] is True
    assert result["full_precursor_exact"] is True


def test_open_flow_invalid_candidate_is_not_repaired():
    reference = {"id": "one", "target_smiles": "[CH3:1][OH:2]",
                 "structural_precursor": "[CH3:1][Br:3]"}
    result = score_candidate({"prediction": "<flow>bad</flow>"}, reference, 0)
    assert result["execute_ok"] is False
    assert result["structural_exact"] is False
    assert result["failure_code"] == "FLOW_HEADER_OR_TERMINAL_INVALID"
