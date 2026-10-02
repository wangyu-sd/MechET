import json

import pytest

from mechet.open_flow_program import OpenFlowFormatError, execute_open_flow, parse_open_flow
from mechet.proof_program import sides_equal


MOVES = [
    {"source": {"kind": "BOND", "atoms": [1, 2]},
     "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
    {"source": {"kind": "LP", "atoms": [3]},
     "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
]


def _flow(imports=("[Br-:3]",), moves=MOVES):
    return "<flow>\nOPEN_FLOW v1\n" + "".join(f"IMPORT {value}\n" for value in imports) + \
        f"STEP 0 {json.dumps(moves)}\nEXECUTE\n</flow>"


def test_gold_style_open_flow_executes_without_gold_endpoint_input():
    result = execute_open_flow(_flow(), "[CH3:1][OH:2]")
    assert result["execute_ok"] is True
    assert sides_equal(result["derived_precursor"], "[CH3:1][Br:3].[OH-:2]", ignore_maps=False)
    assert result["n_imports"] == 1
    assert result["n_steps"] == 1


def test_open_flow_rejects_unordered_or_unfinished_program():
    with pytest.raises(OpenFlowFormatError, match="IMPORT_AFTER_STEP"):
        parse_open_flow(_flow(imports=()).replace("EXECUTE", "IMPORT [Br-:3]\nEXECUTE"))
    assert execute_open_flow(_flow().replace("EXECUTE", "STOP"), "[CH3:1][OH:2]")["failure_code"] == \
        "FLOW_HEADER_OR_TERMINAL_INVALID"
    assert execute_open_flow(_flow(), "[CH3:1][OH:2]", max_tool_calls=2)["failure_code"] == \
        "TOOL_BUDGET_EXCEEDED"


def test_invalid_fragment_does_not_trigger_gold_guided_repair():
    result = execute_open_flow(_flow(imports=("O",)), "[CH3:1][OH:2]")
    assert result["execute_ok"] is False
    assert result["failure_code"] == "IMPORT_FAILED"
