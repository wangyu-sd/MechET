"""V3 SFT rows preserve the v2 chemistry, observations and training contract."""
from copy import deepcopy

from mechet.a7_rescue import canonical_event
from mechet.natural_language_electron_flow import (
    compile_event_arguments, render_event_arguments,
)
from scripts.build_compact_electron_flow_sft import (
    convert_decision, SYSTEM, TOOLS, HISTORY, STATE,
)
from scripts.build_natural_language_event_sft import _decision_row
from scripts.train_tool_sft import validate_rows

STATE_SMILES = "[O:1]=[C:2]([OH:3])[CH3:4].[O-:5][CH2:6][CH3:7]"
MOVES = [
    {"source": {"kind": "BOND", "atoms": [1, 2]},
     "sink": {"kind": "ATOM", "atoms": [1]}, "electrons": 2},
    {"source": {"kind": "LP", "atoms": [5]},
     "sink": {"kind": "BOND", "atoms": [2, 5]}, "electrons": 2},
]


def _row(name, args, decision_type):
    return _decision_row(
        row={"source_id": "r1", "target_smiles": "CO",
             "expected_precursor": "CBr"},
        sequence_index=0, decision_type=decision_type,
        mapped_state=STATE_SMILES, name=name, arguments=args,
        result={"ok": True, "code": "PASS", "current_state": "CO"},
    )


def test_converts_event_without_changing_input_or_executor_result():
    original = _row("apply_electron_flow", render_event_arguments(STATE_SMILES, MOVES), "event")
    saved = deepcopy(original)
    converted = convert_decision(original)
    assert original == saved
    assert converted["messages"][0]["content"] == SYSTEM
    assert converted["messages"][1] == saved["messages"][1]
    assert converted["messages"][3] == saved["messages"][3]
    assert converted["tools"] == TOOLS
    assert converted["metadata"]["decision_contract"] == STATE
    arguments = converted["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert set(arguments) == {"flow"}
    assert canonical_event(compile_event_arguments(STATE_SMILES, arguments)) == canonical_event(MOVES)
    assert validate_rows([converted], require_trace_owned=False, require_tool_decision=True)["tool_calls"] == 1


def test_preserves_fragment_and_finish_calls():
    imports = {"fragments": [{"smiles": "[Br-]", "count": 1, "purpose": "electron_participant"}]}
    for name, arguments, kind in (
        ("import_fragments", imports, "import"), ("finish_trace", {}, "finish")
    ):
        source = _row(name, arguments, kind)
        output = convert_decision(source)
        assert output["messages"][2]["tool_calls"][0]["function"]["arguments"] == arguments


def test_history_contract_does_not_leak_reference():
    source = _row("apply_electron_flow", render_event_arguments(STATE_SMILES, MOVES), "event")
    source["metadata"].update({
        "decision_contract": "unified_inventory_compressed_history_tool_decision_v2",
        "history_contract": "executor_compact_accepted_actions_v1",
        "history_contains_failed_actions": False,
        "history_model_visible_gold_horizon": False,
    })
    source["messages"][1]["content"] += "\nTRAJECTORY HISTORY (executor-owned; past actions only)"
    converted = convert_decision(source)
    assert converted["metadata"]["decision_contract"] == HISTORY
    assert converted["messages"][1]["content"] == source["messages"][1]["content"]
    assert validate_rows([converted], require_trace_owned=False, require_tool_decision=True)["tool_calls"] == 1
