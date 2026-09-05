import json

from scripts.build_a8_lite_state_reset_sft import (
    _offer_smallest,
    Candidate,
    make_anchor_row,
    make_state_reset_row,
)
from scripts.train_tool_sft import validate_conversation


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tool_result(call_id: str, name: str, state: str, *, terminal: bool = False) -> dict:
    content = {
        "ok": True,
        "code": "PASS",
        "observation_mode": "compact_full_state_v1",
        "remaining_tool_calls": 4,
    }
    if terminal:
        content["derived_precursor"] = state
    else:
        content["current_state_smiles"] = state
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(content),
    }


def _row() -> dict:
    return {
        "id": "source-1",
        "source_id": "source-1",
        "artifact_type": "source",
        "target_smiles": "[CH3:1][Br:2]",
        "expected_precursor": "[Br-:2].[CH3:1][OH:3]",
        "tools": [
            {"type": "function", "function": {"name": "import_fragment"}},
            {"type": "function", "function": {"name": "apply_electron_moves"}},
            {"type": "function", "function": {"name": "finish_trace"}},
        ],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "TARGET: [CH3:1][Br:2]"},
            _tool_call("call_000", "import_fragment", {"fragment_smiles": "[OH-:3]"}),
            _tool_result("call_000", "import_fragment", "[CH3:1][Br:2].[OH-:3]"),
            _tool_call("call_001", "apply_electron_moves", {"moves": []}),
            _tool_result("call_001", "apply_electron_moves", "[Br-:2].[CH3:1][OH:3]"),
            _tool_call("call_002", "finish_trace", {}),
            _tool_result(
                "call_002",
                "finish_trace",
                "[Br-:2].[CH3:1][OH:3]",
                terminal=True,
            ),
        ],
        "metadata": {
            "endpoint_source": "environment_owned_trace",
            "executor_replayed": True,
            "trace_digest": "abc",
            "n_trace_steps": 1,
            "n_trace_imports": 1,
        },
    }


def test_state_reset_keeps_target_state_and_executable_suffix():
    source = _row()
    converted = make_state_reset_row(source)

    assert converted["id"].startswith("a8-lite-reset:source-1:after-")
    assert converted["messages"][0] == source["messages"][0]
    assert converted["messages"][1]["role"] == "user"
    assert "TARGET: [CH3:1][Br:2]" in converted["messages"][1]["content"]
    assert "current_state_smiles" in converted["messages"][1]["content"]
    assert converted["messages"][2]["role"] == "assistant"
    assert converted["metadata"]["a8_lite_off_policy_recovery"] is False

    counts = validate_conversation(converted, require_trace_owned=True)
    assert counts["finish_trace"] == 1
    assert counts["tool_calls"] >= 1


def test_anchor_preserves_conversation_without_mutating_source():
    source = _row()
    converted = make_anchor_row(source)

    assert converted["messages"] == source["messages"]
    assert converted["id"] == "a8-lite-anchor:source-1"
    assert source["id"] == "source-1"
    assert converted["metadata"]["a8_lite_role"] == "expert_anchor"


def test_bounded_heap_keeps_smallest_scores():
    heap = []
    for index, score in enumerate((8, 1, 5, 2, 9)):
        item = Candidate(index, f"row-{index}", score, 12)
        _offer_smallest(heap, item, 3)

    assert sorted(entry[2].score for entry in heap) == [1, 2, 5]
