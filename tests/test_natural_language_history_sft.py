import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.trajectory_history import TrajectoryHistory
from scripts.build_natural_language_history_sft import transform_rows
from scripts.train_tool_sft import validate_rows


def _row(index: int, name: str, arguments: dict) -> dict:
    call_id = f"decision_{index:03d}"
    return {
        "id": f"rxn-1::nl_decision_{index:03d}",
        "source_id": "rxn-1",
        "task_type": "natural_language_electron_event_v1",
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "TARGET PRODUCT SMILES: CC=O\nCURRENT STATE SMILES: CC=O\n"
                    "\nChoose the single next retrosynthetic action."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps({"ok": True, "code": "PASS", "current_state": "CCO"}),
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for tool_name in (
                "import_fragments",
                "apply_electron_flow",
                "finish_trace",
            )
        ],
        "metadata": {"decision_index": index, "decision_type": name},
    }


def test_history_capsule_is_gold_only_and_runtime_reconstructible() -> None:
    history = TrajectoryHistory()
    assert "accepted_action_types: START" in history.render()
    history = history.accept(
        "import_fragments",
        {"fragments": [{"smiles": "[H][H]", "count": 2}]},
        {"ok": True, "code": "PASS"},
    )
    history = history.accept(
        "apply_electron_flow", {}, {"ok": True, "code": "PASS"}
    )
    text = history.render()
    assert "accepted_actions: 2" in text
    assert "imported_fragments_committed: 2" in text
    assert "electron_events_committed: 1" in text
    assert "last_action: apply_electron_flow" in text


def test_standard_decisions_gain_compressed_prior_history() -> None:
    rows = [
        _row(0, "import_fragments", {"fragments": [{"smiles": "[H][H]", "count": 1}]}),
        _row(1, "apply_electron_flow", {"electron_flow": []}),
        _row(2, "finish_trace", {}),
    ]
    output = list(transform_rows(rows))
    prompts = [row["messages"][1]["content"] for row in output]
    assert "accepted_action_types: START" in prompts[0]
    assert "accepted_action_types: import_fragments" in prompts[1]
    assert "accepted_action_types: import_fragments>apply_electron_flow" in prompts[2]
    assert all(row["metadata"]["history_contains_failed_actions"] is False for row in output)
    assert all("remaining_events" not in prompt for prompt in prompts)
    assert [
        row["messages"][2]["tool_calls"][0]["function"]["name"] for row in output
    ] == ["import_fragments", "apply_electron_flow", "finish_trace"]
    audit = validate_rows(
        output, require_trace_owned=False, require_tool_decision=True
    )
    assert audit["tool_calls"] == 3
    assert audit["finish_trace_rows"] == 1


def test_failed_gold_action_is_rejected() -> None:
    history = TrajectoryHistory()
    try:
        history.accept("apply_electron_flow", {}, {"ok": False, "code": "FAIL"})
    except ValueError as exc:
        assert "only passing gold actions" in str(exc)
    else:
        raise AssertionError("failed actions must not enter standard history SFT")
