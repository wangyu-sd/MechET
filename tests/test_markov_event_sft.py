import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_markov_event_sft import decision_rows
from scripts.train_tool_sft import validate_rows


def test_decision_rows_remove_gold_horizon_and_cover_finish() -> None:
    source = {
        "id": "trace-1",
        "source_id": "rxn-1",
        "target_smiles": "CC=O",
        "expected_precursor": "CCO",
        "messages": [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "TARGET: CC=O\nCURRENT STATE: CC=O"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "event_000",
                    "type": "function",
                    "function": {
                        "name": "apply_grounded_event",
                        "arguments": {"imports": ["[H][H]"], "marked_state": "<A>CC=<B>O", "flow": "A>AB"},
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "event_000",
                "name": "apply_grounded_event",
                "content": json.dumps({"ok": True, "code": "PASS", "current_state": "CCO", "remaining_events": 0}),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "finish_trace",
                    "type": "function",
                    "function": {"name": "finish_trace", "arguments": {}},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "finish_trace",
                "name": "finish_trace",
                "content": json.dumps({"ok": True, "endpoint_exact": True, "derived_precursor": "CCO"}),
            },
        ],
        "tools": [
            {"type": "function", "function": {"name": "apply_grounded_event", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "finish_trace", "parameters": {"type": "object"}}},
        ],
        "metadata": {"n_events": 1, "trace_digest": "abc"},
    }
    rows = decision_rows(source)
    assert len(rows) == 2
    assert [row["metadata"]["decision_type"] for row in rows] == ["event", "finish"]
    assert rows[1]["messages"][1]["content"].endswith("CURRENT STATE: CCO")
    assert all("remaining_events" not in json.dumps(row) for row in rows)
    audit = validate_rows(
        rows, require_trace_owned=False, require_tool_decision=True
    )
    assert audit["tool_calls"] == 2
    assert audit["finish_trace_rows"] == 1
