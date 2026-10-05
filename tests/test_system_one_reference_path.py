import pytest

from scripts.analyze_system_one_reference_path import summarize_reference_paths


def _row(reaction, index, action, route=True, event=None):
    return {
        "reaction_id": reaction,
        "decision_index": index,
        "gold_action": action,
        "route_correct": route,
        "event_successor_exact": event,
    }


def test_reference_path_counts_first_failure_without_import_argument_credit():
    rows = [
        _row("r1", 0, "apply_electron_flow", event=True),
        _row("r1", 1, "finish_trace"),
        _row("r2", 0, "apply_electron_flow", event=False),
        _row("r2", 1, "finish_trace"),
        _row("r3", 0, "apply_electron_flow", event=True),
        _row("r3", 1, "import_fragments", route=False),
        _row("r3", 2, "apply_electron_flow", event=True),
        _row("r3", 3, "finish_trace"),
    ]
    result = summarize_reference_paths(rows)
    assert result["reactions"] == 3
    assert result["all_routes_correct"] == 2
    assert result["all_event_successors_exact"] == 2
    assert result["all_reference_path_local_agreement"] == 1
    assert result["first_failure"] == {
        "event_successor": 1,
        "none": 1,
        "route_import_fragments": 1,
    }
    assert result["by_gold_event_count"]["2"]["reactions"] == 1


def test_reference_path_rejects_missing_or_nonconsecutive_finish():
    with pytest.raises(ValueError, match="nonconsecutive"):
        summarize_reference_paths([
            _row("r", 0, "apply_electron_flow", event=True),
            _row("r", 2, "finish_trace"),
        ])
    with pytest.raises(ValueError, match="final FINISH"):
        summarize_reference_paths([_row("r", 0, "apply_electron_flow", event=True)])
