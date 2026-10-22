"""Frozen validation must cover root FLOW and never admit a partial denominator."""
from collections import Counter

import pytest


def records():
    return [{"source_id": str(i), "first_tool": "apply_electron_flow" if i < 20 else "import_fragments",
             "flow_successors_match_source": (i % 6) + 1,
             "import_roles": {"electron_participant": 1, "endpoint_context": i % 2}}
            for i in range(240)]


def test_frozen_cohorts_are_disjoint_deterministic_and_include_rare_root_flow():
    from mechet.pr96_validation import frozen_cohorts
    first = frozen_cohorts(records(), seed=17)
    shuffled = frozen_cohorts(list(reversed(records())), seed=17)
    assert first == shuffled
    assert len(first["paired_ids"]) == 64
    assert len(first["independent_ids"]) == 64
    assert not set(first["paired_ids"]) & set(first["independent_ids"])
    rows = {r["source_id"]: r for r in records()}
    for key in ("paired_ids", "independent_ids"):
        chosen = [rows[i] for i in first[key]]
        assert Counter(r["first_tool"] for r in chosen) == {"apply_electron_flow": 4, "import_fragments": 60}
        assert {r["flow_successors_match_source"] for r in chosen} == {1, 2, 3, 4, 5, 6}


def passing_metrics():
    return {
        "closed_loop_completed": 64, "closed_loop_expected": 64, "structural_hits": 4,
        "parse": {"correct": 990, "total": 1000},
        "root_import_tool": {"correct": 48, "total": 60},
        "root_import_fragment": {"correct": 15, "total": 60},
        "tf_flow_successor": {"correct": 60, "total": 100},
        "expected_tf_flow_decisions": 100,
        "tf_finish": {"correct": 52, "total": 64},
        "root_predicted_tools": {"import_fragments": 60, "apply_electron_flow": 4},
        "truncated_calls": 0, "gt_future_leak_detected": False,
    }


def test_gate_is_go_at_fixed_thresholds_and_no_go_for_partial_evaluation():
    from mechet.pr96_validation import stage_one_gate
    assert stage_one_gate(passing_metrics())["go"]
    m = passing_metrics()
    m.update(closed_loop_completed=32, closed_loop_expected=32)
    result = stage_one_gate(m)
    assert not result["go"]
    assert "full_frozen_64" in result["failed"]


@pytest.mark.parametrize("field", ["parse", "root_import_tool", "root_import_fragment", "tf_flow_successor", "tf_finish"])
def test_gate_does_not_treat_zero_or_failed_denominators_as_success(field):
    from mechet.pr96_validation import stage_one_gate
    m = passing_metrics()
    m[field] = {"correct": 0, "total": 0}
    assert not stage_one_gate(m)["go"]
    m[field] = {"correct": 0, "total": 64}
    assert not stage_one_gate(m)["go"]


def test_gate_rejects_constant_tool_collapse_and_truncation():
    from mechet.pr96_validation import stage_one_gate
    m = passing_metrics()
    m["root_predicted_tools"] = {"import_fragments": 64}
    assert "no_root_collapse" in stage_one_gate(m)["failed"]
    m = passing_metrics()
    m["truncated_calls"] = 1
    assert not stage_one_gate(m)["go"]


@pytest.mark.parametrize("field,total", [("root_import_tool", 1), ("root_import_fragment", 1),
                                         ("tf_finish", 1), ("tf_flow_successor", 1)])
def test_gate_requires_all_reference_decisions_not_only_successful_calls(field, total):
    from mechet.pr96_validation import stage_one_gate
    m = passing_metrics()
    m[field] = {"correct": total, "total": total}
    assert not stage_one_gate(m)["go"]
