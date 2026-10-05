from __future__ import annotations

import copy

import pytest

from scripts.compare_system_one_policy_versions import summarize_paired


def _case(reaction_id: str, *, exact: bool, state: str) -> dict:
    return {
        "id": reaction_id,
        "principal_product_input": "CO",
        "predicted_context_batch": ["O"],
        "inferred_final_mixture": "CO.O",
        "expected_structural_precursor": "CO",
        "structural_exact": exact,
        "completed": True,
        "actions": [{"action": "apply_electron_flow", "accepted": True,
                     "selected_pairs": [1], "state_after": state}],
    }


def test_paired_summary_counts_gains_losses_and_executed_changes():
    old = {"a": _case("a", exact=True, state="CO"),
           "b": _case("b", exact=False, state="CO")}
    new = {"a": _case("a", exact=False, state="C.O"),
           "b": _case("b", exact=True, state="C.O")}
    result = summarize_paired(old, new)
    assert result["counts"] == {
        "reactions": 2, "baseline_exact": 1, "candidate_exact": 1,
        "baseline_formal_finish": 2, "candidate_formal_finish": 2,
        "executed_path_changed": 2, "gained_exact": 1, "lost_exact": 1,
    }
    assert result["gained_ids"] == ["b"]
    assert result["lost_ids"] == ["a"]


def test_paired_summary_rejects_different_input():
    old = {"a": _case("a", exact=False, state="CO")}
    new = copy.deepcopy(old)
    new["a"]["predicted_context_batch"] = ["N"]
    with pytest.raises(ValueError, match="paired input/reference differs"):
        summarize_paired(old, new)
