from __future__ import annotations

import copy

import pytest

from scripts.compare_principal_target_typed_successor import summarize_pairs


def _event(event_id: str, *, exact: bool) -> dict:
    outcome = {"execute_ok": True, "successor_exact": exact, "successor": "CO"}
    return {
        "id": event_id, "gold_successor": "CO", "pair_targets": [1],
        "gold_flow_count": 1,
        "policies": {"fixed1": outcome, "fixed2": outcome,
                     "validity_backoff_2_to_1": outcome},
    }


def test_paired_typed_summary_counts_local_gains_and_losses():
    old = {"a::0": _event("a::0", exact=True),
           "b::0": _event("b::0", exact=False)}
    new = {"a::0": _event("a::0", exact=False),
           "b::0": _event("b::0", exact=True)}
    result = summarize_pairs(old, new)
    assert result["counts"]["events"] == 2
    assert result["counts"]["old_exact"] == result["counts"]["new_exact"] == 1
    assert result["counts"]["old_only_exact"] == result["counts"]["new_only_exact"] == 1
    assert result["new_minus_old_successor_exact_rate"] == 0


def test_paired_typed_summary_rejects_changed_reference():
    old = {"a::0": _event("a::0", exact=False)}
    new = copy.deepcopy(old)
    new["a::0"]["gold_successor"] = "CN"
    with pytest.raises(ValueError, match="reference chemistry differs"):
        summarize_pairs(old, new)
