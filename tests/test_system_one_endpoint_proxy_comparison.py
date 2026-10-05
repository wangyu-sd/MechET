from __future__ import annotations

import pytest

from scripts.compare_system_one_endpoint_proxy_versions import summarize_paired


def _case(product: str, reference: str, *, hit: bool, context: list[str]) -> dict:
    return {
        "principal_product_input": product,
        "expected_structural_precursor": reference,
        "structural_exact": hit,
        "completed": True,
        "predicted_context_batch": context,
    }


def test_paired_proxy_groups_separate_changed_targets() -> None:
    old = {
        "1": _case("C", "CC", hit=False, context=["O"]),
        "2": _case("CO", "CO", hit=True, context=[]),
    }
    new = {
        "1": _case("O", "O", hit=True, context=["N"]),
        "2": _case("CO", "CO", hit=True, context=[]),
    }
    old_context = {"1": {"top1_exact": False}, "2": {"top1_exact": True}}
    new_context = {"1": {"top1_exact": True}, "2": {"top1_exact": True}}
    groups = summarize_paired(old, new, old_context, new_context, {"1"})
    assert groups["target_changed"]["new_only_exact"] == 1
    assert groups["target_changed"]["context_proposal_changed"] == 1
    assert groups["target_unchanged"]["both_exact"] == 1
    assert groups["all"]["reactions"] == 2


def test_paired_proxy_rejects_unaudited_target_change() -> None:
    old = {"1": _case("C", "CC", hit=False, context=[])}
    new = {"1": _case("O", "CC", hit=False, context=[])}
    contexts = {"1": {"top1_exact": False}}
    with pytest.raises(ValueError, match="disagrees with audit"):
        summarize_paired(old, new, contexts, contexts, set())


def test_paired_proxy_rejects_changed_reference_for_unchanged_target() -> None:
    old = {"1": _case("C", "CC", hit=False, context=[])}
    new = {"1": _case("C", "CO", hit=False, context=[])}
    contexts = {"1": {"top1_exact": False}}
    with pytest.raises(ValueError, match="changed reference"):
        summarize_paired(old, new, contexts, contexts, set())
