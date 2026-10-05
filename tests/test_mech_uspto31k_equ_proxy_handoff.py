from __future__ import annotations

import pytest

from scripts.verify_mech_uspto31k_equ_proxy_handoff import compare_rows


def _row(reaction_id: str, product: str, precursor: str) -> dict:
    return {
        "id": f"row-{reaction_id}",
        "reactants_unmapped": "CC.O",
        "product_unmapped": product,
        "precursor_unmapped": precursor,
    }


def test_target_change_can_change_structural_projection() -> None:
    old = {"1": _row("1", "C", "CC"), "2": _row("2", "CO", "CC")}
    new = {"1": _row("1", "O", "O"), "2": _row("2", "CO", "CC")}
    assert compare_rows(old, new, {"1"}) == {
        "reactions": 2,
        "product_changed": 1,
        "product_unchanged": 1,
        "structural_precursor_changed": 1,
    }


@pytest.mark.parametrize("mutation", ["id", "reactants_unmapped"])
def test_source_reaction_change_is_rejected(mutation: str) -> None:
    old = {"1": _row("1", "C", "CC")}
    new = {"1": dict(old["1"], **{mutation: "different"})}
    with pytest.raises(ValueError, match="reaction identity/full reactants changed"):
        compare_rows(old, new, set())


def test_structural_projection_cannot_change_for_same_product() -> None:
    old = {"1": _row("1", "C", "CC")}
    new = {"1": _row("1", "C", "O")}
    with pytest.raises(ValueError, match="structural precursor changed"):
        compare_rows(old, new, set())


def test_changed_target_ids_must_match_raw_audit() -> None:
    old = {"1": _row("1", "C", "CC")}
    new = {"1": _row("1", "O", "CC")}
    with pytest.raises(ValueError, match="changed product IDs differ"):
        compare_rows(old, new, set())
