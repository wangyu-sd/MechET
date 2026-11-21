import pytest

from scripts import run_pr117_reference as screen


def test_all_and_obligations_and_duplicate_stock_occurrences_are_required():
    edges = [
        dict(occurrence="0", product="A", precursors=["S", "B"]),
        dict(occurrence="0.1", product="B", precursors=["S", "T"]),
    ]
    assert screen.verify_closure("A", {"S", "T"}, dict(
        status="stock_solved", reactions=edges, stock_leaves=["S", "S", "T"]))
    with pytest.raises(ValueError, match="stock obligations"):
        screen.verify_closure("A", {"S", "T"}, dict(
            status="stock_solved", reactions=edges, stock_leaves=["S", "T"]))


def test_ancestor_and_zero_step_do_not_count_as_closed():
    assert not screen.verify_closure("A", {"A"}, dict(status="already_stock", reactions=[]))
    with pytest.raises(ValueError, match="ancestor"):
        screen.verify_closure("A", {"S"}, dict(status="stock_solved", stock_leaves=["S"], reactions=[
            dict(occurrence="0", product="A", precursors=["B", "S"]),
            dict(occurrence="0.0", product="B", precursors=["A"]),
        ]))
