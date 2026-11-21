import pytest

from scripts.pr117_cbyd import policy_panel, split_policy_views
from scripts.pr117_cbyd_score import summarize_scores
from scripts import pr117_cbyd_score
from scripts.pr117_controllers import C0


def test_initial_ids_are_selected_by_identity_and_expansion_is_disjoint():
    rows = [dict(target_id=i, target="CC", seed=s) for s, i in enumerate(("gamma", "beta", "alpha"))]
    initial, remaining = split_policy_views(rows, {"alpha": "x", "beta": "y", "gamma": "z"}, ["x", "z"])
    assert initial == [rows[2], rows[0]]
    assert remaining == [rows[1]]
    with pytest.raises(ValueError, match="unique"):
        split_policy_views(rows, {"alpha": "x", "beta": "x", "gamma": "z"}, ["x", "z"])


def test_policy_panel_rejects_evaluator_labels_and_already_stock_targets():
    reference = dict(actor={"adapter_sha256": "a"}, stocks={"n1-n5-stock": {"size_after_executor_identity": 1}})
    row = dict(target_id="0123456789abcdef01234567", target="CC", seed=17)
    view = dict(seed=11, targets=[row], source_target_lock_sha256="0" * 64)
    panel = policy_panel(view, reference, {"C"}, C0, 64)
    assert panel["targets"] == [dict(row, cohort="cbyd", stock="n1-n5-stock")]
    assert panel["arms"] == [C0]
    with pytest.raises(ValueError, match="target-only"):
        policy_panel(dict(view, targets=[dict(row, precursor_signature="x")]), reference, {"C"}, C0, 64)
    with pytest.raises(ValueError, match="already in stock"):
        policy_panel(view, reference, {"CC"}, C0, 64)


def test_source_disconnection_and_stock_closure_are_different_full_denominator_metrics():
    rows = [
        dict(has_productive_root_commit=True, exact_first_commit=True, any_exact_match=True, stock_closed=False),
        dict(has_productive_root_commit=True, exact_first_commit=False, any_exact_match=True, stock_closed=True),
        dict(has_productive_root_commit=False, exact_first_commit=False, any_exact_match=False, stock_closed=False),
    ]
    assert summarize_scores(rows) == dict(targets=3, productive_first_reaction=2, exact_first_disconnection=1,
                                          exact_disconnection_found=2, full_inventory_closed=1)


def test_root_stereo_diagnostic_preserves_connectivity_and_counts_the_correct_call():
    reference = pr117_cbyd_score.stereo_free_precursors(["C[C@H](O)F", "CO"])
    assert reference == pr117_cbyd_score.stereo_free_precursors(["CO", "C[C@@H](O)F"])
    assert reference != pr117_cbyd_score.stereo_free_precursors(["[13CH3][C@H](O)F", "CO"])
    assert reference != pr117_cbyd_score.stereo_free_precursors(["C[C@H]([O-])F", "CO"])
    assert reference != pr117_cbyd_score.stereo_free_precursors(["C[C@H](O)F", "CO", "CO"])
    commits = [dict(occurrence="0", receipt="first", precursors=["C[C@@H](O)F", "CO"]),
               dict(occurrence="0", receipt="later", precursors=["C[C@H](O)F", "CO"])]
    exact = dict(first_exact_match_call=2, first_root_commit_call=1, exact_first_commit=False)
    outcome = pr117_cbyd_score.root_stereo_diagnostic(reference, commits, ["first", "later"], 2, exact)
    assert outcome == dict(first_root_structure="same_connectivity_different_stereochemistry",
                           first_connectivity_match_call=1)
