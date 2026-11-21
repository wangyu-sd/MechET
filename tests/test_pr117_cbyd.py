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


def test_cbyd_merge_keeps_original_counts_and_rejects_mixed_actors_or_duplicate_roots():
    binding = dict(actor_sha256="actor", stock_source_sha256="stock", per_target_arm_request_cap=64,
                   panel_sha256="full", cbyd_policy_sha256="full-policy", targets=2, total_request_ceiling=128)

    def part(target, calls, solved):
        row = dict(target_id=target, source_target_id="source-" + target, actual_native_calls=calls,
                   has_productive_root_commit=solved, exact_first_commit=solved,
                   any_exact_match=solved, stock_closed=solved)
        return dict(binding=dict(binding, panel_sha256=target, cbyd_policy_sha256="policy-" + target,
                                 targets=1, total_request_ceiling=64),
                    evaluator_sha256="evaluator", mapping_sha256="mapping", native_summary_sha256=target,
                    actual_native_calls=calls, totals=summarize_scores([row]), rows=[row])

    stages = {"initial": part("a", 5, False), "additional": part("b", 9, True)}
    sources = {"a": "source-a", "b": "source-b"}
    result = pr117_cbyd_score.merge_cbyd_scores(stages, binding, sources)
    assert result["totals"]["targets"] == 2 and result["totals"]["full_inventory_closed"] == 1
    assert result["actual_native_calls"] == 14
    assert result["stages"]["initial"]["actual_native_calls"] == 5
    assert [r["target_id"] for r in result["rows"]] == ["a", "b"]
    with pytest.raises(ValueError, match="Duplicate"):
        pr117_cbyd_score.merge_cbyd_scores(dict(stages, additional=stages["initial"]), binding, sources)
    conflicting = dict(stages["additional"], binding=dict(stages["additional"]["binding"], actor_sha256="other"))
    with pytest.raises(ValueError, match="frozen evaluation"):
        pr117_cbyd_score.merge_cbyd_scores(dict(stages, additional=conflicting), binding, sources)
