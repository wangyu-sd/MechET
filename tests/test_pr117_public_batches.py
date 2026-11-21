import pytest

from scripts.pr117_public_batches import fully_published, merge_verified, partition_indexes


def test_partitions_preserve_each_target_seed_and_balance_the_registered_cohorts():
    rows = [dict(cohort=c, target_id=f"{c}-{i}", seed=i) for c in ("linear", "long") for i in range(5)]
    partitions = partition_indexes(rows, 3)
    assert partitions == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(index for batch in partitions for index in batch) == list(range(10))
    with pytest.raises(ValueError, match="identities"):
        partition_indexes([*rows, rows[0]], 3)


def test_partial_archive_retains_incomplete_denominator_and_rejects_duplicate_cases():
    expected = {("linear", "a"), ("long", "b")}
    first = {("linear", "a"): dict(native_calls=3, stock_closed=True,
                                   first_solution={"native_calls":2})}
    second = {("long", "b"): dict(native_calls=5,stock_closed=False,first_solution=None)}
    partial = merge_verified(expected, [first], cap=8)
    assert partial["status"] == "PARTIAL_REFERENCE_EVIDENCE"
    assert partial["evaluated_targets"] == 1 and partial["registered_targets"] == 2
    assert partial["actual_calls_in_replayed_targets"] == 3 and "solved_by_budget" not in partial
    full = merge_verified(expected, [first, second], cap=8)
    assert full["status"] == "COMPLETE_REFERENCE_EVIDENCE"
    assert full["solved_by_budget"]["8"] == {"closed":1,"targets":2}
    assert full["actual_calls_in_replayed_targets"] == 8
    awaiting_final_summary = merge_verified(expected, [first, second], cap=8, all_workers_verified=False)
    assert awaiting_final_summary["evaluated_targets"] == 2
    assert awaiting_final_summary["status"] == "PARTIAL_REFERENCE_EVIDENCE"
    assert "solved_by_budget" not in awaiting_final_summary
    with pytest.raises(ValueError, match="duplicate"):
        merge_verified(expected, [first, first, second], cap=8)


def test_live_partial_worker_summary_does_not_count_as_a_finished_batch(tmp_path):
    path = tmp_path / "screen_result.json"
    assert not fully_published(path)
    path.write_text('{"targets": ')
    assert not fully_published(path)
    path.write_text('{"targets": 1}\n')
    assert fully_published(path)


def test_training_exposure_reports_fixed_disjoint_groups_and_development_overlap():
    exact, stereo_only, reused, independent = [("linear", target) for target in ("a", "b", "c", "d")]
    expected = {exact, stereo_only, reused, independent}
    exposure = {
        exact: dict(exact_train_sources=["stage2"], connectivity_train_sources=["stage2"], reused_development=False),
        stereo_only: dict(exact_train_sources=[], connectivity_train_sources=["d3"], reused_development=False),
        reused: dict(exact_train_sources=[], connectivity_train_sources=[], reused_development=True),
        independent: dict(exact_train_sources=[], connectivity_train_sources=[], reused_development=False),
    }
    solved = dict(native_calls=3, stock_closed=True, first_solution={"native_calls": 2})
    unsolved = dict(native_calls=8, stock_closed=False, first_solution=None)
    all_cases = {exact: solved, stereo_only: unsolved, reused: solved, independent: unsolved}

    full = merge_verified(expected, [all_cases], cap=8, exposure=exposure)
    assert full["solved_by_budget"]["8"] == {"closed": 2, "targets": 4}
    groups = full["solved_by_training_exposure_and_budget"]
    assert groups["exact_train"]["8"] == {"closed": 1, "targets": 1}
    assert groups["train_connectivity_only"]["8"] == {"closed": 0, "targets": 1}
    assert groups["no_train_connectivity"]["8"] == {"closed": 1, "targets": 2}
    assert full["solved_on_independent_targets_by_budget"]["8"] == {"closed": 0, "targets": 1}

    partial = merge_verified(expected, [{exact: solved}], cap=8, exposure=exposure)
    assert "solved_by_training_exposure_and_budget" not in partial
    assert "solved_on_independent_targets_by_budget" not in partial
    del exposure[independent]
    with pytest.raises(ValueError, match="exposure annotations"):
        merge_verified(expected, [all_cases], cap=8, exposure=exposure)
