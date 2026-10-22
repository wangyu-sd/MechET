import pytest


def test_aggregation_keeps_all_completed_ids_and_sums_raw_denominators():
    from mechet.pr96_eval_runtime import aggregate_parts
    part = {"metrics": {"closed_loop_completed": 1, "closed_loop_expected": 1,
             "structural_hits": 1, "full_hits": 0, "truncated_calls": 0,
             "expected_tf_flow_decisions": 2, "root_predicted_tools": {"import_fragments": 1},
             "rejected": {}, "gt_future_leak_detected": False,
             **{k: {"correct": 1, "total": 2} for k in
                 ("parse", "root_import_tool", "root_import_fragment", "tf_flow_successor", "tf_finish")}},
            "episodes": [{"reaction_id": "one"}]}
    result = aggregate_parts([part], ["one"])
    assert result["parse"] == {"correct": 1, "total": 2}
    assert result["closed_loop_completed"] == 1
    with pytest.raises(ValueError, match="IDS"):
        aggregate_parts([part], ["one", "missing"])
    with pytest.raises(ValueError, match="IDS"):
        aggregate_parts([part, part], ["one"])


def test_eight_rank_validation_directory_is_created_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from mechet.pr96_eval_runtime import prepare_output_directory
    barrier = Barrier(8)
    target = tmp_path / "validation"
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(prepare_output_directory, target, rank, barrier.wait) for rank in range(8)]
        for future in futures:
            future.result(timeout=5)
    assert target.is_dir()
    with pytest.raises(FileExistsError):
        prepare_output_directory(target, 0, lambda: None)
