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


def test_interrupted_validation_uses_new_attempt_without_overwriting_raw_io(tmp_path):
    from mechet.pr96_eval_runtime import reserve_validation_attempt
    target = tmp_path / "validation"
    first = reserve_validation_attempt(target)
    (first / "raw_generation.rank00.jsonl").write_text("original raw output\n")
    second = reserve_validation_attempt(target)
    assert first != second and second.is_dir()
    assert (first / "raw_generation.rank00.jsonl").read_text() == "original raw output\n"


def test_cached_validation_rejects_changed_decoder_or_checkpoint(tmp_path, monkeypatch):
    import json
    from mechet.pr96_eval_runtime import load_validation_report
    # Source verification is exercised with real files in artifact-gate tests.
    # Here isolate report reuse from its expensive source hashing boundary.
    binding = {"adapter_model_sha256": "original", "cohorts_sha256": "fixed"}
    monkeypatch.setattr("mechet.pr96_artifacts.verify_evaluation_bindings", lambda cfg, adapter: binding)
    cfg = {"pr96": {"track": "mech31k", "stage": "state", "max_new_tokens": 512}}
    report = {"track": "mech31k", "stage": "state", "cohort": "paired", "bindings": binding,
              "decoder": {"k": 1, "do_sample": False, "num_beams": 1, "max_decisions": 64, "max_new_tokens": 512}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert load_validation_report(cfg, "adapter", tmp_path, "paired") == report
    cfg["pr96"]["max_new_tokens"] = 1024
    with pytest.raises(ValueError, match="CACHED_VALIDATION"):
        load_validation_report(cfg, "adapter", tmp_path, "paired")
