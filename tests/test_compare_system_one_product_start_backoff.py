import pytest

from scripts.compare_system_one_product_start_backoff import compare


def _report(*, exact, backoff):
    return {
        "scope": "product_only_autonomous_decisions_strict_step_executor_no_formal_proof_compilation",
        "split": "valid", "source": {"sha256": "valid-hash"},
        "train_import_source": {"sha256": "train-hash"},
        "reaction_denominator": 2, "evaluated_reactions": 2,
        "selection": {"limit": 0, "seed": 17}, "max_actions": 12,
        "weights": {"typed_head_sha256": "head-hash"},
        "endpoint_exact": exact,
        "terminal_counts": {"FINISHED": exact, "ELECTRON_EXECUTION_FAILED": 2 - exact},
        "legality_backoff": backoff,
    }


def _case(case_id, *, exact, terminal, actions):
    return {
        "id": case_id, "target": "CO", "expected_precursor": "C.O",
        "reference_decisions": 2, "terminal": terminal,
        "completed": terminal == "FINISHED", "endpoint_exact": exact,
        "predicted_precursor": "C.O" if exact else None,
        "electron_events": int(exact), "import_batches": 0,
        "actions": actions,
    }


def test_paired_backoff_credits_only_changed_failed_episode():
    failed = {"step": 0, "action": "apply_electron_flow", "state_before": "CO",
              "execute_ok": False, "code": "CHEMICAL_STATE_INVALID",
              "ranked_top8": [1, 2], "selected_pairs": [1]}
    rescued = {**failed, "execute_ok": True, "code": "PASS",
               "selected_pairs": [2], "state_after": "C.O"}
    unchanged = _case("a", exact=True, terminal="FINISHED", actions=[])
    base = {"a": unchanged,
            "b": _case("b", exact=False, terminal="ELECTRON_EXECUTION_FAILED",
                       actions=[failed])}
    candidate = {"a": unchanged,
                 "b": _case("b", exact=True, terminal="FINISHED", actions=[rescued])}
    result = compare(_report(exact=1, backoff=False), base,
                     _report(exact=2, backoff=True), candidate)
    assert result["improved_reactions"] == 1
    assert result["worsened_reactions"] == 0
    assert result["failed_events_locally_rescued"] == 1
    assert result["endpoint_delta_percentage_points"] == 50.0


def test_paired_backoff_rejects_drift_on_unchanged_episode():
    base = {"a": _case("a", exact=True, terminal="FINISHED", actions=[])}
    candidate = {"a": {**base["a"], "predicted_precursor": "CO"}}
    baseline_report = _report(exact=1, backoff=False)
    candidate_report = _report(exact=1, backoff=True)
    baseline_report["evaluated_reactions"] = candidate_report["evaluated_reactions"] = 1
    with pytest.raises(ValueError, match="non-failure episode changed"):
        compare(baseline_report, base, candidate_report, candidate)
