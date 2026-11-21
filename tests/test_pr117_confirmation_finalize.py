from scripts.pr117_confirmation_finalize import replay_status


def test_partial_replay_preserves_issued_and_audited_calls_separately():
    result = replay_status({
        "expected": 288,
        "fully_audited": False,
        "final_summary_verified": False,
        "fully_paired_targets": 0,
        "journal": {"gpu_requests": 31, "gpu_returns": 30},
        "rows": [{"arm": "independent", "gpu_requests": 20, "solved": True}],
    })
    assert result["status"] == "PARTIAL_RECEIPTS_VERIFIED"
    assert result["expected_arms"] == 288
    assert result["fully_paired_targets"] == 0
    assert result["audited_native_calls"] == 20
    assert result["issued_native_calls"] == 31
    assert result["returned_native_calls"] == 30
    assert result["audited_closures_by_arm"] == {"independent": 1}


def test_all_arms_with_a_missing_terminal_summary_remain_partial():
    result = replay_status({
        "expected": 1,
        "fully_audited": True,
        "final_summary_verified": False,
        "fully_paired_targets": 1,
        "journal": {"gpu_requests": 20, "gpu_returns": 20},
        "rows": [{"arm": "independent", "gpu_requests": 20, "solved": False}],
    })
    assert result["status"] == "PARTIAL_RECEIPTS_VERIFIED"
