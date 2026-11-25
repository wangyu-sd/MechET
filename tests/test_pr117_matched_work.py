import pytest

from scripts.pr117_matched_work import paired_prefix_counts, solution_cost


def test_first_solution_cost_charges_failed_dispatch_and_excludes_context_preflight():
    audit = dict(solved=True, status="stock_solved", gpu_requests=3, requests=4,
                 generated_tokens=9, first_solution_requests=3, first_solution_tokens=5)
    evidence = dict(seconds=17.2, result={"decision_receipts": [
        {"generated_token_ids": [10, 11]},
        {"generated_token_ids": [], "resource_error": "PHYSICAL_CONTEXT_LIMIT"},
        {"generated_token_ids": [12, 13, 14]},
        {"generated_token_ids": [15, 16, 17, 18]},
    ]})
    cost = solution_cost(evidence, audit)
    assert cost == {"native_calls": 2, "generated_tokens": 5, "terminal_seconds": 17.2}
    audit["gpu_requests"] = 4
    with pytest.raises(ValueError, match="dispatch counts"):
        solution_cost(evidence, audit)


def test_prefix_counts_keep_failed_and_censored_targets_in_each_comparison():
    first = "independent"
    second = "reaction-first"
    rows = {
        ("linear", "x"): {
            first: {"native_calls": 2, "generated_tokens": 600, "terminal_seconds": 160.0},
            second: {"native_calls": 4, "generated_tokens": 200, "terminal_seconds": 90.0},
        },
        ("long", "y"): {first: None, second: None},
    }
    effects = paired_prefix_counts(rows, [first, second], {"native_calls": [3],
                                    "generated_tokens": [300], "terminal_seconds": [120]})
    assert effects[f"{first}/{second}"]["native_calls<=3"] == {
        "n": 2, "neither": 1, "left_only": 1, "right_only": 0, "both": 0}
    for prefix in ("generated_tokens<=300", "terminal_seconds<=120"):
        assert effects[f"{first}/{second}"][prefix] == {
            "n": 2, "neither": 1, "left_only": 0, "right_only": 1, "both": 0}
