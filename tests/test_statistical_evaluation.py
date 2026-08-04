from mechet.statistical_evaluation import (
    aggregate_seed_effects,
    hierarchical_paired_binary_contrast,
    holm_adjust,
    paired_binary_contrast,
)


def test_paired_binary_contrast_reports_direction_ci_and_mcnemar():
    left = [True] * 8 + [False] * 2
    right = [True] * 3 + [False] * 7
    result = paired_binary_contrast(
        left,
        right,
        bootstrap_samples=1000,
        bootstrap_seed=3,
    )
    assert result["delta_left_minus_right"] == 0.5
    assert result["left_correct_right_incorrect"] == 5
    assert result["left_incorrect_right_correct"] == 0
    assert result["paired_bootstrap_ci"][0] > 0
    assert 0 <= result["mcnemar_exact_p_value"] <= 1


def test_holm_adjust_is_monotone_and_preserves_missing_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20, "missing": None})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert adjusted["missing"] is None
    assert adjusted["a"] == 0.03


def test_seed_aggregation_is_deterministic():
    values = {"1": 0.1, "2": 0.2, "3": 0.3}
    first = aggregate_seed_effects(values, bootstrap_samples=500, bootstrap_seed=9)
    second = aggregate_seed_effects(values, bootstrap_samples=500, bootstrap_seed=9)
    assert first == second
    assert first["n_seeds"] == 3
    assert first["mean_effect"] == 0.2
    assert first["positive_seed_fraction"] == 1.0


def test_hierarchical_bootstrap_resamples_seeds_and_paired_rows():
    result = hierarchical_paired_binary_contrast(
        {
            "1": ([True, True, False], [False, True, False]),
            "2": ([True, True, True], [True, False, False]),
        },
        bootstrap_samples=500,
        bootstrap_seed=5,
    )
    assert result["n_seeds"] == 2
    assert result["mean_effect"] > 0
    assert len(result["hierarchical_bootstrap_ci"]) == 2
