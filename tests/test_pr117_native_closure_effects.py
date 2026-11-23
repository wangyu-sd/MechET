from scripts.pr117_native_closure_effects import measure


def test_native_prefix_effect_keeps_non_closures_and_counts_boundary_once():
    rows = [
        dict(cohort="linear", target_id="a", arm="early",
             first_verified_solution={"native_calls": 10}),
        dict(cohort="linear", target_id="a", arm="reaction",
             first_verified_solution={"native_calls": 12}),
        dict(cohort="long", target_id="b", arm="early", first_verified_solution=None),
        dict(cohort="long", target_id="b", arm="reaction", first_verified_solution=None),
    ]

    def estimator(targets, left, right, endpoint):
        return {
            "pairs": [(target[left]["cohort"], endpoint(target[left]), endpoint(target[right]))
                      for target in targets],
        }

    result = measure(rows, ["early", "reaction"], [("early", "reaction")], [10, 20], estimator)
    assert result["early/reaction"]["native_calls<=10"]["pairs"] == [
        ("linear", True, False), ("long", False, False)]
    assert result["early/reaction"]["native_calls<=20"]["pairs"] == [
        ("linear", True, True), ("long", False, False)]
