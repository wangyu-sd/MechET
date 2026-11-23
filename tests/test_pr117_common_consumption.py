from scripts.pr117_common_consumption import compare_prefixes


def test_common_prefix_charges_same_work_and_keeps_early_stops_in_denominator():
    independent, electronic, reaction = "independent", "electronic", "reaction"
    arms = (independent, electronic, reaction)
    x = ("linear", "target-x")
    y = ("long", "target-y")

    def chemistry(calls, flow, commits=0, roots=0):
        return dict(native_calls=calls, distinct_flow_transitions=flow,
                    distinct_committed_edges=commits, distinct_root_precursor_sets=roots)

    prefixes = {
        (*x, independent): {"2": chemistry(2, 0)},
        (*x, electronic): {"2": chemistry(2, 1, 1, 1)},
        (*x, reaction): {"2": chemistry(2, 1)},
    }
    work = {
        (*x, independent): dict(native_calls=2, first_verified_solution=None),
        (*x, electronic): dict(native_calls=3, first_verified_solution={"native_calls": 2}),
        (*x, reaction): dict(native_calls=4, first_verified_solution={"native_calls": 3}),
        (*y, independent): dict(native_calls=1, first_verified_solution=None),
        (*y, electronic): dict(native_calls=0, first_verified_solution=None),
        (*y, reaction): dict(native_calls=2, first_verified_solution=None),
    }
    report = compare_prefixes((x, y), prefixes, work, arms)
    summary = report["by_stratum"]["all"]
    assert summary["targets"] == 2 and summary["shared_physical_calls_per_arm"] == 2
    assert summary["arms"][electronic]["verified_stock_closures"] == 1
    assert summary["arms"][reaction]["verified_stock_closures"] == 0
    assert summary["paired"][f"{electronic}/{reaction}"]["distinct_flow_transitions"] == {
        "left_more": 0, "right_more": 0, "tie": 2,
    }
    assert summary["paired"][f"{electronic}/{reaction}"]["verified_stock_closures"] == {
        "left_more": 1, "right_more": 0, "tie": 1,
    }
    assert report["by_stratum"]["long"]["shared_physical_calls_per_arm"] == 0
    assert [row["shared_physical_calls"] for row in report["targets_detail"]] == [2, 0]
