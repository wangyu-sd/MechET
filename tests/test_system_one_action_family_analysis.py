from scripts.analyze_system_one_action_family import calibration


def test_route_calibration_counts_bins_and_selective_accuracy():
    report = calibration([[0.8, 0.1, 0.1], [0.05, 0.9, 0.05], [0.1, 0.2, 0.7]],
                         [0, 0, 2])
    assert sum(group["n"] for group in report["bins"]) == 3
    assert report["selective_accuracy"][0]["n"] == 3
    assert report["selective_accuracy"][0]["accuracy"] == 2 / 3
    assert report["selective_accuracy"][2]["n"] == 1
    assert report["selective_accuracy"][2]["accuracy"] == 0.0
