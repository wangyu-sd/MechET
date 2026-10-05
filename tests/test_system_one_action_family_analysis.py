from scripts.analyze_system_one_action_family import calibration, history_signature


def test_route_calibration_counts_bins_and_selective_accuracy():
    report = calibration([[0.8, 0.1, 0.1], [0.05, 0.9, 0.05], [0.1, 0.2, 0.7]],
                         [0, 0, 2])
    assert sum(group["n"] for group in report["bins"]) == 3
    assert report["selective_accuracy"][0]["n"] == 3
    assert report["selective_accuracy"][0]["accuracy"] == 2 / 3
    assert report["selective_accuracy"][2]["n"] == 1
    assert report["selective_accuracy"][2]["accuracy"] == 0.0


def test_history_signature_uses_past_fields_only():
    text = ("CURRENT STATE SMILES: CCO\n"
            "accepted_action_types: apply_electron_flow\n"
            "import_batches_committed: 0\n"
            "electron_events_committed: 1\n"
            "last_action: apply_electron_flow\n"
            "last_result: PASS\n")
    assert history_signature(text) == (
        "apply_electron_flow", "0", "1", "apply_electron_flow", "PASS"
    )
