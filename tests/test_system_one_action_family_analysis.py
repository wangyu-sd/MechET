import json

from scripts.analyze_system_one_action_family import (
    calibration, history_signature, import_batch_strata,
)


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


def test_import_batch_strata_counts_wrong_finish(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [{"id": str(index), "metadata": {"decision_type": "import"},
             "messages": [{"role": "assistant", "tool_calls": [{"function": {
                 "name": "import_fragments", "arguments": {"fragments": [{
                     "smiles": "[Cl-]", "count": 1,
                     "purpose": "electron_participant"
                 }]}
             }}]}]} for index in range(2)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    strata = import_batch_strata(path, [1, 2])
    assert strata[0]["n"] == 2
    assert strata[0]["predictions"]["finish_trace"] == 1
    assert strata[0]["import_recall"] == 0.5
