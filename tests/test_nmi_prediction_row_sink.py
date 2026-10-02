from mechet.prediction_metrics import prediction_set_metrics


def test_row_sink_preserves_stable_source_id_and_candidate_outcomes():
    emitted = []
    row = {
        "id": "pred:x",
        "source_id": "flower_mech_proof_train_1",
        "artifact_type": "prediction",
        "prediction_mode": "direct",
        "prediction_status": "completed",
        "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
        "metadata": {"reference_source_id": "flower_mech_proof_train_1"},
        "candidates": [
            {"prediction": "<answer>[CH3:1][Br:3].[OH-:2]</answer>"},
            {"prediction": "<answer>[CH3:1][Cl:3].[OH-:2]</answer>"},
        ],
        "selected_candidate_index": 0,
        "prediction": "<answer>[CH3:1][Br:3].[OH-:2]</answer>",
    }
    report = prediction_set_metrics([row], ks=(1, 2), row_sink=emitted.append)
    assert report["structural_endpoint_pass_at_1"] == 1
    assert len(emitted) == 1
    assert emitted[0]["source_id"] == "flower_mech_proof_train_1"
    assert [item["structural_exact"] for item in emitted[0]["candidates"]] == [True, False]
