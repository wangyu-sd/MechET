from scripts.smoke_earho_v2_balanced_policy import balance
from scripts.python_continual_stage import trainer_records


def _row(index, advantage, fingerprint, logp):
    return {
        "id": "reaction-1",
        "kind": "rl",
        "anchor": {"state_hash": "state-1"},
        "candidate_index": index,
        "advantage": advantage,
        "update_eligible": advantage != 0,
        "action_fingerprint": fingerprint,
        "old_logps": [0.0, logp],
        "loss_mask": [0, 1],
    }


def test_balancing_keeps_positives_and_most_likely_distinct_negative():
    rows = [
        _row(0, 1.0, "good", -0.3),
        _row(1, -1.0, "wrong-a", -0.1),
        _row(2, -1.0, "wrong-b", -0.4),
        _row(3, -1.0, "wrong-a", -0.2),
        {"id": "reaction-1", "kind": "verified_replay"},
    ]
    changed, report = balance(rows)
    assert report["positive_policy_rows"] == 1
    assert report["original_negative_policy_rows"] == 3
    assert report["selected_negative_policy_rows"] == 1
    assert report["verified_replay_rows"] == 1
    assert [row["advantage"] for row in changed[:4]] == [1.0, -1.0, 0.0, 0.0]
    assert [row["update_eligible"] for row in changed[:4]] == [True, True, False, False]
    assert changed[4] == rows[4]


def test_all_negative_anchor_stays_zero_update():
    rows = [_row(0, -1.0, "wrong-a", -0.1), _row(1, -1.0, "wrong-b", -0.2)]
    changed, report = balance(rows)
    assert report["selected_negative_policy_rows"] == 0
    assert all(row["advantage"] == 0.0 and not row["update_eligible"] for row in changed)


def test_learner_projection_excludes_heterogeneous_rollout_diagnostics():
    from datasets import Dataset

    rows = []
    for changes in ([], {"unexpected": "model-generated-shape"}):
        rows.append({
            "kind": "rl", "input_ids": [1, 2], "loss_mask": [0, 1],
            "old_logps": [0.0, -0.5], "advantage": 1.0,
            "score": {"trajectory": [{"arguments": {"bond_order_changes": changes}}]},
        })
    dataset = Dataset.from_list(trainer_records(rows))
    assert dataset.num_rows == 2
    assert set(dataset.column_names) == {
        "kind", "input_ids", "loss_mask", "old_logps", "advantage"
    }
