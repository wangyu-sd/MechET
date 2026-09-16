from mechet.natural_language_anchor_branch_rl import (
    assign_local_advantages,
    successor_fingerprint,
    task_from_episode,
)


def test_task_hides_reference_suffix_and_tracks_reset():
    task = task_from_episode(
        {
            "reaction_id": "train_1",
            "target": "C=O",
            "start_state": "[CH2:1]=[O:2]",
            "expected_precursor": "CO",
            "horizon": 2,
            "total_events": 4,
        }
    )
    assert task.prefix_events == 2
    assert not task.is_full_episode
    assert not hasattr(task, "gold_suffix")


def test_successor_pooling_is_chemical_state_based():
    left = successor_fingerprint(
        prompt_mode="event",
        action_name="apply_electron_flow",
        successor_state="[CH4:1].[OH2:2]",
        terminal=False,
    )
    right = successor_fingerprint(
        prompt_mode="event",
        action_name="apply_electron_flow",
        successor_state="[OH2:2].[CH4:1]",
        terminal=False,
    )
    assert left == right


def test_advantages_are_local_to_prompt_mode():
    records = []
    for mode, rewards in (("action", (1.0, 0.0)), ("event", (0.0, -0.1))):
        for index, reward in enumerate(rewards):
            records.append(
                {
                    "id": "train_1",
                    "anchor": {"state_hash": "same-state"},
                    "prompt_mode": mode,
                    "action_fingerprint": f"{mode}-{index}",
                    "reward": reward,
                    "score": {"correct": reward == 1.0},
                }
            )
    summary = assign_local_advantages(records)
    assert summary["effective"]
    assert records[0]["advantage"] > 0 > records[1]["advantage"]
    assert records[2]["advantage"] > 0 > records[3]["advantage"]
