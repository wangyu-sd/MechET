import json
from pathlib import Path
import sys

from mechet.endpoint_process_rl_env import EndpointProcessRLEnv
from mechet.forward_expert import verify_electron_step
from mechet.grounded_event_search import GroundedProposal


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_endpoint_rlvr_pilot import build_pilot_record, select_ids_streaming


TARGET = "[O-:1].[CH3:2][Br:3]"
MOVES = [
    {
        "source": {"kind": "LP", "atoms": [1]},
        "sink": {"kind": "BOND", "atoms": [1, 2]},
        "electrons": 2,
    },
    {
        "source": {"kind": "BOND", "atoms": [2, 3]},
        "sink": {"kind": "ATOM", "atoms": [3]},
        "electrons": 2,
    },
]
ENDPOINT = verify_electron_step(TARGET, MOVES)["state_smiles"]


def source_row():
    return {
        "id": "toy",
        "source_id": "toy",
        "target_smiles": TARGET,
        "expected_precursor": ENDPOINT,
        "full_precursor_state": ENDPOINT,
        "messages": [
            {"role": "system", "content": "private source prompt"},
            {"role": "user", "content": f"TARGET: {TARGET}"},
        ],
        "metadata": {
            "trace_digest": "trace",
            "move_sequence_digest": "moves",
            "trace_plan": {
                "steps": [
                    {
                        "step_index": 0,
                        "state_before": TARGET,
                        "state_after": ENDPOINT,
                        "moves": MOVES,
                        "imports": [],
                    }
                ]
            },
        },
    }


def test_build_pilot_record_replays_and_keeps_gold_private():
    row = build_pilot_record(source_row())
    assert row["public"]["product"] == "CBr.[O-]"
    assert row["metadata"]["n_events"] == 1
    assert len(row["private_reward"]["prefix_states"]) == 2
    env = EndpointProcessRLEnv(row)
    env.assert_no_reward_leakage()
    action = row["private_reward"]["gold_actions"][0]
    env.step(
        GroundedProposal(
            name=action["name"],
            arguments=action["arguments"],
            raw_response="gold action",
            logprob_sum=-1.0,
            token_count=2,
            seed=17,
        )
    )
    assert env.current_distance.total == 0


def test_streaming_selection_enforces_all_three_length_strata(tmp_path):
    path = tmp_path / "source.jsonl"
    rows = []
    for n_events in (1, 2, 3, 5, 6, 9):
        rows.append({"id": f"event_{n_events}", "metadata": {"n_trace_steps": n_events}})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = select_ids_streaming(
        path, count=6, seed=17, expected_source_rows=6
    )
    assert len(selected) == 6
    assert {n_events for _, n_events in selected} == {1, 2, 3, 5, 6, 9}
