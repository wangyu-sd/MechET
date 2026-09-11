from mechet.endpoint_process_rl_env import EndpointProcessRLEnv
from mechet.forward_expert import verify_electron_step
from mechet.grounded_event_search import GroundedProposal


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


def pilot_row():
    return {
        "id": "sn2",
        "public": {
            "system_prompt": "Execute inverse electron-flow events.",
            "product": "CBr.[O-]",
            "tools": [],
        },
        "private_reward": {
            "target_mapped_state": TARGET,
            "expected_precursor_mapped": ENDPOINT,
            "target_atom_maps": [1, 2, 3],
            "contributing_atom_maps": [],
            "gold_import_fragments": [],
            "prefix_states": [TARGET, ENDPOINT],
            "transitions": [
                {
                    "step_index": 0,
                    "state_before": TARGET,
                    "state_after": ENDPOINT,
                    "moves": MOVES,
                    "imports": [],
                }
            ],
        },
        "metadata": {"n_events": 1},
    }


def event():
    return GroundedProposal(
        name="apply_grounded_event",
        arguments={
            "imports": [],
            "marked_state": "<A>[O-].<B>C<C>Br",
            "flow": "A>AB ; BC>C",
        },
        raw_response="event",
        logprob_sum=-2.0,
        token_count=4,
        seed=17,
    )


def invalid():
    return GroundedProposal(
        name="apply_grounded_event",
        arguments={
            "imports": [],
            "marked_state": "not the current state",
            "flow": "A>AB",
        },
        raw_response="invalid",
        logprob_sum=-1.0,
        token_count=2,
        seed=18,
    )


def finish():
    return GroundedProposal(
        name="finish_trace",
        arguments={},
        raw_response="finish",
        logprob_sum=-1.0,
        token_count=1,
        seed=19,
    )


def test_rejection_keeps_state_and_second_rejection_exhausts_retry():
    env = EndpointProcessRLEnv(pilot_row())
    original = env.node.current_mapped_state
    first = env.step(invalid())
    assert not first.accepted
    assert first.state_unchanged
    assert env.node.current_mapped_state == original
    assert not env.done
    assert "GROUNDING_GRAPH_MISMATCH" in str(env.public_messages()[-1])
    second = env.step(invalid())
    assert env.done
    assert second.code == "RETRY_EXHAUSTED"
    assert second.total_reward == -1.0
    assert env.summary().retry_exhausted


def test_exact_finish_dominates_event_progress_and_compiles_prefix():
    env = EndpointProcessRLEnv(pilot_row())
    progress = env.step(event())
    assert progress.accepted
    assert 0 < progress.progress_reward <= 0.25
    terminal = env.step(finish())
    assert env.done
    assert terminal.code == "EXACT_ENDPOINT"
    assert terminal.terminal_reward == 4.0
    assert terminal.terminal_reward > abs(progress.progress_reward)
    summary = env.summary()
    assert summary.endpoint_exact
    assert summary.explicit_finish
    assert summary.formal_terminal


def test_private_endpoint_and_distance_are_absent_from_initial_and_retry_observations():
    env = EndpointProcessRLEnv(pilot_row())
    env.assert_no_reward_leakage()
    env.step(invalid())
    env.assert_no_reward_leakage()
    payload = str(env.public_messages()).lower()
    assert "endpoint_distance" not in payload
    assert "expected_precursor" not in payload


def test_unparsed_response_is_a_negative_trainable_event():
    env = EndpointProcessRLEnv(pilot_row())
    credit = env.reject_unparsed(raw_response="not a tool call", token_count=4)
    assert credit.action_name == "invalid_response"
    assert credit.total_reward == -0.25
    assert not env.done
