import json

import pytest

from mechet.agent_env import AgentEnvConfig, MechETAgentEnv
from mechet.agent_inference import (
    ParsedToolCall,
    append_tool_exchange,
    execute_tool_call,
    tool_result_pool,
)
from mechet.electron_flow_trace import ElectronFlowTrace, compile_trace_to_proof
from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.knowledge_ablation import (
    align_prediction_artifact,
    condition_metrics,
    make_direct_textbook_condition,
)
from mechet.prediction_metrics import prediction_set_metrics
from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    format_proof_output,
)
from mechet.proof_splits import extract_split_features
from mechet.proof_to_trace import proof_to_trace_plan, replay_trace_plan
from mechet.trace_agent_env import TraceOwnedAgentEnv
from mechet.trl_environments import TraceOwnedTRLEnvironment


def root_import_sn2_proof() -> str:
    return format_proof_output(
        ProofProgram(
            target_smiles="[CH3:1][OH:2]",
            roots={"s0": ["[Br-:3]"]},
            precursor_state_id="s1",
            edges=[
                ProofEdge(
                    "s0",
                    "s1",
                    bonds=[(1, 2, -1), (1, 3, 1)],
                    lone_pairs=[(2, 2), (3, -2)],
                    charges=[
                        ChargeAction(2, 0, -1),
                        ChargeAction(3, -1, 0),
                    ],
                )
            ],
        )
    )


def test_root_imports_are_preserved_and_replayed():
    plan = proof_to_trace_plan(root_import_sn2_proof())
    assert plan.initial_imports == ("[Br-:3]",)
    assert plan.steps[0].imports == ()
    env = TraceOwnedAgentEnv(config=AgentEnvConfig(max_tool_calls=8))
    replay = replay_trace_plan(env, plan)
    assert replay["terminal"]["formal_execute"]
    assert replay["terminal"]["endpoint_exact"]
    assert replay["terminal"]["declared_moves_replayed"]


def test_compiler_rejects_moves_that_do_not_generate_recorded_state():
    trace = ElectronFlowTrace("[CH3:1][OH:2]")
    trace.append(
        state_before="[CH3:1][OH:2]",
        state_after="[CH3:1][Br:3].[OH-:2]",
        imports=["[Br-:3]"],
        moves=[
            {
                "source": {"kind": "BOND", "atoms": [1, 2]},
                "sink": {"kind": "ATOM", "atoms": [2]},
                "electrons": 2,
            }
        ],
    )
    with pytest.raises(ValueError, match="TRACE_MOVE_STATE_MISMATCH"):
        compile_trace_to_proof(trace)


def test_invalid_coupled_call_consumes_budget_and_failure():
    env = MechETAgentEnv(config=AgentEnvConfig(max_tool_calls=3))
    env.reset(target_smiles="[CH3:1][OH:2]")
    result = json.loads(env.apply_coupled_electron_moves("not-json"))
    assert result["code"] == "MOVE_JSON_INVALID"
    assert env.tool_calls == 1
    assert env.failed_steps == 1
    assert env.trace[-1]["event"] == "apply_moves"


def test_trace_facade_hides_internal_and_legacy_methods():
    env = TraceOwnedTRLEnvironment()
    public = {
        name
        for name in dir(env)
        if not name.startswith("_") and name not in {"reset", "get_reward"}
    }
    assert "finish_trace" in public
    assert "state_dict" not in public
    assert "submit_proof" not in public
    assert "_snapshot" not in public


def test_unknown_inference_tool_consumes_environment_budget():
    env = TraceOwnedTRLEnvironment(config=AgentEnvConfig(max_tool_calls=3))
    env.reset(target_smiles="[CH3:1][OH:2]")
    result = json.loads(
        execute_tool_call(env, ParsedToolCall("x", "state_dict", {}))
    )
    assert result["code"] == "TOOL_NOT_AVAILABLE"
    state = env._snapshot()
    assert state["tool_calls"] == 1
    assert state["failed_steps"] == 1


def test_terminal_multicall_turn_keeps_one_result_per_call():
    env = TraceOwnedTRLEnvironment(config=AgentEnvConfig(max_tool_calls=3))
    env.reset(target_smiles="[CH3:1][OH:2]")
    messages = []
    calls = [
        ParsedToolCall("terminal", "abstain", {"reason": "unsupported"}),
        ParsedToolCall("late", "inspect_state", {}),
    ]
    exchanges = append_tool_exchange(messages, "", calls, env)
    tool_messages = [item for item in messages if item.get("role") == "tool"]
    assert len(tool_messages) == len(calls) == len(exchanges)
    assert exchanges[0]["executed"] is True
    assert exchanges[1]["executed"] is False
    assert exchanges[1]["result"]["code"] == "SKIPPED_AFTER_TERMINAL"
    assert tool_messages[1]["tool_call_id"] == "late"


def _prediction_row(identifier: str, target: str, observation: str):
    return {
        "id": identifier,
        "target_smiles": target,
        "messages": [
            {
                "role": "tool",
                "name": "inspect_state",
                "tool_call_id": f"{identifier}-call",
                "content": observation,
            }
        ],
    }


def test_shuffle_pool_uses_cross_target_donors_and_records_manifest():
    first = json.dumps({"ok": True, "state_smiles": "[CH4:1]"})
    second = json.dumps({"ok": True, "state_smiles": "[NH3:2]"})
    pool = tool_result_pool(
        [
            _prediction_row("a", "[CH4:1]", first),
            _prediction_row("b", "[NH3:2]", second),
        ]
    )
    assert pool["contract"]["all_observed_tools_have_donors"]
    for target, tools in pool["assignments"].items():
        donor = tools["inspect_state"]
        assert donor["donor_target_smiles"] != target
        assert donor["values"]
    assert pool["contract"]["donor_manifest_sha256"]


def test_shuffle_pool_never_self_donates_when_only_one_target_exists():
    pool = tool_result_pool(
        [_prediction_row("a", "[CH4:1]", json.dumps({"ok": True}))]
    )
    assert not pool["contract"]["all_observed_tools_have_donors"]
    assert pool["assignments"]["[CH4:1]"] == {}
    assert pool["unavailable"]["[CH4:1]"] == ["inspect_state"]


def test_removed_observation_preserves_json_shape_control_fields_and_length():
    normal = TraceOwnedTRLEnvironment(config=AgentEnvConfig(max_tool_calls=4))
    removed = TraceOwnedTRLEnvironment(
        config=AgentEnvConfig(max_tool_calls=4),
        intervention="remove_tool_observations",
    )
    normal.reset(target_smiles="[CH3:1][OH:2]")
    removed.reset(target_smiles="[CH3:1][OH:2]")
    raw = normal.inspect_state()
    masked = removed.inspect_state()
    raw_json = json.loads(raw)
    masked_json = json.loads(masked)
    assert len(masked) == len(raw)
    assert set(masked_json) == set(raw_json)
    assert masked_json["ok"] == raw_json["ok"]
    assert masked_json["remaining_tool_calls"] == raw_json["remaining_tool_calls"]
    assert masked_json != raw_json
    assert removed._snapshot()["intervention_audit"][
        "observation_length_preserved"
    ]


def test_endpoint_views_separate_auxiliary_fragments():
    endpoints = split_precursor_endpoints(
        "[CH3:1][Br:3].[OH-:2].[Na+:4]",
        "[CH3:1][OH:2]",
    )
    assert "[Na+:4]" in endpoints.auxiliary
    assert "Na" not in endpoints.structural
    assert structural_exact(
        endpoints.structural,
        "[OH-:9].[CH3:8][Br:7]",
    )


def textbook_supervision_row(identifier="r1"):
    context = {
        "text": "Substitution evidence.",
        "passage_ids": ["p1"],
        "context_sha256": "hash",
        "n_characters": 22,
    }
    return {
        "id": identifier,
        "artifact_type": "supervision",
        "target_smiles": "[CH3:1][OH:2]",
        "expected_precursor": "[CH3:1][Br:3].[OH-:2]",
        "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "TARGET: [CH3:1][OH:2]"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "k",
                        "type": "function",
                        "function": {
                            "name": "retrieve_textbook_guidance",
                            "arguments": {"query": "alcohol"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "k",
                "name": "retrieve_textbook_guidance",
                "content": json.dumps(
                    {"ok": True, "context": context, "direct_reward": False}
                ),
            },
        ],
        "metadata": {
            "textbook_context_characters": 22,
            "endpoint_source": "environment_owned_trace",
        },
    }


def test_direct_control_is_supervision_not_prediction_and_ignores_maps():
    direct = make_direct_textbook_condition(textbook_supervision_row())
    assert direct["artifact_type"] == "supervision"
    prediction = {
        "id": "r1",
        "artifact_type": "prediction",
        "prediction_mode": "direct",
        "prediction": "PRECURSOR: [OH-:9].[CH3:8][Br:7]",
        "messages": [],
    }
    aligned = align_prediction_artifact([direct], [prediction], condition_name="direct")
    metrics = condition_metrics(aligned)
    assert metrics["structural_exact_rate"] == 1.0
    assert metrics["mapped_exact_rate"] == 0.0


def test_unranked_candidate_metrics_are_named_pass_at_k():
    row = {
        "id": "r1",
        "artifact_type": "prediction",
        "prediction_mode": "direct",
        "target_smiles": "[CH4:1]",
        "structural_precursor": "[CH4:1]",
        "prediction": "PRECURSOR: [NH3:2]",
        "candidates": [
            {"prediction": "PRECURSOR: [NH3:2]", "rollout_state": {}},
            {"prediction": "PRECURSOR: [CH4:9]", "rollout_state": {}},
        ],
    }
    metrics = prediction_set_metrics([row], ks=(1, 2))
    assert metrics["structural_endpoint_pass_at_1"] == 0.0
    assert metrics["structural_endpoint_pass_at_2"] == 1.0
    assert metrics["candidate_metric_semantics"] == (
        "generation_order_pass_at_k_not_ranked_top_k"
    )
    assert metrics["true_top_k_available"] is False
    assert "structural_precursor_top2" not in metrics
    assert (
        metrics["deprecated_metric_aliases_not_for_reporting"]
        ["structural_precursor_top2"]
        == 1.0
    )


def test_missing_prediction_is_retained_as_failure():
    references = [
        {"id": "a", "target_smiles": "[CH4:1]", "structural_precursor": "[CH4:1]"},
        {"id": "b", "target_smiles": "[NH3:2]", "structural_precursor": "[NH3:2]"},
    ]
    predictions = [
        {
            "id": "a",
            "artifact_type": "prediction",
            "prediction_mode": "direct",
            "prediction": "PRECURSOR: [CH4:9]",
        }
    ]
    aligned = align_prediction_artifact(references, predictions, condition_name="direct")
    assert len(aligned) == 2
    metrics = condition_metrics(aligned)
    assert metrics["missing_prediction_rate"] == 0.5
    assert metrics["structural_exact_rate"] == 0.5


def test_h2_features_use_explicit_execution_moves():
    plan = proof_to_trace_plan(root_import_sn2_proof())
    row = {
        "id": "sn2",
        "metadata": {
            "executor_replayed": True,
            "trace_plan": plan.to_dict(),
        },
    }
    features = extract_split_features(row)
    assert features.primitives
    assert features.composition
    assert any("source_kind" in value for value in features.primitives)
