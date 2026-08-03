import json

import pytest

from mechet.agent_env import AgentEnvConfig, MechETAgentEnv
from mechet.agent_inference import ParsedToolCall, execute_tool_call
from mechet.electron_flow_trace import ElectronFlowTrace, compile_trace_to_proof
from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.knowledge_ablation import (
    align_prediction_artifact,
    condition_metrics,
    make_direct_textbook_condition,
)
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
