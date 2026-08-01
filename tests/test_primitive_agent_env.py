import json
from pathlib import Path

from mechet.primitive_agent_env import PrimitiveAgentConfig, PrimitiveAugmentedAgentEnv

REPO = Path(__file__).resolve().parents[1]


def environment(reward_scale: float = 0.0):
    return PrimitiveAugmentedAgentEnv(
        config=PrimitiveAgentConfig(
            primitive_library_path=str(REPO / "knowledge/primitives/core_polar_primitives.yaml"),
            primitive_source_registry_path=str(REPO / "knowledge/source_registry.yaml"),
            primitive_reward_scale=reward_scale,
            require_tool_use=True,
            max_tool_calls=8,
        )
    )


def test_environment_exposes_primitive_retrieval_tool():
    env = environment()
    observation = json.loads(env.reset(target_smiles="[CH3:1][Br:2].[OH-:3]", expected_precursor="[CH3:1][Br:2].[OH-:3]"))
    assert observation["primitive_library"]["enabled"]
    result = json.loads(env.retrieve_primitives(query="substitution"))
    assert result["ok"]
    assert any(item["primitive_id"] == "nucleophilic_substitution_sp3" for item in result["matches"])


def test_supported_move_receives_only_bounded_soft_bonus():
    env = environment(reward_scale=0.1)
    env.reset(target_smiles="[CH3:1][Br:2].[OH-:3]", expected_precursor="[CH3:1][Br:2].[OH-:3]")
    result = json.loads(env.apply_coupled_electron_moves(json.dumps([
        {"source": {"kind": "LP", "atoms": [3]}, "sink": {"kind": "BOND", "atoms": [1, 3]}},
        {"source": {"kind": "BOND", "atoms": [1, 2]}, "sink": {"kind": "ATOM", "atoms": [2]}},
    ])))
    assert result["ok"]
    assert result["primitive_support"]["supported"]
    assert env.primitive_support_total > 0
    assert result["primitive_support"]["soft_evidence_only"]
