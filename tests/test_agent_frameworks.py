import json

from mechet.adversarial import MiningConfig, mine_forward_audit_candidates
from mechet.agent_env import AgentEnvConfig, MechETAgentEnv
from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    format_proof_output,
)
from mechet.syntheseus_adapter import MechETCandidatePool


def substitution_proof() -> str:
    return format_proof_output(
        ProofProgram(
            target_smiles="[CH3:1][OH:2]",
            roots={"s0": ["[Br-:3]"]},
            precursor_state_id="s1",
            edges=[
                ProofEdge(
                    "s0",
                    "s1",
                    bonds=[(1, 2, -1), (1, 3, +1)],
                    lone_pairs=[(2, +2), (3, -2)],
                    charges=[
                        ChargeAction(2, 0, -1),
                        ChargeAction(3, -1, 0),
                    ],
                )
            ],
        )
    )


def test_agent_environment_inspects_moves_and_submits_proof():
    env = MechETAgentEnv(
        config=AgentEnvConfig(require_tool_use=True, max_tool_calls=6)
    )
    observation = json.loads(
        env.reset(
            target_smiles="[CH3:1][OH:2]",
            expected_precursor="[CH3:1][Br:3].[OH-:2]",
        )
    )
    assert observation["task"] == "inverse_electron_flow"
    inventory = json.loads(env.inspect_state())
    assert inventory["ok"]
    assert any(item["id"] == "BOND:1,2" for item in inventory["sources"])
    move = json.loads(
        env.apply_electron_move(
            source_kind="BOND",
            source_atoms=[1, 2],
            sink_kind="ATOM",
            sink_atoms=[2],
        )
    )
    assert move["ok"]
    result = json.loads(env.submit_proof(substitution_proof()))
    assert result["formal_execute"]
    assert result["endpoint_exact"]
    assert result["successful_steps"] == 1
    assert env.get_reward() > 0


def test_agent_environment_can_abstain():
    env = MechETAgentEnv()
    env.reset(target_smiles="[CH3:1][OH:2]")
    result = json.loads(env.abstain("unsupported reaction family"))
    assert result["abstained"]
    assert env.get_reward() == env.config.abstain_reward


def test_candidate_pool_parses_hypotheses_and_ignores_maps(tmp_path):
    path = tmp_path / "hypotheses.jsonl"
    path.write_text(
        json.dumps(
            {
                "hypotheses": [
                    {
                        "proof": substitution_proof(),
                        "execute_ok": True,
                        "derived_core_precursor": "[CH3:1][Br:3].[OH-:2]",
                        "model_logprob": -0.4,
                        "forward_evidence": {
                            "target_score": 0.8,
                            "selectivity_margin": 0.3,
                            "uncertainty": 0.1,
                        },
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pool = MechETCandidatePool.from_jsonl(path)
    assert pool.n_targets == 1
    assert pool.n_candidates == 1
    values = pool.query("CO", num_results=5)
    assert len(values) == 1
    assert "Br" in values[0].precursor


def test_actor_verifier_disagreement_requires_independent_audit(monkeypatch):
    class Evidence:
        target_score = 0.91
        selectivity_margin = 0.02

        def to_dict(self):
            return {
                "target_score": self.target_score,
                "selectivity_margin": self.selectivity_margin,
            }

    monkeypatch.setattr(
        "mechet.adversarial.score_reaction",
        lambda *args, **kwargs: Evidence(),
    )
    rows = [
        {
            "id": "sample-1",
            "hypotheses": [
                {
                    "proof": substitution_proof(),
                    "execute_ok": True,
                    "endpoint_exact": False,
                    "model_logprob": -0.1,
                }
            ],
        }
    ]
    candidates = mine_forward_audit_candidates(
        object(),
        rows,
        config=MiningConfig(minimum_target_score=0.8),
    )
    assert len(candidates) == 1
    assert candidates[0]["label"] is None
    assert candidates[0]["training_eligible"] is False
    assert candidates[0]["audit_status"] == "unreviewed"
    assert candidates[0]["source"] == "inverse_actor_forward_disagreement"
