import hashlib

import pytest

from mechet.knowledge_agent_env import KnowledgeAgentConfig, KnowledgeAugmentedAgentEnv
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram, format_proof_output
from mechet.proof_to_trace import infer_moves_from_edge, proof_to_trace_plan, replay_trace_plan
from mechet.textbook_store import TextbookPassage, TextbookStore


def sn2_proof():
    program = ProofProgram(
        target_smiles="[CH3:1][OH:2]",
        roots={"s0": []},
        precursor_state_id="s1",
        edges=[
            ProofEdge(
                "s0",
                "s1",
                imports=["[Br-:3]"],
                bonds=[(1, 2, -1), (1, 3, +1)],
                lone_pairs=[(2, +2), (3, -2)],
                charges=[
                    ChargeAction(2, 0, -1),
                    ChargeAction(3, -1, 0),
                ],
            )
        ],
    )
    return format_proof_output(program)


def write_corpus(path):
    text = "Nucleophilic substitution can couple nucleophile bond formation with leaving group departure in one elementary event."
    TextbookStore(
        [
            TextbookPassage(
                passage_id="sn2-text",
                title="Nucleophilic substitution",
                text=text,
                source_id="open_textbook",
                locator="substitution",
                revision="r1",
                license="CC-BY-4.0",
                source_url="https://example.org",
                evidence_sha256=hashlib.sha256(text.encode()).hexdigest(),
                topics=("substitution",),
            )
        ]
    ).save(path)


def test_sn2_proof_converts_to_two_coupled_moves():
    plan = proof_to_trace_plan(sn2_proof())
    assert len(plan.steps) == 1
    assert plan.steps[0].imports == ("[Br-:3]",)
    moves = plan.steps[0].moves
    assert len(moves) == 2
    assert {move["source"]["kind"] for move in moves} == {"LP", "BOND"}
    assert plan.expected_precursor


def test_converted_plan_replays_through_trace_owned_environment(tmp_path):
    corpus = tmp_path / "passages.jsonl"
    write_corpus(corpus)
    env = KnowledgeAugmentedAgentEnv(
        config=KnowledgeAgentConfig(
            textbook_corpus_path=str(corpus),
            require_textbook_corpus=True,
            max_tool_calls=10,
        )
    )
    replay = replay_trace_plan(env, proof_to_trace_plan(sn2_proof()))
    assert replay["terminal"]["endpoint_exact"]
    assert replay["terminal"]["trace_bound"]
    assert replay["terminal"]["compiled_proof"]


def test_ambiguous_electron_pairing_is_rejected():
    edge = ProofEdge(
        "s0",
        "s1",
        bonds=[(1, 2, -1), (3, 4, -1), (1, 3, +1), (2, 4, +1)],
        lone_pairs=[],
        charges=[],
    )
    with pytest.raises(ValueError, match="AMBIGUOUS_ELECTRON_PAIRING"):
        infer_moves_from_edge(edge)


def test_unpaired_lone_pair_delta_is_rejected():
    edge = ProofEdge(
        "s0",
        "s1",
        bonds=[(1, 2, -1)],
        lone_pairs=[(3, +2)],
        charges=[],
    )
    with pytest.raises(ValueError):
        infer_moves_from_edge(edge)
