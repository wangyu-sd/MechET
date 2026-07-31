from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram, format_proof_output
from mechet.reaction_network import ReactionNetwork, frontier_score, network_digest


def proof() -> str:
    return format_proof_output(
        ProofProgram(
            target_smiles="[CH3:1][OH:2]",
            roots={"s0": ["[Br-:3]"]},
            precursor_state_id="s1",
            edges=[ProofEdge(
                "s0",
                "s1",
                bonds=[(1, 2, -1), (1, 3, +1)],
                lone_pairs=[(2, +2), (3, -2)],
                charges=[ChargeAction(2, 0, -1), ChargeAction(3, -1, 0)],
            )],
        )
    )


def test_network_accepts_executable_hyperedges_and_scores_frontier():
    network = ReactionNetwork()
    edge = network.add_proof(proof(), uncertainty=0.5)
    assert edge is not None
    assert len(network.species) == 3
    assert len(network.edges) == 1
    empty = ReactionNetwork()
    assert frontier_score(edge, empty) > frontier_score(edge, network)
    assert len(network_digest(network)) == 64
