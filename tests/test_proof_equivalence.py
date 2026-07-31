from mechet.proof_equivalence import (
    canonical_partial_order_signature,
    composition_signature,
    proofs_equivalent,
)
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram


def two_substitution_program(*, reverse: bool = False) -> ProofProgram:
    bromide = ProofEdge(
        "s0" if not reverse else "s1",
        "s1" if not reverse else "s2",
        bonds=[(1, 2, -1), (1, 3, +1)],
        lone_pairs=[(2, +2), (3, -2)],
        charges=[
            ChargeAction(2, 0, -1),
            ChargeAction(3, -1, 0),
        ],
    )
    chloride = ProofEdge(
        "s1" if not reverse else "s0",
        "s2" if not reverse else "s1",
        bonds=[(4, 5, -1), (4, 6, +1)],
        lone_pairs=[(5, +2), (6, -2)],
        charges=[
            ChargeAction(5, 0, -1),
            ChargeAction(6, -1, 0),
        ],
    )
    edges = [bromide, chloride] if not reverse else [chloride, bromide]
    return ProofProgram(
        target_smiles="[CH3:1][OH:2].[CH3:4][OH:5]",
        roots={"s0": ["[Br-:3]", "[Cl-:6]"]},
        precursor_state_id="s2",
        edges=edges,
    )


def substitution_with_maps(offset: int) -> ProofProgram:
    carbon, oxygen, bromine = offset + 1, offset + 2, offset + 3
    return ProofProgram(
        target_smiles=f"[CH3:{carbon}][OH:{oxygen}]",
        roots={"root": [f"[Br-:{bromine}]"]},
        precursor_state_id="precursor",
        edges=[
            ProofEdge(
                "root",
                "precursor",
                bonds=[
                    (carbon, oxygen, -1),
                    (carbon, bromine, +1),
                ],
                lone_pairs=[
                    (oxygen, +2),
                    (bromine, -2),
                ],
                charges=[
                    ChargeAction(oxygen, 0, -1),
                    ChargeAction(bromine, -1, 0),
                ],
            )
        ],
    )


def test_commuting_events_are_order_invariant():
    forward = two_substitution_program(reverse=False)
    reverse = two_substitution_program(reverse=True)
    assert proofs_equivalent(forward, reverse)
    assert composition_signature(forward) == composition_signature(reverse)
    signature = canonical_partial_order_signature(forward)
    assert not signature.dependency_counts


def test_signature_is_invariant_to_atom_map_labels():
    assert proofs_equivalent(
        substitution_with_maps(0),
        substitution_with_maps(10),
    )
