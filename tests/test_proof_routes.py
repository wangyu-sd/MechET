from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram, format_proof_output, sides_equal
from mechet.proof_routes import best_first_route_search, verify_route


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


def test_best_first_search_admits_only_verified_edges():
    target = "[CH3:1][OH:2]"
    building = ["[CH3:1][Br:3]", "[OH-:2]"]

    def is_building_block(value):
        return any(sides_equal(value, item, ignore_maps=True) for item in building)

    def expand(value):
        if sides_equal(value, target, ignore_maps=True):
            return [
                {"proof": "MECH_PROOF v1\nnot valid", "model_score": 2.0},
                {"proof": proof(), "model_score": 1.0},
            ]
        return []

    routes, stats = best_first_route_search(
        target,
        expand=expand,
        is_building_block=is_building_block,
        max_nodes=20,
    )
    assert len(routes) == 1
    assert stats["invalid_expansions"] == 1
    verification = verify_route(target, routes[0].steps, is_building_block=is_building_block)
    assert verification.ok
    assert verification.n_executable_steps == 1
