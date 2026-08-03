from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    format_proof_output,
)
from mechet.proof_splits import build_compositional_ood_split
from mechet.proof_to_trace import proof_to_trace_plan


def edge_a(src: str, dst: str) -> ProofEdge:
    return ProofEdge(
        src,
        dst,
        bonds=[(1, 2, -1), (1, 3, +1)],
        lone_pairs=[(2, +2), (3, -2)],
        charges=[ChargeAction(2, 0, -1), ChargeAction(3, -1, 0)],
    )


def edge_b(src: str, dst: str) -> ProofEdge:
    return ProofEdge(
        src,
        dst,
        bonds=[(4, 5, -1), (4, 6, +1)],
        lone_pairs=[(5, +2), (6, -2)],
        charges=[ChargeAction(5, 0, -1), ChargeAction(6, -1, 0)],
    )


def row(row_id: str, kind: str) -> dict:
    if kind == "a":
        edges = [edge_a("s0", "s1")]
        precursor = "s1"
    elif kind == "b":
        edges = [edge_b("s0", "s1")]
        precursor = "s1"
    else:
        edges = [edge_a("s0", "s1"), edge_b("s1", "s2")]
        precursor = "s2"
    program = ProofProgram(
        target_smiles="[CH3:1][OH:2].[CH3:4][OH:5]",
        roots={"s0": ["[Br-:3]", "[Cl-:6]"]},
        precursor_state_id=precursor,
        edges=edges,
    )
    proof = format_proof_output(program)
    plan = proof_to_trace_plan(proof)
    return {
        "id": row_id,
        "artifact_type": "supervision",
        "target_smiles": plan.target_smiles,
        "expected_precursor": plan.expected_precursor,
        "metadata": {
            "executor_replayed": True,
            "trace_plan": plan.to_dict(),
        },
    }


def test_mechcomp_split_holds_out_move_compositions_but_not_moves():
    rows = [
        *(row(f"a-{index}", "a") for index in range(3)),
        *(row(f"b-{index}", "b") for index in range(3)),
        *(row(f"ab-{index}", "ab") for index in range(2)),
    ]
    splits, manifest = build_compositional_ood_split(
        rows,
        test_fraction=0.25,
        valid_fraction=0.0,
        min_train_primitive_count=2,
        seed=7,
    )
    assert len(splits["test"]) == 2
    assert {item["id"] for item in splits["test"]} == {"ab-0", "ab-1"}
    assert manifest["primitive_basis"] == "source_to_sink_execution_moves_v1"
    assert manifest["composition_overlap"]["train_test"] == 0
    assert manifest["heldout_primitive_coverage"]["test"] == 1.0
    assert manifest["claim_gate"]["test_primitives_seen_in_train"]
    assert all(
        item["metadata"]["mechcomp_split"] == split
        for split, split_rows in splits.items()
        for item in split_rows
    )
