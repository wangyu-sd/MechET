import json
from pathlib import Path

import torch

from mechet.forward_data import normalize_reaction_row, standardize_path
from mechet.forward_expert import (
    ElectronMove,
    ForwardElectronExpert,
    enumerate_containers,
    forward_edge_cost,
    score_reaction,
    verify_electron_step,
)


def sn2_moves():
    return [
        {
            "source": {"kind": "LP", "atoms": [2]},
            "sink": {"kind": "BOND", "atoms": [1, 2]},
        },
        {
            "source": {"kind": "BOND", "atoms": [1, 3]},
            "sink": {"kind": "ATOM", "atoms": [3]},
        },
    ]


def test_coupled_sn2_arrows_are_applied_atomically():
    result = verify_electron_step("[CH3:1][Br:3].[OH-:2]", sn2_moves())
    assert result["ok"]
    assert result["state_smiles"] == "[Br-:3].[CH3:1][OH:2]"


def test_nonlocal_bond_shift_is_rejected():
    move = {
        "source": {"kind": "BOND", "atoms": [1, 2]},
        "sink": {"kind": "BOND", "atoms": [3, 4]},
    }
    result = verify_electron_step(
        "[CH3:1][CH3:2].[CH3:3][CH3:4]",
        [move],
    )
    assert not result["ok"]
    assert "NONLOCAL_BOND_SHIFT" in result.get("message", "")


def test_graph_pointer_expert_ranks_finite_moves_and_scores_competitors():
    torch.manual_seed(1)
    model = ForwardElectronExpert(
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    ).eval()
    ranked = model.rank_moves("[CH3:1][Br:3].[OH-:2]", top_k=5)
    assert len(ranked) == 5
    assert all(torch.isfinite(torch.tensor(item["logprob"])) for item in ranked)
    evidence = score_reaction(
        model,
        "[CH3:1][Br:3].[OH-:2]",
        "[Br-:3].[CH3:1][OH:2]",
        ["[CH2:1]=[CH2:2]"],
    )
    assert evidence.target_rank in {1, 2}
    assert evidence.selectivity_margin is not None
    assert forward_edge_cost(-1.0, evidence) != float("inf")


def test_standardization_preserves_source_sink_labels(tmp_path: Path):
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "sn2-1",
                "reaction_smiles": (
                    "[CH3:1][Br:3].[OH-:2]>>[Br-:3].[CH3:1][OH:2]"
                ),
                "moves": sn2_moves(),
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "standard.jsonl"
    report = standardize_path(source, output, source="unit")
    assert report.written == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["steps"][0]["moves"][0]["source"]["kind"] == "LP"
    assert row["steps"][0]["moves"][1]["sink"]["kind"] == "ATOM"


def test_container_inventory_contains_gold_sn2_actions():
    sources, sinks = enumerate_containers("[CH3:1][Br:3].[OH-:2]")
    moves = [ElectronMove.parse(value) for value in sn2_moves()]
    assert moves[0].source in sources and moves[0].sink in sinks
    assert moves[1].source in sources and moves[1].sink in sinks


def test_mechanistic_smiles_adapter_decodes_arrow_pairs():
    row = normalize_reaction_row(
        {
            "rxn_idx": 7,
            "mech_smi_ori": "[CH3:1][Br:3].[OH-:2]|(2, 1);((1, 3), 3)",
            "elem_prod_spe": "[Br-:3].[CH3:1][OH:2]",
            "rxn_prod_spe": "[Br-:3].[CH3:1][OH:2]",
            "split": "train",
        },
        source="mech_uspto_31k",
        row_index=0,
    )
    assert [
        ElectronMove.parse(value).id for value in row["steps"][0]["moves"]
    ] == [ElectronMove.parse(value).id for value in sn2_moves()]
    assert row["steps"][0]["state_smiles"] == "[CH3:1][Br:3].[OH-:2]"


def test_pmechdb_arrow_code_adapter():
    row = normalize_reaction_row(
        {
            "SMIRKS": (
                "[CH3:1][Br:3].[OH-:2]>>[Br-:3].[CH3:1][OH:2]"
            ),
            "Arrow Codes": "2=1;1,3=3",
            "Orbital Pair Classification": "SN2",
            "split": "train",
        },
        source="pmechdb",
        row_index=0,
    )
    ids = [
        ElectronMove.parse(value).id for value in row["steps"][0]["moves"]
    ]
    assert ids == ["LP:2->BOND:1,2/2e", "BOND:1,3->ATOM:3/2e"]


def test_complete_inverse_proof_can_receive_independent_forward_reward():
    from mechet import proof_program
    from mechet import forward_rewards

    program = proof_program.ProofProgram(
        target_smiles="[CH3:1][OH:2]",
        roots={"s0": ["[Br-:3]"]},
        precursor_state_id="s1",
        edges=[
            proof_program.ProofEdge(
                "s0",
                "s1",
                bonds=[(1, 2, -1), (1, 3, +1)],
                lone_pairs=[(2, +2), (3, -2)],
                charges=[
                    proof_program.ChargeAction(2, 0, -1),
                    proof_program.ChargeAction(3, -1, 0),
                ],
            )
        ],
    )
    model = ForwardElectronExpert(
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    ).eval()
    result = forward_rewards.score_inverse_proof_forward(
        model,
        proof_program.format_proof_output(program),
    )
    assert result["formal_execute"]
    assert isinstance(result["forward_reward"], float)
