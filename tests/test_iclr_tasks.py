from mechet.iclr_tasks import build_net_edit_row, build_outcome_only_row, build_proof_row


ROW = {
    "id": "x",
    "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "TARGET: [CH3:1][OH:2]"},
        {"role": "assistant", "content": '''<proof>\nMECH_PROOF v1\nTARGET_SMILES "[CH3:1][OH:2]"\nROOT s0\n  IMPORT "[Br-:3]"\nPRECURSOR_STATE s1\nEDGE s0 s1\n  BOND 1 2 -1\n  BOND 1 3 +1\n  LP 2 +2\n  LP 3 -2\n  CHARGE 2 0 -1\n  CHARGE 3 -1 0\n</proof>'''},
    ],
    "metadata": {"derived_precursor": "[CH3:1][Br:3].[OH-:2].[Na+]"},
}


def test_outcome_uses_structural_precursor_only():
    row = build_outcome_only_row(ROW)
    answer = row["messages"][-1]["content"]
    assert "[Na+]" not in answer
    assert "[CH3:1][Br:3]" in answer


def test_proof_has_no_answer_channel():
    row = build_proof_row(ROW)
    assert "<proof>" in row["messages"][-1]["content"]
    assert "<answer>" not in row["messages"][-1]["content"]


def test_net_edit_is_single_step_baseline():
    row = build_net_edit_row(ROW)
    content = row["messages"][-1]["content"]
    assert "NET_EDIT v1" in content
    assert "<answer>" in content
