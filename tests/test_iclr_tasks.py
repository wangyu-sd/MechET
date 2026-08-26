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
    assert "outcome" not in row["messages"][0]["content"].lower()
    assert row["messages"][1]["content"] == "TARGET: [CH3:1][OH:2]"


def test_proof_has_no_answer_channel():
    row = build_proof_row(ROW)
    assert "<proof>" in row["messages"][-1]["content"]
    assert "<answer>" not in row["messages"][-1]["content"]
    assert "MECH_PROOF v1" in row["messages"][0]["content"]


def test_top_level_structural_precursor_is_supported():
    row = dict(ROW)
    row["structural_precursor"] = "[CH3:1][Br:3].[OH-:2]"
    row["metadata"] = {}
    assert "[CH3:1][Br:3]" in build_outcome_only_row(row)["messages"][-1]["content"]


def test_representation_baseline_preserves_source_fallback_without_tool_semantics():
    row = dict(ROW)
    row["metadata"] = {
        **ROW["metadata"],
        "upstream_endpoint_fallback": True,
    }
    output = build_outcome_only_row(row)
    assert "upstream_endpoint_fallback" not in output["metadata"]
    assert output["metadata"]["source_upstream_endpoint_fallback"] is True


def test_net_edit_is_single_step_baseline():
    row = build_net_edit_row(ROW)
    content = row["messages"][-1]["content"]
    assert "NET_EDIT v1" in content
    assert "<answer>" in content
