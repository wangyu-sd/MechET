from mechet.data_audit import record_from_mechet_row


def test_record_from_current_agent_row_uses_top_level_endpoints() -> None:
    row = {
        "id": "example",
        "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": "[CH3:1][Br:2]",
        "messages": [],
        "metadata": {},
    }
    record = record_from_mechet_row(row)
    assert record.product == "[CH3:1][OH:2]"
    assert record.reactants == "[CH3:1][Br:2]"
