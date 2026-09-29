"""Pinned SynEPD-to-MechET public-decision conversion tests."""

import pytest

from scripts.autoresearch.build_synepd_curated import convert_record, principal_product


def test_single_product_inverse_event_replays_without_import():
    record = {
        "id": 17,
        "tax_code": "POLAR.01.03.001",
        "rsmi": "[O:1]=[CH:3][CH2:2][H:4]>>[O:1]([CH:3]=[CH2:2])[H:4]",
        "epd": [
            ["LP-/Sigma+", [1], [1, 4]],
            ["Sigma-/Pi+", [2, 4], [2, 3]],
            ["Pi-/LP+", [1, 3], [1]],
        ],
    }
    rows = convert_record(record)
    assert [row["metadata"]["decision_type"] for row in rows] == ["event", "finish"]
    assert all(row["metadata"]["mechanism_class"] == "POLAR.01" for row in rows)
    assert all(row["metadata"]["executor_replayed"] for row in rows)


def test_multi_product_import_is_public_and_replayed():
    record = {
        "id": 1,
        "tax_code": "POLAR.01.01.001",
        "rsmi": "[NH3+:2][H:4].[O-:1][CH3:3]>>[NH3:2].[O:1]([CH3:3])[H:4]",
        "epd": [
            ["LP-/Sigma+", [1], [1, 4]],
            ["Sigma-/LP+", [2, 4], [2]],
        ],
    }
    rows = convert_record(record)
    assert [row["metadata"]["decision_type"] for row in rows] == ["import", "event", "finish"]
    assert rows[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "import_fragments"
    assert all(":1]" not in row["target_smiles"] for row in rows)


def test_ambiguous_principal_product_is_not_silently_chosen():
    with pytest.raises(ValueError, match="ambiguous principal product"):
        principal_product("[CH3:1][Cl:2].[CH3:3][Br:4]")
