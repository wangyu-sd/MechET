import pytest

from mechet.jevretro_endpoint import (
    apply_endpoint_program,
    canonical_unmapped,
    derive_endpoint_program,
    program_summary,
)


def test_sn2_endpoint_program_is_lossless():
    product = "[CH3:1][OH:2]"
    precursor = "[CH3:1][Br:3].[OH-:2]"
    program = derive_endpoint_program(product, precursor)

    assert program["schema"] == "jevretro_endpoint_program_v1"
    assert len(program["bond_edits"]) == 1
    assert program["bond_edits"][0]["maps"] == [1, 2]
    assert program["bond_edits"][0]["to"] is None
    assert len(program["attachments"]) == 1
    assert program["attachments"][0]["slots"][0]["anchor_map"] == 1
    # Precursor map numbers must not become template vocabulary identifiers.
    assert ":3" not in program["attachments"][0]["template"]
    assert apply_endpoint_program(product, program) == canonical_unmapped(precursor)


def test_bond_order_endpoint_program_is_lossless():
    product = "[CH2:1]=[CH2:2]"
    precursor = "[CH3:1][CH3:2]"
    program = derive_endpoint_program(product, precursor)
    assert len(program["bond_edits"]) == 1
    assert program["bond_edits"][0]["from"]["type"] == "DOUBLE"
    assert program["bond_edits"][0]["to"]["type"] == "SINGLE"
    assert apply_endpoint_program(product, program) == canonical_unmapped(precursor)


def test_noop_endpoint_program_is_lossless():
    product = "[CH3:1][OH:2]"
    program = derive_endpoint_program(product, product)
    assert program_summary(program) == {
        "atom_edits": 0,
        "bond_edits": 0,
        "delete_product_atoms": 0,
        "attachments": 0,
        "attachment_slots": 0,
    }
    assert apply_endpoint_program(product, program) == canonical_unmapped(product)


def test_unmapped_endpoint_is_rejected():
    with pytest.raises(ValueError, match="unmapped atom"):
        derive_endpoint_program("CO", "CO")
