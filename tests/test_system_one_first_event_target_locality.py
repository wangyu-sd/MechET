from scripts.audit_system_one_first_event_target_locality import reference_touched_atoms


def test_reference_locality_reads_all_electron_and_delta_atom_handles():
    arguments = {
        "electron_flow": [{"source": "a lone pair on atom A03",
                           "destination": "the bond between atoms A03 and A12"}],
        "bond_order_changes": [{"atoms": ["A04", "A05"], "delta": -1}],
        "charge_changes": [{"atom": "A07", "from": 0, "to": 1}],
    }
    assert reference_touched_atoms(arguments) == {2, 3, 4, 6, 11}
