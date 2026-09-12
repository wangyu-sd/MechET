from mechet.forward_expert import ElectronContainer
from mechet.transactional_event_space import MoveInventory, audit_reference_event


STATE = "[O-:1].[CH3:2][Br:3]"


def test_inventory_contains_substitution_arrows_without_gold_input():
    inventory = MoveInventory.from_state(STATE)
    donor = ElectronContainer("LP", (1,))
    carbon_bromine = ElectronContainer("BOND", (2, 3))

    assert donor in inventory.sources
    assert ElectronContainer("BOND", (1, 2)) in inventory.compatible_sinks(donor)
    assert ElectronContainer("ATOM", (3,)) in inventory.compatible_sinks(carbon_bromine)


def test_reference_is_used_only_to_audit_state_derived_inventory():
    event = [
        {
            "source": {"kind": "LP", "atoms": [1]},
            "sink": {"kind": "BOND", "atoms": [1, 2]},
            "electrons": 2,
        },
        {
            "source": {"kind": "BOND", "atoms": [2, 3]},
            "sink": {"kind": "ATOM", "atoms": [3]},
            "electrons": 2,
        },
    ]
    result = audit_reference_event(STATE, event)
    assert result["covered"]
    assert result["source_candidates"] >= 2
    assert all(value >= 1 for value in result["gold_conditioned_sink_candidates"])


def test_radical_pair_is_explicitly_outside_first_polar_inventory():
    event = [
        {
            "source": {"kind": "RADICAL_PAIR", "atoms": [1, 2]},
            "sink": {"kind": "BOND", "atoms": [1, 2]},
            "electrons": 2,
        }
    ]
    result = audit_reference_event(STATE, event)
    assert not result["covered"]
    assert result["reason"] == "MOVE_OUTSIDE_POLAR_SPACE"


def test_metal_ligand_radical_pair_has_separate_gt_independent_support():
    state = "[Pd:1].[P:2]([CH3:3])([CH3:4])[CH3:5]"
    event = [
        {
            "source": {"kind": "RADICAL_PAIR", "atoms": [1, 2]},
            "sink": {"kind": "BOND", "atoms": [1, 2]},
            "electrons": 2,
        }
    ]
    assert audit_reference_event(state, event)["covered"]


def test_aromatic_lone_pair_inventory_matches_executor_kekule_graph():
    state = "[n:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[H:7][H:8]"
    inventory = MoveInventory.from_state(state)
    assert ElectronContainer("LP", (1,)) in inventory.sources
