from mechet.endpoint_progress import (
    capped_progress_increment,
    endpoint_distance,
    potential_progress_reward,
)


def test_exact_distance_is_zero_and_component_order_invariant():
    first = "[CH3:1][OH:2].[Na+:3]"
    second = "[Na+:3].[OH:2][CH3:1]"
    value = endpoint_distance(first, second, target_atom_maps=[1, 2])
    assert value.total == 0.0


def test_distance_is_invariant_after_frozen_identity_alignment():
    reference = "[CH3:1][OH:2]"
    relabelled = "[CH3:21][OH:17]"
    value = endpoint_distance(
        relabelled,
        reference,
        target_atom_maps=[1, 2],
        current_to_reference_maps={21: 1, 17: 2},
    )
    assert value.total == 0.0


def test_charge_and_bond_order_changes_are_detected():
    reference = "[NH2:1][OH:2]"
    charge = endpoint_distance(
        "[NH3+:1][O-:2]", reference, target_atom_maps=[1, 2]
    )
    bond = endpoint_distance(
        "[NH:1]=[O:2]", reference, target_atom_maps=[1, 2]
    )
    assert charge.charge > 0
    assert bond.bond > 0


def test_missing_contributing_fragment_increases_distance_but_gold_spectator_does_not():
    reference = "[CH3:1][OH:2].[Cl-:3].[Na+:4]"
    missing = endpoint_distance(
        "[CH3:1][OH:2].[Na+:4]",
        reference,
        target_atom_maps=[1, 2],
        contributing_atom_maps=[3],
    )
    exact_with_reordered_spectator = endpoint_distance(
        "[Na+:4].[Cl-:3].[OH:2][CH3:1]",
        reference,
        target_atom_maps=[1, 2],
        contributing_atom_maps=[3],
    )
    assert missing.fragment > 0
    assert exact_with_reordered_spectator.total == 0.0


def test_progress_reward_and_rollout_cap():
    start = endpoint_distance(
        "[O-:1].[CH3:2][Br:3]",
        "[O:1][CH3:2].[Br-:3]",
        target_atom_maps=[1, 2, 3],
    )
    end = endpoint_distance(
        "[O:1][CH3:2].[Br-:3]",
        "[O:1][CH3:2].[Br-:3]",
        target_atom_maps=[1, 2, 3],
    )
    reward = potential_progress_reward(start, end, start)
    assert reward == 0.25
    applied, total = capped_progress_increment(0.9, reward)
    assert abs(applied - 0.1) < 1e-12
    assert total == 1.0
