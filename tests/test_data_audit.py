from mechet.data_audit import (
    NormalizationConfig,
    ReactionRecord,
    reaction_keys,
    split_structural_and_environment,
)


def test_structural_role_split_keeps_fragment_with_product_atom():
    roles = split_structural_and_environment(
        "[CH3:1][Br:3].[Na+].[OH-:2]",
        "[CH3:1][OH:2]",
    )
    assert "[CH3:1][Br:3]" in roles.structural
    assert "[OH-:2]" in roles.structural
    assert "[Na+]" in roles.environment


def test_reaction_keys_ignore_atom_map_labels():
    config = NormalizationConfig()
    first = ReactionRecord("a", "[CH3:1][OH:2]", "[CH3:1][Br:3].[OH-:2]")
    second = ReactionRecord("b", "[CH3:8][OH:9]", "[CH3:8][Br:7].[OH-:9]")
    keys_a = reaction_keys(first, config)
    keys_b = reaction_keys(second, config)
    assert keys_a.exact_structural == keys_b.exact_structural
    assert keys_a.product == keys_b.product
