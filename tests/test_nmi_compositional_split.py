from mechet.proof_splits import ProofSplitFeatures
from scripts.build_nmi_compositional_split import select_split


def test_split_holds_out_unseen_program_with_seen_primitives():
    ids = [f"r{index}" for index in range(8)]
    features = (
        [ProofSplitFeatures("a", ("p1",)) for _ in range(3)]
        + [ProofSplitFeatures("b", ("p2",)) for _ in range(3)]
        + [ProofSplitFeatures("ab", ("p1", "p2")) for _ in range(2)]
    )
    reactions = [f"reaction{index}" for index in range(8)]
    splits, report = select_split(
        ids, features, reactions, seed=7, test_fraction=0.25,
        valid_fraction=0, min_train_primitive_count=2,
    )
    assert splits["test"] == {6, 7}
    assert report["rows"] == {"train": 6, "valid": 0, "test": 2}
    assert all(report["gates"].values())


def test_shared_exact_reaction_forces_compositions_into_same_component():
    ids = ["a1", "a2", "b1", "b2", "ab1", "ab2"]
    features = [
        ProofSplitFeatures("a", ("p1",)), ProofSplitFeatures("a", ("p1",)),
        ProofSplitFeatures("b", ("p2",)), ProofSplitFeatures("b", ("p2",)),
        ProofSplitFeatures("ab", ("p1", "p2")), ProofSplitFeatures("ab", ("p1", "p2")),
    ]
    reactions = ["shared", "r2", "shared", "r4", "r5", "r6"]
    splits, report = select_split(
        ids, features, reactions, seed=7, test_fraction=0.2,
        valid_fraction=0, min_train_primitive_count=1,
    )
    assert report["n_union_components"] == 2
    assert report["gates"]["zero_train_test_exact_reaction_overlap"]
    assert not (0 in splits["train"] and 2 in splits["test"])
