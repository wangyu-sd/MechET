import pytest

from collections import Counter

from scripts.audit_system_one_import_space import normalize_batch, novelty_against_train


def test_import_batch_normalizes_order_but_not_chemistry():
    batch = normalize_batch({"fragments": [
        {"smiles": "[Na+]", "count": 1, "purpose": "endpoint_context"},
        {"smiles": "[H][H]", "count": 1, "purpose": "electron_participant"},
    ]})
    assert batch == (("[H][H]", 1, "electron_participant"),
                     ("[Na+]", 1, "endpoint_context"))


@pytest.mark.parametrize("fragments", [[], [{"smiles": "O", "count": 0,
                                            "purpose": "electron_participant"}],
                                      [{"smiles": "O", "count": True,
                                        "purpose": "electron_participant"}]])
def test_import_batch_rejects_missing_or_nonpositive_counts(fragments):
    with pytest.raises(ValueError):
        normalize_batch({"fragments": fragments})


def test_import_novelty_counts_distinct_batches_and_decisions_separately():
    result = novelty_against_train(Counter({"seen": 10}),
                                   Counter({"seen": 20, "new_a": 3, "new_b": 1}))
    assert result["unseen_batches_vs_train"] == 2
    assert result["unseen_decisions_vs_train"] == 4
    assert result["unseen_decision_rate_vs_train"] == 4 / 24
