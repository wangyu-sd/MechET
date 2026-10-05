import pytest

from scripts.audit_system_one_import_space import normalize_batch


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
