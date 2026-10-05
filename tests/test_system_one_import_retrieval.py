from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from scripts.eval_system_one_import_retrieval import (
    ImportExample,
    append_import_batch,
    chemical_batch,
    fingerprint,
    rank_unique_batches,
)


def _example(name, target, current, batch):
    return ImportExample(name, name, target, current, batch)


def test_chemical_batch_ignores_purpose_but_preserves_counts():
    one = chemical_batch({"fragments": [
        {"smiles": "O", "count": 1, "purpose": "electron_participant"},
        {"smiles": "O", "count": 2, "purpose": "endpoint_context"},
    ]})
    two = chemical_batch({"fragments": [
        {"smiles": "O", "count": 3, "purpose": "any_valid_annotation"},
    ]})
    assert one == two == (("O", 3),)


def test_retrieval_ranks_by_product_and_current_state_without_gold():
    train = [
        _example("a", "CCO", "CC", (("O", 1),)),
        _example("b", "c1ccccc1", "c1ccccc1", (("N", 1),)),
        _example("c", "CCO", "CC", (("O", 1),)),
    ]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target_fps = [fingerprint(row.target, generator) for row in train]
    current_fps = [fingerprint(row.current, generator) for row in train]
    ranked = rank_unique_batches(
        fingerprint("CCO", generator), fingerprint("CC", generator),
        train, target_fps, current_fps, limit=8,
    )
    assert ranked == [(('O', 1),), (('N', 1),)]
    assert Chem.MolFromSmiles(ranked[0][0][0]) is not None


def test_append_import_batch_preserves_old_chemistry_and_copy_count():
    assert append_import_batch("CCO", (("[Na+]", 2),)) == "CCO.[Na+].[Na+]"
