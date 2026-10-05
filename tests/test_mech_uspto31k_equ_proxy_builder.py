from __future__ import annotations

import json
import pandas as pd
import pytest

from scripts import build_mech_uspto31k_rxnmapper_baseline as builder

DESIRED = "CCOc1ccc(C(=O)NCCc2ccccc2)cc1"


def test_equ_product_field_recovers_coupling_product_without_changing_min(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.EXPECTED, "test", 1)
    pd.DataFrame([{
        "rxn_idx": 1,
        "step_idx_forward": 0,
        "elem_reac_spe": "CC(=O)O.NC.C(=NC1CCCCC1)=NC1CCCCC1",
        "rxn_prod_min": "O=C(NC1CCCCC1)NC1CCCCC1",
        "rxn_prod_equ": f"{DESIRED}.O=C(NC1CCCCC1)NC1CCCCC1",
    }]).to_parquet(tmp_path / "test-00000-of-00001.parquet")
    old = builder.load_reaction_rows(tmp_path, ["test"])
    new = builder.load_reaction_rows(tmp_path, ["test"], product_field="rxn_prod_equ")
    assert old[0]["product_unmapped"] == "O=C(NC1CCCCC1)NC1CCCCC1"
    assert new[0]["product_unmapped"] == DESIRED
    assert old[0]["precursor_unmapped"] == new[0]["precursor_unmapped"]
    assert new[0]["product_field"] == "rxn_prod_equ"


def test_product_field_is_explicitly_restricted(tmp_path):
    with pytest.raises(ValueError, match="unsupported product field"):
        builder.load_reaction_rows(tmp_path, ["test"], product_field="unknown")


def test_mapping_cache_reuse_only_for_identical_reaction_pairs(tmp_path):
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    old.write_text("".join(json.dumps(row) + "\n" for row in [
        {"stable_id": "same", "reaction_unmapped": "C>>CO",
         "reaction_mapped": "[CH4:1]>>[CH3:1][OH:2]"},
        {"stable_id": "changed", "reaction_unmapped": "N>>NO",
         "reaction_mapped": "[NH3:1]>>[NH2:1][OH:2]"},
    ]))
    rows = [
        {"stable_id": "same", "reaction_unmapped": "C>>CO"},
        {"stable_id": "changed", "reaction_unmapped": "N>>NC"},
    ]
    reused = builder.seed_identical_mapping_pairs(
        rows, cache_path=new, prior_cache_path=old,
        prior_sha256=builder.sha256_file(old),
    )
    assert reused == 1
    assert set(builder.read_mapping_cache(new)) == {"same"}
    with pytest.raises(ValueError, match="SHA-256"):
        builder.seed_identical_mapping_pairs(
            rows, cache_path=tmp_path / "bad.jsonl", prior_cache_path=old,
            prior_sha256="wrong",
        )
