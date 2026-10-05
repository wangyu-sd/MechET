from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import audit_mech_uspto31k_endpoint_product as mod


UREA = "O=C(NC1CCCCC1)NC1CCCCC1"
DESIRED = "CCOc1ccc(C(=O)NCCc2ccccc2)cc1"


def test_audit_detects_byproduct_selected_from_min_field(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(mod.EXPECTED, "test", 1)
    raw = tmp_path / "test.parquet"
    endpoint = tmp_path / "test.jsonl"
    pd.DataFrame([{
        "rxn_idx": 1, "step_idx_forward": 0,
        "rxn_prod_min": UREA,
        "rxn_prod_equ": f"{DESIRED}.{UREA}",
    }]).to_parquet(raw)
    endpoint.write_text(json.dumps({"source_id": "1", "product_unmapped": UREA}) + "\n")
    result = mod.audit_split(raw, endpoint, split="test")
    assert result["counts"]["main_product_changed"] == 1
    assert result["counts"]["changed_min_still_in_equ"] == 1
    assert result["changed_reaction_ids"] == ["1"]


def test_audit_rejects_noninvariant_full_final_mixture(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(mod.EXPECTED, "test", 1)
    raw = tmp_path / "test.parquet"
    endpoint = tmp_path / "test.jsonl"
    pd.DataFrame([
        {"rxn_idx": 1, "step_idx_forward": 0, "rxn_prod_min": UREA,
         "rxn_prod_equ": f"{DESIRED}.{UREA}"},
        {"rxn_idx": 1, "step_idx_forward": 1, "rxn_prod_min": UREA,
         "rxn_prod_equ": UREA},
    ]).to_parquet(raw)
    endpoint.write_text(json.dumps({"source_id": "1", "product_unmapped": UREA}) + "\n")
    with pytest.raises(ValueError, match="non-invariant"):
        mod.audit_split(raw, endpoint, split="test")
