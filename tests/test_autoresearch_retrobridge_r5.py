"""Pure contract checks for the pinned official RetroBridge R5 intake."""

from __future__ import annotations

from pathlib import Path

from scripts.autoresearch.run_retrobridge_r5 import official_train_overlap, rank_samples


def test_frequency_rank_is_gold_independent_and_stably_tied() -> None:
    ranked = rank_samples(["C.O", "N", "C.O", "O", "N", "Cl"], top_k=3)
    assert [row["precursors"] for row in ranked] == ["C.O", "N", "O"]
    assert [row["sample_count"] for row in ranked] == [2, 2, 1]
    assert ranked[0]["frequency_confidence"] == 2 / 6
    assert all("reference" not in row for row in ranked)


def test_train_overlap_uses_chemical_key_not_frozen_spelling(tmp_path: Path) -> None:
    train = tmp_path / "uspto50k_train.csv"
    train.write_text("id,class,reactants>reagents>production\n"
                     "r1,1,C.O>>CCO\n", encoding="utf-8")
    assert official_train_overlap(train, ["C(C)O", "CCN"]) == {
        "C(C)O": True, "CCN": False,
    }
