"""R5 products are frozen before external model predictions exist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.build_r5_products import balanced_products, build
from scripts.autoresearch.stratified_manifest import digest, product_key


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_balanced_products_requires_size_quartile_capacity() -> None:
    with pytest.raises(ValueError, match="four quartiles"):
        balanced_products(["C", "CC"], count=3, seed=17, cohort="toy")
    with pytest.raises(ValueError, match="only 2"):
        balanced_products(["C", "CC"], count=4, seed=17, cohort="toy")


def test_r5_query_cohort_has_private_references_but_product_only_input(tmp_path: Path) -> None:
    multi_products = ["CCO", "CCN", "CCC", "CCOC"]
    other_products = ["CO", "CN", "CC", "CCCl"]
    source_rows = []
    for index, product in enumerate(multi_products):
        for suffix, precursor in enumerate(("C", "O")):
            source_rows.append({"id": f"m{index}:{suffix}", "source_id": f"m{index}:{suffix}",
                                "target_smiles": product, "structural_precursor": precursor})
    for index, product in enumerate(other_products):
        source_rows.append({"id": f"s{index}", "source_id": f"s{index}",
                            "target_smiles": product, "structural_precursor": "C"})
    source = tmp_path / "official_test.jsonl"
    _jsonl(source, source_rows)
    official_manifest = tmp_path / "official_manifest.json"
    official_manifest.write_text(json.dumps({
        "splits": {"test": {"output_sha256": digest(source)}}}))
    r1_cohort = tmp_path / "r1_multi_reference.jsonl"
    _jsonl(r1_cohort, [{"product_smiles": product_key(product),
                        "reference_count": 2} for product in multi_products])
    r1_manifest = tmp_path / "r1_manifest.json"
    r1_manifest.write_text(json.dumps({"cohort_sha256": digest(r1_cohort)}))
    output = tmp_path / "r5"
    report = build(source, official_manifest, r1_cohort, r1_manifest,
                   output, per_cohort=4)
    rows = [json.loads(line) for line in (output / "r5_products.jsonl").open()]
    assert report["products"] == 8
    assert report["multi_recorded"] == report["other_official_test"] == 4
    assert report["quartiles"] == {f"Q{index}": 2 for index in range(1, 5)}
    assert all(row["model_input"] == {"product_smiles": row["product_smiles"]}
               for row in rows)
    assert all("precursor" not in row["model_input"] for row in rows)
    assert sum(row["private_reference"]["reference_count"] == 2 for row in rows) == 4
    with pytest.raises(FileExistsError):
        build(source, official_manifest, r1_cohort, r1_manifest, output,
              per_cohort=4)
