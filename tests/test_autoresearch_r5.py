"""R5 products are frozen before external model predictions exist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.build_r5_products import balanced_products, build
from scripts.autoresearch.freeze_r5_external_predictions import freeze
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


def test_r5_external_intake_preserves_missing_ranks_and_nonreference_uncertainty(tmp_path: Path) -> None:
    query = tmp_path / "r5_products.jsonl"
    _jsonl(query, [{
        "product_smiles": product, "source_dataset": "FlowER official full endpoint",
        "source_split": "test", "model_input": {"product_smiles": product},
        "strata": {"heavy_atom_quartile": "Q1"},
        "private_reference": {"recorded_precursor_sets": [
            {"precursor_smiles": "C", "support_record_ids": ["r1"]}]},
    } for product in ("CC", "CO")])
    query_manifest = tmp_path / "query_manifest.json"
    query_manifest.write_text(json.dumps({"cohort_sha256": digest(query), "products": 2}))
    raw = tmp_path / "external.jsonl"
    _jsonl(raw, [
        {"product_smiles": "CC", "inference_status": "completed",
         "candidates": [{"rank": 1, "precursors": "O"},
                        {"rank": 2, "precursors": "C"},
                        {"rank": 3, "precursors": "not-smiles"}]},
        {"product_smiles": "CO", "inference_status": "failed", "candidates": []},
    ])
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "model_name": "external-toy", "checkpoint_identifier": "toy-revision",
        "checkpoint_source": "https://example.org/toy", "training_corpus": "toy",
        "license_or_terms": "toy", "input_fields": ["product_smiles"],
        "target_semantics": "retrosynthetic_precursor_set",
        "inference_status": "completed", "inference_config": {"beam": 5},
        "training_overlap_audited": True,
    }))
    output = tmp_path / "frozen"
    report = freeze(query, query_manifest, raw, provenance, output,
                    expected_products=2, top_k=5)
    rows = [json.loads(line) for line in (output / "r5_external_predictions.jsonl").open()]
    assert report["products"] == 2
    assert report["counts"]["missing"] == 7
    assert report["counts"]["invalid_smiles"] == 1
    assert rows[0]["candidates"][0]["recorded_reference_status"] == "not_recorded_not_proven_invalid"
    assert rows[0]["candidates"][1]["recorded_reference_status"] == "recorded_reference"
    assert all(len(row["candidates"]) == 5 for row in rows)
    assert all(row["model_input"] == {"product_smiles": row["product_smiles"]}
               for row in rows)
    status = json.loads((output / "ARTIFACT_STATUS.json").read_text())
    assert status["evaluation_allowed"] is True
    assert status["training_allowed"] is False
    assert status["headline_allowed"] is False


def test_r5_external_intake_rejects_missing_product(tmp_path: Path) -> None:
    query = tmp_path / "r5_products.jsonl"
    _jsonl(query, [{"product_smiles": "CC", "model_input": {"product_smiles": "CC"},
                    "source_dataset": "FlowER", "source_split": "test",
                    "strata": {}, "private_reference": {"recorded_precursor_sets": []}}])
    query_manifest = tmp_path / "manifest.json"
    query_manifest.write_text(json.dumps({"cohort_sha256": digest(query), "products": 1}))
    raw = tmp_path / "external.jsonl"
    _jsonl(raw, [])
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "model_name": "toy", "checkpoint_identifier": "toy", "checkpoint_source": "toy",
        "training_corpus": "toy", "license_or_terms": "toy",
        "input_fields": ["product_smiles"], "inference_status": "completed",
        "target_semantics": "retrosynthetic_precursor_set",
        "inference_config": {"beam": 5},
    }))
    with pytest.raises(ValueError, match="every frozen product"):
        freeze(query, query_manifest, raw, provenance, tmp_path / "out",
               expected_products=1)


def test_r5_external_intake_rejects_full_reaction_world_target(tmp_path: Path) -> None:
    query = tmp_path / "query.jsonl"
    _jsonl(query, [{"product_smiles": "CC", "model_input": {"product_smiles": "CC"},
                    "source_dataset": "FlowER", "source_split": "test",
                    "strata": {}, "private_reference": {"recorded_precursor_sets": []}}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cohort_sha256": digest(query), "products": 1}))
    raw = tmp_path / "predictions.jsonl"
    _jsonl(raw, [{"product_smiles": "CC", "inference_status": "completed",
                  "candidates": [{"rank": 1, "precursors": "CC.O.[Na+]"}]}])
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "model_name": "world-generator", "checkpoint_identifier": "toy",
        "checkpoint_source": "toy", "training_corpus": "toy",
        "license_or_terms": "toy", "input_fields": ["product_smiles"],
        "target_semantics": "full_reaction_world",
        "inference_status": "completed", "inference_config": {"beam": 5},
    }))
    with pytest.raises(ValueError, match="precursor-set predictions"):
        freeze(query, manifest, raw, provenance, tmp_path / "frozen",
               expected_products=1)


def test_r5_external_intake_preserves_frozen_noncanonical_spelling(tmp_path: Path) -> None:
    product = "C(C)O"
    assert product_key(product) == "CCO"
    query = tmp_path / "query.jsonl"
    _jsonl(query, [{"product_smiles": product, "model_input": {"product_smiles": product},
                    "source_dataset": "FlowER", "source_split": "test",
                    "strata": {}, "private_reference": {"recorded_precursor_sets": []}}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cohort_sha256": digest(query), "products": 1}))
    predictions = tmp_path / "predictions.jsonl"
    _jsonl(predictions, [{"product_smiles": product, "inference_status": "completed",
                          "candidates": [{"rank": 1, "precursors": "C.O"}]}])
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "model_name": "toy", "checkpoint_identifier": "toy", "checkpoint_source": "toy",
        "training_corpus": "toy", "license_or_terms": "toy",
        "input_fields": ["product_smiles"], "inference_status": "completed",
        "target_semantics": "retrosynthetic_precursor_set",
        "inference_config": {"ranking": "frequency"},
    }))
    output = tmp_path / "frozen"
    freeze(query, manifest, predictions, provenance, output,
           expected_products=1)
    row = json.loads((output / "r5_external_predictions.jsonl").read_text().splitlines()[0])
    assert row["product_smiles"] == product
