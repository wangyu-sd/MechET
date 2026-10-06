import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_reliable_full_endpoint_eval import audit, validate_source


def _artifact(tmp_path: Path):
    source = tmp_path / "test.jsonl"
    rows = [
        {"id": f"flower-full-endpoint:test:{index}", "source_id": str(index),
         "metadata": {"coverage_track": "full_endpoint"},
         "structural_precursor": "CO"}
        for index in (1, 2, 3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"test": {
        "rows": 3, "expected_rows": 3,
        "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }}}), encoding="utf-8")
    output = tmp_path / "results"
    output.mkdir()
    predictions = [
        {"id": rows[0]["id"], "source_id": "1", "endpoint_metric": "structural",
         "top1_exact": True, "top_terminal": True, "top1_full_exact": False},
        {"id": rows[1]["id"], "source_id": "2", "endpoint_metric": "structural",
         "top1_exact": False, "top_terminal": True, "top1_full_exact": False},
    ]
    shard = output / "results.shard-00-of-01.jsonl"
    shard.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    return source, manifest, output, shard, predictions


def test_full_endpoint_missing_prediction_counts_as_failure(tmp_path: Path):
    source, manifest, output, _, _ = _artifact(tmp_path)
    identifiers, digest = validate_source(source=source, manifest=manifest, expected_rows=3)
    assert len(identifiers) == 3
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    report = audit(source=source, manifest=manifest, results_dir=output, expected_rows=3)
    assert report["artifact_type"].endswith("_unbound_diagnostic_v1")
    assert report["test_denominator"] == 3
    assert report["observed_predictions"] == 2
    assert report["missing_predictions"] == 1
    assert report["top1_structural_exact"] == 1
    assert report["top1_structural_accuracy"] == pytest.approx(1 / 3)


def test_full_endpoint_preflight_executes_product_only_private_mapping(tmp_path: Path):
    source, manifest, _, _, _ = _artifact(tmp_path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["target_smiles"] = "[CH4:987]"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["splits"]["test"]["output_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    ids, _ = validate_source(
        source=source, manifest=manifest, expected_rows=3,
        check_product_only_mapping=True,
    )
    assert len(ids) == 3
    rows[1]["target_smiles"] = "not-a-smiles"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata["splits"]["test"]["output_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="product-only private mapping failed"):
        validate_source(
            source=source, manifest=manifest, expected_rows=3,
            check_product_only_mapping=True,
        )


def test_full_endpoint_launcher_uses_frozen_product_only_denominator():
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" /
        "run_taiji_reliable_mechet_full_endpoint_a100.sh"
    ).read_text(encoding="utf-8")
    assert "flower_full_endpoint_sft/test.jsonl" in launcher
    assert "--sample-reactions 28971" in launcher
    assert "--matched-v2 --product-only-remap" in launcher
    assert "--branching 1 --early-beam 1 --late-beam 1" in launcher
    assert "audit_reliable_full_endpoint_eval.py" in launcher
    assert "--expected-adapter-sha256 \"$adapter_sha\" --stage \"$stage\"" in launcher


def test_full_endpoint_audit_binds_the_actual_adapter_weights(tmp_path: Path):
    source, manifest, output, _, _ = _artifact(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"frozen-weights")
    (adapter / "adapter_manifest.json").write_text(json.dumps({
        "base_model": "Qwen/Qwen3-0.6B",
        "base_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "environment_revision": "natural_language_electron_event_v2",
        "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
    }))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    report = audit(
        source=source, manifest=manifest, results_dir=output, expected_rows=3,
        adapter=adapter, expected_adapter_sha256=digest, stage="state",
    )
    assert report["artifact_type"] == "reliable_mechet_full_endpoint_test_audit_v1"
    assert report["adapter_identity"]["adapter_model_sha256"] == digest
    weights.write_bytes(b"changed-weights")
    with pytest.raises(ValueError, match="weights changed"):
        audit(
            source=source, manifest=manifest, results_dir=output, expected_rows=3,
            adapter=adapter, expected_adapter_sha256=digest, stage="state",
        )
    with pytest.raises(ValueError, match="requires adapter"):
        audit(
            source=source, manifest=manifest, results_dir=output, expected_rows=3,
            expected_adapter_sha256=digest,
        )


def test_full_endpoint_rejects_wrong_source_and_duplicate_predictions(tmp_path: Path):
    source, manifest, output, shard, predictions = _artifact(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["splits"]["test"]["output_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        audit(source=source, manifest=manifest, results_dir=output, expected_rows=3)
    data["splits"]["test"]["output_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    shard.write_text("".join(json.dumps(row) + "\n" for row in [*predictions, predictions[0]]))
    with pytest.raises(ValueError, match="duplicate prediction"):
        audit(source=source, manifest=manifest, results_dir=output, expected_rows=3)


def test_full_endpoint_rejects_wrong_metric_and_source_id(tmp_path: Path):
    source, manifest, output, shard, predictions = _artifact(tmp_path)
    predictions[0]["endpoint_metric"] = "full_unmapped"
    shard.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    with pytest.raises(ValueError, match="wrong endpoint metric"):
        audit(source=source, manifest=manifest, results_dir=output, expected_rows=3)
    predictions[0]["endpoint_metric"] = "structural"
    predictions[0]["source_id"] = "wrong"
    shard.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    with pytest.raises(ValueError, match="source ID mismatch"):
        audit(source=source, manifest=manifest, results_dir=output, expected_rows=3)
