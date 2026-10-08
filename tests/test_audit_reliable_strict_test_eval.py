import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_reliable_strict_test_eval import audit, validate_source


def _fixture(tmp_path: Path):
    source, manifest = tmp_path / "strict.jsonl", tmp_path / "manifest.json"
    rows = [
        {"id": f"textbook-tool-sft:flower_mech_proof_test_{i}",
         "source_id": f"flower_mech_proof_test_{i}",
         "target_smiles": "[CH4:9]", "structural_precursor": "CO"}
        for i in (1, 2, 3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest.write_text(json.dumps({"splits": {"test": {
        "rows": 3, "unique_ids": 3,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }}}), encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    predictions = [
        {"id": rows[0]["id"], "source_id": rows[0]["source_id"],
         "endpoint_metric": "structural", "top1_exact": True,
         "top1_structural_exact": True, "top1_full_exact": False,
         "top_terminal": True, "n_actions": 1,
         "top_actions": [{"name": "finish_trace"}],
         "attempts": [{"accepted": True}], "rejected": {}},
        {"id": rows[1]["id"], "source_id": rows[1]["source_id"],
         "endpoint_metric": "structural", "top1_exact": False,
         "top1_structural_exact": False, "top1_full_exact": False,
         "top_terminal": False, "n_actions": 0,
         "top_actions": [],
         "attempts": [{"accepted": False}], "rejected": {"BAD_ALIAS": 1}},
    ]
    shard = results / "results.shard-00-of-01.jsonl"
    shard.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    return source, manifest, results, shard, predictions


def test_strict_audit_keeps_missing_in_reaction_denominator(tmp_path: Path):
    source, manifest, results, _, _ = _fixture(tmp_path)
    ids, _ = validate_source(
        source=source, manifest=manifest, expected_rows=3,
        check_product_only_mapping=True,
    )
    assert len(ids) == 3
    report = audit(source=source, manifest=manifest, results_dir=results, expected_rows=3)
    assert report["artifact_type"].endswith("_unbound_diagnostic_v1")
    assert report["observed_predictions"] == 2
    assert report["missing_predictions"] == 1
    assert report["structural_accuracy"] == pytest.approx(1 / 3)
    assert report["terminal"] == 1
    assert report["accepted_decisions_observed"] == 1
    assert report["rejected_decisions_observed"] == 1
    assert report["rejected_causes_observed"] == {"BAD_ALIAS": 1}


def test_strict_audit_rejects_inconsistent_execution_counters(tmp_path: Path):
    source, manifest, results, shard, predictions = _fixture(tmp_path)
    predictions[1]["rejected"] = {}
    shard.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    with pytest.raises(ValueError, match="rejected-action counts disagree"):
        audit(source=source, manifest=manifest, results_dir=results, expected_rows=3)


def test_strict_audit_refuses_nonterminal_exactness(tmp_path: Path):
    source, manifest, results, shard, predictions = _fixture(tmp_path)
    predictions[0]["top_terminal"] = False
    shard.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    with pytest.raises(ValueError, match="nonterminal exact prediction"):
        audit(source=source, manifest=manifest, results_dir=results, expected_rows=3)


def test_strict_launcher_keeps_own_input_and_denominator():
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" /
        "run_taiji_reliable_mechet_strict_test_a100.sh"
    ).read_text(encoding="utf-8")
    assert "flower_inverse_tool_sft_action_delta_v1/test.jsonl" in launcher
    assert "--sample-reactions 28967" in launcher
    assert "--matched-v2 --product-only-remap" in launcher
    assert "--branching 1 --early-beam 1 --late-beam 1" in launcher
    assert "audit_reliable_strict_test_eval.py" in launcher
    assert "--expected-adapter-sha256 \"$adapter_sha\" --stage \"$stage\"" in launcher


def test_strict_audit_records_trajectory_adapter_identity(tmp_path: Path):
    source, manifest, results, _, _ = _fixture(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"trajectory-weights")
    (adapter / "adapter_manifest.json").write_text(json.dumps({
        "base_model": "Qwen/Qwen3-0.6B",
        "base_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "environment_revision": "natural_language_electron_event_history_v2",
        "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
    }))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    report = audit(
        source=source, manifest=manifest, results_dir=results, expected_rows=3,
        adapter=adapter, expected_adapter_sha256=digest, stage="trajectory",
    )
    assert report["artifact_type"] == "reliable_mechet_strict_process_test_audit_v1"
    assert report["adapter_identity"]["stage"] == "trajectory"
    assert report["adapter_identity"]["adapter_model_sha256"] == digest
    with pytest.raises(ValueError, match="environment_revision"):
        audit(
            source=source, manifest=manifest, results_dir=results, expected_rows=3,
            adapter=adapter, expected_adapter_sha256=digest, stage="state",
        )
