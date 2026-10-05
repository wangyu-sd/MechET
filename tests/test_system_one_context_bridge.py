import json
import pytest

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.compare_system_one_context_bridge import compare
from scripts.eval_system_one_context_bridge import inferred_mixture, load_policy_contexts


def test_context_bridge_projects_away_heldout_reference_labels(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({
        "reaction_id": "7", "product_unmapped": "CCO",
        "predicted_context_batch": ["[Br-]"],
        "reference_context_batch": ["[Cl-]"], "top1_exact": False,
    }) + "\n")
    (tmp_path / "report.json").write_text(json.dumps({
        "artifact_type": "system_one_principal_product_context_retrieval_diagnostic",
        "split": "valid",
        "method": "train_only_morgan_radius2_2048_nearest_reaction_distinct_context_batches",
        "heldout_source": {"strict_source_sha256": "heldout"},
        "train_source": {"strict_source_sha256": "train"},
        "cases_sha256": sha256(cases), "evaluated": 1,
    }))
    projected, _ = load_policy_contexts(
        tmp_path, split="valid", strict_source_sha256="heldout",
        train_source_sha256="train",
    )
    assert projected == {"7": ("CCO", ("[Br-]",))}
    assert "[Cl-]" not in str(projected)


def test_context_bridge_adds_counted_components_without_changing_product():
    assert inferred_mixture("CCO", ()) == "CCO"
    assert inferred_mixture("CCO", ("[Cl-]", "[Cl-]")) == "CCO.[Cl-].[Cl-]"


def test_context_bridge_pairing_requires_identical_rollout_when_input_identical(tmp_path):
    baseline_dir, bridge_dir = tmp_path / "baseline", tmp_path / "bridge"
    baseline_dir.mkdir()
    bridge_dir.mkdir()
    baseline = {
        "id": "7", "target": "CCO.[Cl-]", "expected_precursor": "CCBr.[Cl-]",
        "predicted_precursor": "CCBr.[Cl-]", "terminal": "FINISHED",
        "endpoint_exact": True, "actions": [{"action": "apply_electron_flow"}],
    }
    bridged = {
        "id": "7", "strict_reference_final_mixture": baseline["target"],
        "inferred_final_mixture": baseline["target"],
        "strict_mixture_reconstructed_byte_exact": True,
        "expected_strict_full_precursor": baseline["expected_precursor"],
        "predicted_precursor": baseline["predicted_precursor"],
        "terminal": baseline["terminal"], "actions": baseline["actions"],
        "endpoint_exact_strict_full_precursor": True,
    }
    (baseline_dir / "cases.jsonl").write_text(json.dumps(baseline) + "\n")
    (baseline_dir / "report.json").write_text(json.dumps({
        "source": {"sha256": "same"}, "weights": {"adapter": "same"},
        "legality_backoff": True, "evaluated_reactions": 1,
    }))
    bridge_cases = bridge_dir / "cases.jsonl"
    bridge_cases.write_text(json.dumps(bridged) + "\n")
    bridge_report = {
        "artifact_type": "system_one_pr81_principal_product_context_bridge_diagnostic",
        "strict_source": {"sha256": "same"}, "weights": {"adapter": "same"},
        "evaluated_reactions": 1, "endpoint_exact_strict_full_precursor": 1,
        "strict_mixture_reconstructed_byte_exact": 1,
        "scope": "strict_trace_view", "split": "valid", "cases_sha256": sha256(bridge_cases),
    }
    (bridge_dir / "report.json").write_text(json.dumps(bridge_report))
    assert compare(bridge_dir, baseline_dir)["counts"]["reconstructed_bridge_hit"] == 1
    bridged["actions"] = [{"action": "import_fragments"}]
    bridge_cases.write_text(json.dumps(bridged) + "\n")
    bridge_report["cases_sha256"] = sha256(bridge_cases)
    (bridge_dir / "report.json").write_text(json.dumps(bridge_report))
    with pytest.raises(ValueError, match="same input produced different rollout"):
        compare(bridge_dir, baseline_dir)
