import json

from scripts.audit_system_one_full_endpoint_input_gap import sha256
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
