import json

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.eval_system_one_full_context_knn import METHOD
from scripts.eval_system_one_full_endpoint import load_policy_contexts, select_ids


def test_full_endpoint_policy_projection_discards_heldout_context(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({
        "reaction_id": "4", "product_unmapped": "CCO",
        "predicted_context_batch": ["[Br-]"],
        "reference_context_batch": ["[Cl-]"], "top1_exact": False,
    }) + "\n")
    (tmp_path / "report.json").write_text(json.dumps({
        "artifact_type": "system_one_full_endpoint_principal_product_context_proposal",
        "method": METHOD, "split": "valid",
        "train_source": {"sha256": "train"},
        "heldout_source": {"sha256": "heldout"},
        "cases_sha256": sha256(path), "evaluated": 1,
    }))
    projected, _ = load_policy_contexts(
        tmp_path, split="valid", train_sha="train", heldout_sha="heldout"
    )
    assert projected == {"4": ("CCO", ("[Br-]",))}
    assert "[Cl-]" not in str(projected)


def test_full_endpoint_sampling_is_deterministic_and_full_limit_keeps_all():
    ids = ["10", "2", "3", "1"]
    assert select_ids(ids, seed=17, limit=0) == select_ids(ids[::-1], seed=17, limit=0)
    assert len(select_ids(ids, seed=17, limit=0)) == 4
    assert select_ids(ids, seed=17, limit=2) == select_ids(ids, seed=17, limit=0)[:2]
