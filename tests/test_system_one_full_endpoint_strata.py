from __future__ import annotations

import pytest

from scripts.stratify_system_one_full_endpoint import (
    electron_state_revisit_count, summarize_cases, verified_changed_ids,
)


def _case(reaction_id: str, *, product: str, predicted: str, exact: bool,
          electron_events: int = 1, imports: int = 0) -> dict:
    return {
        "id": reaction_id,
        "completed": True,
        "terminal": "FINISHED",
        "structural_exact": exact,
        "principal_product_input": product,
        "predicted_structural_precursor": predicted,
        "electron_events": electron_events,
        "import_batches": imports,
    }


def test_compiler_coverage_and_no_transform_are_separate() -> None:
    cases = [
        _case("1", product="CO", predicted="C.O", exact=True),
        _case("2", product="CCO", predicted="CCO", exact=False, imports=1),
        _case("3", product="CCC", predicted="CC.C", exact=False),
    ]
    contexts = {str(i): {"top1_exact": i != 3} for i in range(1, 4)}
    groups = summarize_cases(cases, contexts, {"1"}, {"1", "2"}, {"2"})
    assert groups["stitched_strict_trace"]["structural_exact"] == 1
    assert groups["all_steps_executable_but_unstitched"]["finished_wrong_no_transform"] == 1
    assert groups["all_steps_executable_but_unstitched"]["context_exact_but_endpoint_wrong"] == 1
    assert groups["all_steps_executable_but_unstitched"]["min_equ_target_changed"] == 1
    assert groups["stitched_strict_trace"]["min_equ_target_changed"] == 0
    assert groups["incomplete_elementary_steps"]["reactions"] == 1
    assert groups["all"]["reactions"] == 3


def test_stitched_ids_must_be_all_step_executable() -> None:
    with pytest.raises(ValueError, match="subset"):
        summarize_cases([], {}, {"1"}, set())


def _audit_and_handoff() -> tuple[dict, dict]:
    audit = {
        "artifact_type": "mech_uspto31k_existing_min_target_vs_equ_final_mixture_audit",
        "endpoint_manifest_sha256": "old-manifest",
        "splits": {"test": {
            "endpoint_sha256": "old-endpoint", "raw_sha256": "raw",
            "changed_reaction_ids": ["1", "2"],
            "counts": {"main_product_changed": 2},
        }},
    }
    handoff = {
        "artifact_type": "mech_uspto31k_equ_proxy_endpoint_handoff_verification",
        "old_manifest_sha256": "old-manifest",
        "new_manifest_sha256": "new-manifest",
        "raw_field_audit_sha256": "audit-sha",
        "splits": {"test": {
            "old_endpoint_sha256": "old-endpoint",
            "new_endpoint_sha256": "new-endpoint",
            "counts": {"product_changed": 2, "reactions": 3120},
        }},
    }
    return audit, handoff


def test_equ_stratum_requires_verified_handoff() -> None:
    audit, handoff = _audit_and_handoff()
    args = dict(split="test", product_field="rxn_prod_equ",
                full_manifest_sha="new-manifest", full_source_sha="new-endpoint",
                raw_sha="raw", product_audit_sha="audit-sha")
    assert verified_changed_ids(audit, handoff, **args) == {"1", "2"}
    with pytest.raises(ValueError, match="verified handoff"):
        verified_changed_ids(audit, None, **args)
    handoff["splits"]["test"]["new_endpoint_sha256"] = "wrong-endpoint"
    with pytest.raises(ValueError, match="lineage mismatch"):
        verified_changed_ids(audit, handoff, **args)


def test_min_stratum_rejects_equ_handoff() -> None:
    audit, handoff = _audit_and_handoff()
    args = dict(split="test", product_field="rxn_prod_min",
                full_manifest_sha="old-manifest", full_source_sha="old-endpoint",
                raw_sha="raw", product_audit_sha="audit-sha")
    assert verified_changed_ids(audit, None, **args) == {"1", "2"}
    with pytest.raises(ValueError, match="cannot be used"):
        verified_changed_ids(audit, handoff, **args)


def test_electron_successor_revisit_is_counted() -> None:
    case = _case("1", product="CO", predicted="C.O", exact=False)
    case["actions"] = [
        {"action": "apply_electron_flow", "accepted": True,
         "state_before": "CO", "state_after": "C.O"},
        {"action": "apply_electron_flow", "accepted": True,
         "state_before": "C.O", "state_after": "CO"},
        {"action": "finish_trace", "accepted": True, "state_before": "CO"},
    ]
    assert electron_state_revisit_count(case) == 1
    groups = summarize_cases([case], {"1": {"top1_exact": True}}, {"1"}, {"1"})
    assert groups["all"]["electron_state_revisit_reactions"] == 1
    assert groups["all"]["electron_state_revisit_steps"] == 1
