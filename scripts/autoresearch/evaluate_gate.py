"""Apply the PR #69 smoke-to-full gates without automatic scale-up."""

from __future__ import annotations

from typing import Any


def promotion_evidence(scorecard: dict[str, Any], *, max_degradation_pp: float = 3.0) -> dict[str, Any]:
    packages = scorecard["packages"]
    complete = all(packages[f"r{i}"]["status"] == "complete" for i in range(1, 6))
    clean = all(packages[f"r{i}"].get("data_contract_errors", 0) == 0
                for i in range(1, 6) if packages[f"r{i}"]["status"] == "complete")
    training = scorecard.get("training") or {}
    base = training.get("base_endpoint_top1")
    mech = training.get("mech_endpoint_top1")
    endpoint_delta = None if base is None or mech is None else 100 * (float(mech) - float(base))
    endpoint_gate = None if endpoint_delta is None else endpoint_delta >= -max_degradation_pp
    identifiable = bool(training.get("mech_comparison_identifiable"))
    overlap_clean = scorecard.get("train_eval_overlap_count") == 0
    checks = {
        "all_r1_to_r5_complete": complete,
        "no_data_contract_errors": clean,
        "mech_endpoint_delta_pp": endpoint_delta,
        "endpoint_degradation_within_limit": endpoint_gate,
        "mechanism_augmentation_identifiable": identifiable,
        "train_eval_overlap_zero": overlap_clean,
        "electron_external_signal": "human_review_required",
        "two_of_r1_r3_r5_useful": "human_review_required",
    }
    hard_failure = (not clean or scorecard.get("train_eval_overlap_count", 0) not in (None, 0)
                    or endpoint_gate is False or not identifiable)
    if hard_failure:
        recommendation = "RECOMMEND_REVISE_DATA_OR_PROTOCOL"
    elif not complete or endpoint_gate is None or not overlap_clean:
        recommendation = "INCOMPLETE"
    else:
        recommendation = "HUMAN_REVIEW_FOR_SCALE"
    return {"checks": checks, "recommendation": recommendation,
            "automatic_full_scale_launch_allowed": False}
