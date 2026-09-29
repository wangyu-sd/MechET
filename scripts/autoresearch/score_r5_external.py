#!/usr/bin/env python3
"""Score candidate-conditioned R5 traces without equating no proof with invalidity.

Each condition supplies one verification row per frozen product/rank slot. A
supported candidate needs an executor-owned terminal trace whose structural
precursor equals that external candidate. Missing, invalid and failed slots
remain in the 200 x 5 denominator. This is sampled-trace support, not a proof
that unsupported candidates are chemically impossible.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key, verify_evaluation_source


def _read_verifications(path: Path, *, condition: str, cohort_sha256: str,
                        slots: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"R5 {condition} verification rows and sidecar are required")
    manifest = json.loads(sidecar.read_text())
    checkpoint_sha = manifest.get("checkpoint_sha256")
    if (manifest.get("verification_sha256") != digest(path)
            or manifest.get("cohort_sha256") != cohort_sha256
            or manifest.get("condition") != condition
            or manifest.get("input_fields") != ["product_smiles", "proposed_precursors"]
            or manifest.get("verification_semantics") != "candidate_conditioned_executor_trace_v1"
            or not manifest.get("checkpoint_identifier")
            or not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
            or any(char not in "0123456789abcdef" for char in checkpoint_sha)
            or not isinstance(manifest.get("max_attempts"), int)
            or not 1 <= manifest["max_attempts"] <= 100):
        raise ValueError(f"R5 {condition} verification provenance/hash mismatch")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("product_smiles"), row.get("rank"))
        if key not in slots or key in rows:
            raise ValueError(f"R5 {condition} unknown/duplicate product-rank slot: {key}")
        candidate = slots[key]
        valid = candidate["smiles_status"] == "valid_smiles"
        expected_input = ({"product_smiles": key[0],
                           "proposed_precursors": candidate["canonical_precursors"]}
                          if valid else None)
        if row.get("model_input") != expected_input:
            raise ValueError(f"R5 {condition} candidate-conditioned input mismatch: {key}")
        status = row.get("verification_status")
        attempts = row.get("attempts")
        if (status not in ({"completed", "failed"} if valid else {"skipped_invalid_or_missing"})
                or not isinstance(attempts, list)
                or len(attempts) > manifest["max_attempts"]
                or (status == "completed" and not attempts)
                or (status != "completed" and attempts)):
            raise ValueError(f"R5 {condition} invalid verification status/attempts: {key}")
        for attempt in attempts:
            if not isinstance(attempt, dict) or not isinstance(attempt.get("termination_reason"), str):
                raise ValueError(f"R5 {condition} invalid trace attempt: {key}")
            final = attempt.get("final_result")
            if final is not None and not isinstance(final, dict):
                raise ValueError(f"R5 {condition} invalid executor result: {key}")
        rows[key] = row
    if set(rows) != set(slots):
        raise ValueError(f"R5 {condition} verification must preserve every frozen product-rank slot")
    return rows, {"checkpoint_identifier": manifest["checkpoint_identifier"],
                  "checkpoint_sha256": checkpoint_sha,
                  "verification_sha256": digest(path), "manifest_sha256": digest(sidecar)}


def _trace_support(attempts: list[dict[str, Any]], precursor: str) -> tuple[bool, int]:
    executed = 0
    supported = False
    for attempt in attempts:
        final = attempt.get("final_result") or {}
        if not (attempt["termination_reason"] == "terminal_tool"
                and final.get("ok") is True
                and final.get("formal_execute") is True
                and final.get("trace_bound") is True
                and final.get("endpoint_source") == "environment_owned_trace"):
            continue
        raw = final.get("structural_precursor")
        if not isinstance(raw, str) or not raw:
            raise ValueError("R5 executed terminal trace lacks structural precursor")
        executed += 1
        if product_key(raw) == precursor:
            supported = True
    return supported, executed


def _condition(rows: list[dict[str, Any]],
               verification: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    totals: Counter[str] = Counter()
    by_rank: dict[int, Counter[str]] = {rank: Counter() for rank in range(1, 6)}
    for row in rows:
        product = row["product_smiles"]
        original = sorted(row["candidates"], key=lambda item: item["rank"])
        ranked = []
        for candidate in original:
            rank = candidate["rank"]
            slot = verification[(product, rank)]
            precursor = candidate["canonical_precursors"]
            supported, executed = (_trace_support(slot["attempts"], precursor)
                                   if precursor is not None else (False, 0))
            recorded = candidate["recorded_reference_status"] == "recorded_reference"
            ranked.append({"original_rank": rank, "precursor": precursor,
                           "recorded_reference": recorded, "trace_supported": supported,
                           "executed_trace_attempts": executed,
                           "verification_status": slot["verification_status"]})
            by_rank[rank]["slots"] += 1
            by_rank[rank]["valid"] += precursor is not None
            by_rank[rank]["supported"] += supported
            totals["executed_trace_attempts"] += executed
            totals["trace_supported_slots"] += supported
        reranked = sorted(ranked, key=lambda item: (not item["trace_supported"], item["original_rank"]))
        before, after = ranked[0], reranked[0]
        totals["known_recorded_at_1_before"] += before["recorded_reference"]
        totals["known_recorded_at_1_after"] += after["recorded_reference"]
        totals["known_recorded_at_5"] += any(item["recorded_reference"] for item in ranked)
        totals["trace_supported_at_1_before"] += before["trace_supported"]
        totals["trace_supported_at_1_after"] += after["trace_supported"]
        details.append({"product_smiles": product,
                        "original_ranks": [item["original_rank"] for item in ranked],
                        "reranked_original_ranks": [item["original_rank"] for item in reranked],
                        "candidates": ranked})
    n = len(rows)
    metrics = {"products": n, "candidate_slots": n * 5,
               "trace_supported_slots": totals["trace_supported_slots"],
               "trace_supported_fraction": totals["trace_supported_slots"] / (n * 5),
               "executed_trace_attempts": totals["executed_trace_attempts"],
               "known_recorded_at_1_before": totals["known_recorded_at_1_before"] / n,
               "known_recorded_at_1_after": totals["known_recorded_at_1_after"] / n,
               "known_recorded_at_5": totals["known_recorded_at_5"] / n,
               "trace_supported_at_1_before": totals["trace_supported_at_1_before"] / n,
               "trace_supported_at_1_after": totals["trace_supported_at_1_after"] / n,
               "unverified_at_1_before": 1 - totals["trace_supported_at_1_before"] / n,
               "unverified_at_1_after": 1 - totals["trace_supported_at_1_after"] / n,
               "by_external_rank": {str(rank): {"slots": bucket["slots"],
                                           "valid": bucket["valid"],
                                           "trace_supported": bucket["supported"]}
                                    for rank, bucket in by_rank.items()}}
    return metrics, details


def score(cohort: Path, base_verification: Path, mech_verification: Path,
          scientific_freeze: Path, output: Path, *, expected_products: int = 200,
          top_k: int = 5) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R5 score package already exists: {output}")
    if top_k != 5 or expected_products <= 0:
        raise ValueError("R5 scorer requires the frozen Top-5 denominator")
    cohort_hash = verify_evaluation_source(cohort, name="r5_external_predictions")
    source_status_path = cohort.parent / "ARTIFACT_STATUS.json"
    if not source_status_path.is_file():
        raise ValueError("R5 source needs an explicit evaluation-allowed status")
    source_manifest = json.loads((cohort.parent / "manifest.json").read_text())
    if (source_manifest.get("target_semantics") != "retrosynthetic_precursor_set"
            or source_manifest.get("products") != expected_products
            or source_manifest.get("ranks_per_product") != top_k):
        raise ValueError("R5 source is not the frozen precursor-set Top-5 cohort")
    if not scientific_freeze.is_file():
        raise FileNotFoundError("R5 scoring requires the matched scientific freeze")
    frozen = json.loads(scientific_freeze.read_text())
    if (frozen.get("engineering_only") is not False
            or frozen.get("evaluation_hashes", {}).get("r5_external_predictions") != cohort_hash
            or not frozen.get("mech_comparison_identifiable")):
        raise ValueError("R5 scoring is not bound to an identifiable scientific freeze")
    rows = [json.loads(line) for line in cohort.read_text().splitlines() if line.strip()]
    if len(rows) != expected_products or len({row["product_smiles"] for row in rows}) != len(rows):
        raise ValueError("R5 product denominator/uniqueness changed")
    slots = {}
    for row in rows:
        if row.get("model_input") != {"product_smiles": row["product_smiles"]}:
            raise ValueError("R5 external query is not product-only")
        if [item["rank"] for item in row["candidates"]] != list(range(1, top_k + 1)):
            raise ValueError("R5 product has missing or reordered rank slots")
        for candidate in row["candidates"]:
            slots[(row["product_smiles"], candidate["rank"])] = candidate
    base, base_prov = _read_verifications(base_verification, condition="base",
                                           cohort_sha256=cohort_hash, slots=slots)
    mech, mech_prov = _read_verifications(mech_verification, condition="mech",
                                           cohort_sha256=cohort_hash, slots=slots)
    base_metrics, base_rows = _condition(rows, base)
    mech_metrics, mech_rows = _condition(rows, mech)
    result = {"artifact_type": "r5_paired_trace_support_result_v1", "package": "r5",
              "status": "complete", "data_contract_errors": 0,
              "evaluation_source_hashes": {"r5_external_predictions": cohort_hash},
              "scientific_freeze_sha256": digest(scientific_freeze),
              "model_checkpoint_sha256": {"base": base_prov["checkpoint_sha256"],
                                          "mech": mech_prov["checkpoint_sha256"]},
              "denominators": {"products": expected_products,
                               "external_candidate_slots": expected_products * top_k},
              "metrics": {"base": base_metrics, "mech": mech_metrics,
                          "mech_minus_base_trace_support_fraction":
                              mech_metrics["trace_supported_fraction"] - base_metrics["trace_supported_fraction"],
                          "mech_minus_base_known_recorded_at_1_after":
                              mech_metrics["known_recorded_at_1_after"] - base_metrics["known_recorded_at_1_after"],
                          "unsupported_at_1": None,
                          "unsupported_at_1_unavailable_reason":
                              "No independent chemical-invalidity labels; absent sampled proof means unverified, not unsupported."},
              "verification_provenance": {"base": base_prov, "mech": mech_prov},
              "external_model_training_overlap_audited":
                  source_manifest.get("training_overlap_audited") is True,
              "headline_allowed": False,
              "claim_boundary": "Candidate-conditioned sampled-trace support and stable reranking only; execution is not proof of unique chemical truth, and unverified candidates are not chemical negatives."}
    output.mkdir(parents=True)
    for condition, details in (("base", base_rows), ("mech", mech_rows)):
        with (output / f"{condition}_rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in details:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--base-verification", required=True, type=Path)
    parser.add_argument("--mech-verification", required=True, type=Path)
    parser.add_argument("--scientific-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.cohort, args.base_verification,
                           args.mech_verification, args.scientific_freeze,
                           args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
