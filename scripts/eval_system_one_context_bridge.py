#!/usr/bin/env python3
"""Principal-product-input context bridge into frozen PR81 mixture policy.

Only the full-endpoint principal product enters from the held-out reaction.
A train-only context retriever predicts missing final-mixture components;
the frozen mixture-trained policy then rolls out from that predicted mixture.
The scorer remains the *strict trace-view full precursor*, so this evaluates
an input bridge on the trace overlap, NOT the full 3,120-row benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.eval_system_one_import_retrieval import append_import_batch, load_imports
from scripts.eval_system_one_product_start_pilot import (
    HybridPolicy, ProductInput, TrainImportRetriever, canonical_visible,
    load_reactions, rollout, select_tasks, verify_model_lineage,
)


def load_policy_contexts(path: Path, *, split: str, strict_source_sha256: str,
                         train_source_sha256: str) -> tuple[dict[str, tuple[str, tuple[str, ...]]], dict]:
    """Project a scored context artifact to policy-only fields, discarding GT."""
    report = json.loads((path / "report.json").read_text())
    cases_path = path / "cases.jsonl"
    allowed_methods = {
        "train_only_morgan_radius2_2048_nearest_reaction_distinct_context_batches",
        "train_only_morgan_radius2_2048_weighted_knn_k11_p2",
    }
    if (report["artifact_type"] != "system_one_principal_product_context_retrieval_diagnostic"
            or report["split"] != split
            or report["method"] not in allowed_methods
            or report["heldout_source"]["strict_source_sha256"] != strict_source_sha256
            or report["train_source"]["strict_source_sha256"] != train_source_sha256
            or report["cases_sha256"] != sha256(cases_path)):
        raise ValueError("context predictions do not match frozen train/heldout source")
    projected = {}
    for line in cases_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["reaction_id"])
        if reaction_id in projected:
            raise ValueError(f"duplicate context prediction: {reaction_id}")
        predicted = tuple(str(fragment) for fragment in row["predicted_context_batch"])
        if tuple(sorted(predicted)) != predicted:
            raise ValueError(f"unordered context batch: {reaction_id}")
        # Do not carry reference_context_batch or top1_exact into the policy.
        projected[reaction_id] = (str(row["product_unmapped"]), predicted)
    if len(projected) != report["evaluated"]:
        raise ValueError("context prediction denominator mismatch")
    return projected, report


def inferred_mixture(product: str, predicted_batch: tuple[str, ...]) -> str:
    counts = Counter(predicted_batch)
    batch = tuple(sorted(counts.items()))
    return append_import_batch(product, batch) if batch else canonical_visible(product)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--context-run", type=Path, required=True)
    parser.add_argument("--route-checkpoint", type=Path, required=True)
    parser.add_argument("--route-run", type=Path, required=True)
    parser.add_argument("--typed-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit < 0 or args.max_actions < 1 or args.log_every < 1:
        raise ValueError("invalid bridge limit or action/log budget")
    status = json.loads((args.data_dir / "ARTIFACT_STATUS.json").read_text())
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if (status.get("artifact_id") != manifest.get("artifact_type")
            or not status.get("training_allowed") or not manifest.get("training_allowed")):
        raise ValueError("bridge requires validated current-compiler strict trace view")
    train_imports, train_source = load_imports(args.data_dir / "train.jsonl")
    tasks, source = load_reactions(args.data_dir / f"{args.split}.jsonl")
    contexts, context_report = load_policy_contexts(
        args.context_run, split=args.split,
        strict_source_sha256=source["sha256"],
        train_source_sha256=train_source["sha256"],
    )
    if {task.policy_input.reaction_id for task in tasks} != set(contexts):
        raise ValueError("context predictions do not cover identical strict reaction IDs")
    selected = select_tasks(tasks, seed=args.seed, limit=args.limit)
    lineage = verify_model_lineage(
        args.route_checkpoint, args.route_run, args.typed_checkpoint, source, train_source
    )
    print(json.dumps({"phase": "preflight", "split": args.split,
                      "input_contract": "full_endpoint_principal_product_only_to_train_context_prediction",
                      "strict_reactions": len(tasks), "selected": len(selected),
                      "context_predictions": len(contexts),
                      "source_sha256": source["sha256"],
                      "context_cases_sha256": context_report["cases_sha256"]}), flush=True)
    if args.preflight_only:
        return
    policy = HybridPolicy(route_checkpoint=args.route_checkpoint, route_run=args.route_run,
                          typed_checkpoint=args.typed_checkpoint, lineage=lineage)
    import_retriever = TrainImportRetriever(train_imports)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("w") as handle:
        for index, task in enumerate(selected, 1):
            reaction_id = task.policy_input.reaction_id
            principal_product, predicted_batch = contexts[reaction_id]
            completed_target = inferred_mixture(principal_product, predicted_batch)
            policy_input = ProductInput(
                reaction_id, completed_target,
                task.policy_input.system, task.policy_input.tools,
            )
            result = rollout(
                policy_input, policy, import_retriever,
                max_actions=args.max_actions, legality_backoff=True,
            )
            exact = bool(result["completed"] and canonical_visible(
                result["predicted_precursor"]
            ) == canonical_visible(task.expected_precursor))
            result.update({
                "principal_product_input": principal_product,
                "predicted_context_batch": list(predicted_batch),
                "inferred_final_mixture": completed_target,
                "strict_reference_final_mixture": task.policy_input.target,
                "strict_mixture_reconstructed_byte_exact": (
                    completed_target == task.policy_input.target
                ),
                "endpoint_exact_strict_full_precursor": exact,
                "expected_strict_full_precursor": task.expected_precursor,
                "reference_decisions": task.reference_decisions,
            })
            counts["evaluated"] += 1
            counts["endpoint_exact_strict_full_precursor"] += int(exact)
            counts["strict_mixture_reconstructed_byte_exact"] += int(
                result["strict_mixture_reconstructed_byte_exact"]
            )
            counts[f"terminal_{result['terminal']}"] += 1
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            handle.flush()
            if index % args.log_every == 0 or index == len(selected):
                print(json.dumps({"phase": "bridge_rollout", "reactions": index,
                                  "total": len(selected),
                                  "strict_endpoint_exact": counts["endpoint_exact_strict_full_precursor"],
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    report = {
        "artifact_type": "system_one_pr81_principal_product_context_bridge_diagnostic",
        "scope": "principal_product_only_input_strict_trace_view_full_precursor_scorer_not_full_3120_benchmark",
        "split": args.split,
        "strict_source": source,
        "train_import_source": train_source,
        "context_report_sha256": sha256(args.context_run / "report.json"),
        "context_cases_sha256": context_report["cases_sha256"],
        "context_method": context_report["method"],
        "full_endpoint_source_sha256": context_report["heldout_source"]["full_endpoint_sha256"],
        "strict_reaction_denominator": len(tasks),
        "evaluated_reactions": counts["evaluated"],
        "selection": {"method": "sha256_seed_reaction_id", "seed": args.seed,
                      "limit": args.limit},
        "max_actions": args.max_actions,
        "legality_backoff": True,
        "policy": "train_only_product_context_retrieval_then_frozen_v1_route_typed_v2_electrons_top8_legal_backoff",
        "endpoint_exact_strict_full_precursor": counts["endpoint_exact_strict_full_precursor"],
        "endpoint_exact_rate": counts["endpoint_exact_strict_full_precursor"] / counts["evaluated"],
        "strict_mixture_reconstructed_byte_exact": counts["strict_mixture_reconstructed_byte_exact"],
        "terminal_counts": {key.removeprefix("terminal_"): value for key, value in counts.items()
                            if key.startswith("terminal_")},
        "elapsed_s": time.perf_counter() - started,
        "cases_sha256": sha256(cases_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "weights": {
            "route_adapter_sha256": lineage["route_manifest"]["adapter_model_sha256"],
            "route_head_sha256": lineage["route_report"]["head_sha256"],
            "typed_adapter_sha256": lineage["typed_manifest"]["adapter_model_sha256"],
            "typed_head_sha256": lineage["typed_manifest"]["decision_head_sha256"],
        },
        "limitations": [
            "only the current-compiler strict trace subset has a full-precursor scorer",
            "the full endpoint benchmark evaluates structural precursors on all 3120 reactions",
            "principal-product stereo absent from the full endpoint source is not restored",
            "each electron step executes; whole predicted trajectory is not MECH_PROOF-compiled",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "report": report}), flush=True)


if __name__ == "__main__":
    main()
