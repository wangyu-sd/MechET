#!/usr/bin/env python3
"""Principal-product-only System-One rollout on the complete HF endpoint split.

Predicted context uses only the full training split; model/tool decisions use
only that proposal and the current executor state. The full-endpoint structural
precursor is read solely by the terminal scorer after each trajectory ends.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.endpoints import structural_exact
from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.eval_system_one_context_bridge import inferred_mixture
from scripts.eval_system_one_full_context_knn import METHOD
from scripts.eval_system_one_import_retrieval import load_imports
from scripts.eval_system_one_product_start_pilot import (
    HybridPolicy, ProductInput, TrainImportRetriever, canonical_visible,
    rollout, verify_model_lineage,
)
from scripts.score_system_one_structural_bridge import replay_provenance
from scripts.train_system_one_electron_flow import verify_source


def load_policy_contexts(path: Path, *, split: str, train_sha: str,
                         heldout_sha: str,
                         product_field: str = "rxn_prod_min") -> tuple[dict[str, tuple[str, tuple[str, ...]]], dict]:
    """Discard held-out context labels before constructing any policy input."""
    report = json.loads((path / "report.json").read_text())
    cases_path = path / "cases.jsonl"
    if (report["artifact_type"] != "system_one_full_endpoint_principal_product_context_proposal"
            or report["method"] != METHOD or report["split"] != split
            or report["train_source"]["sha256"] != train_sha
            or report["heldout_source"]["sha256"] != heldout_sha
            or report.get("product_source_field", "rxn_prod_min") != product_field
            or report["cases_sha256"] != sha256(cases_path)):
        raise ValueError("full endpoint context predictions/source mismatch")
    projected = {}
    for line in cases_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["reaction_id"])
        if reaction_id in projected:
            raise ValueError(f"duplicate predicted context: {reaction_id}")
        predicted = tuple(str(fragment) for fragment in row["predicted_context_batch"])
        if tuple(sorted(predicted)) != predicted:
            raise ValueError(f"unordered predicted context: {reaction_id}")
        projected[reaction_id] = (str(row["product_unmapped"]), predicted)
    if len(projected) != report["evaluated"]:
        raise ValueError("context prediction denominator mismatch")
    return projected, report


def select_ids(ids: list[str], *, seed: int, limit: int) -> list[str]:
    if limit < 0:
        raise ValueError("negative evaluation limit")
    ordered = sorted(ids, key=lambda reaction_id: (
        hashlib.sha256(f"{seed}:{reaction_id}".encode()).hexdigest(), reaction_id
    ))
    return ordered[:limit] if limit else ordered


def static_train_prompt(strict_dir: Path) -> tuple[str, list[dict]]:
    with (strict_dir / "train.jsonl").open() as handle:
        row = json.loads(next(handle))
    if row["metadata"]["decision_index"] != 0:
        raise ValueError("first strict train row is not an initial decision")
    systems = [message["content"] for message in row["messages"]
               if message.get("role") == "system"]
    if len(systems) != 1 or not row.get("tools"):
        raise ValueError("strict train static prompt/tools unavailable")
    return str(systems[0]), row["tools"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--context-run", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--route-checkpoint", type=Path, required=True)
    parser.add_argument("--route-run", type=Path, required=True)
    parser.add_argument("--typed-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--first-event-target-focus", action="store_true",
                        help="validation-only diagnostic: prefer executable first moves touching the input product")
    parser.add_argument("--principal-target-prompt", action="store_true",
                        help="validation-only inference prompt ablation: show the input product separately from current mixture")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit < 0 or args.max_actions < 1 or args.log_every < 1:
        raise ValueError("invalid selection/action/log budget")
    full_manifest = json.loads((args.full_endpoint_dir / "manifest.json").read_text())
    product_field = full_manifest.get("product_source_field", "rxn_prod_min")
    train_path = args.full_endpoint_dir / "train.jsonl"
    full_path = args.full_endpoint_dir / f"{args.split}.jsonl"
    train_declared = full_manifest["splits"]["train"]
    full_declared = full_manifest["splits"][args.split]
    if (product_field not in {"rxn_prod_min", "rxn_prod_equ"}
            or full_manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or full_manifest["executor_filtering"] is not False
            or train_declared["rows"] != 24959
            or train_declared["endpoint_sha256"] != sha256(train_path)
            or full_declared["rows"] != 3120
            or full_declared["endpoint_sha256"] != sha256(full_path)):
        raise ValueError("full endpoint benchmark denominator not preserved")
    full_train_source = {"sha256": train_declared["endpoint_sha256"], "rows": 24959}
    full_source = {"sha256": full_declared["endpoint_sha256"], "rows": 3120}
    contexts, context_report = load_policy_contexts(
        args.context_run, split=args.split,
        train_sha=full_train_source["sha256"], heldout_sha=full_source["sha256"],
        product_field=product_field,
    )
    references = {}
    for line in full_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in references:
            raise ValueError(f"duplicate full endpoint reference: {reaction_id}")
        references[reaction_id] = row
    if len(references) != 3120 or set(references) != set(contexts):
        raise ValueError("context predictions do not cover the full benchmark IDs")
    strict_status = json.loads((args.strict_dir / "ARTIFACT_STATUS.json").read_text())
    strict_manifest = json.loads((args.strict_dir / "manifest.json").read_text())
    if (not strict_status.get("training_allowed")
            or not strict_manifest.get("training_allowed")
            or strict_status.get("artifact_id") != strict_manifest.get("artifact_type")):
        raise ValueError("frozen policy strict-trace training lineage forbidden")
    principal_trained = (
        strict_manifest["artifact_type"]
        == "mech_uspto_31k_natural_language_history_principal_target_v2"
    )
    if principal_trained and (
        product_field != "rxn_prod_equ"
        or not args.principal_target_prompt
        or strict_manifest.get("target_prompt_contract")
           != "endpoint_proxy_product_target_line_with_unchanged_executor_mixture_v2"
        or strict_manifest.get("full_endpoint_manifest_sha256")
           != sha256(args.full_endpoint_dir / "manifest.json")
    ):
        raise ValueError("principal-target v2 evaluation input contract mismatch")
    train_imports, train_source = load_imports(args.strict_dir / "train.jsonl")
    if train_source["import_target_mode"] != (
        "principal_product" if principal_trained else "mixture"
    ):
        raise ValueError("import retriever target mode disagrees with training artifact")
    strict_source = verify_source(args.strict_dir / f"{args.split}.jsonl")
    lineage = verify_model_lineage(
        args.route_checkpoint, args.route_run, args.typed_checkpoint,
        strict_source, train_source,
    )
    system, tools = static_train_prompt(args.strict_dir)
    selected = select_ids(list(references), seed=args.seed, limit=args.limit)
    print(json.dumps({
        "phase": "preflight", "input_contract": "full_endpoint_principal_product_only",
        "product_source_field": product_field,
        "first_event_target_focus": args.first_event_target_focus,
        "principal_target_prompt": args.principal_target_prompt,
        "split": args.split, "full_endpoint_reactions": len(references),
        "selected": len(selected), "context_predictions": len(contexts),
        "full_endpoint_sha256": full_source["sha256"],
        "context_cases_sha256": context_report["cases_sha256"],
    }), flush=True)
    if args.preflight_only:
        return
    policy = HybridPolicy(
        route_checkpoint=args.route_checkpoint, route_run=args.route_run,
        typed_checkpoint=args.typed_checkpoint, lineage=lineage,
    )
    retriever = TrainImportRetriever(train_imports, target_is_principal=principal_trained)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    path = args.output / "cases.jsonl"
    with path.open("w") as handle:
        for index, reaction_id in enumerate(selected, 1):
            principal_product, predicted_batch = contexts[reaction_id]
            reference = references[reaction_id]
            if canonical_visible(principal_product) != canonical_visible(reference["product_unmapped"]):
                raise ValueError(f"{reaction_id}: principal product differs from frozen full endpoint")
            inferred = inferred_mixture(principal_product, predicted_batch)
            policy_input = ProductInput(reaction_id, inferred, system, tools)
            result = rollout(
                policy_input, policy, retriever,
                max_actions=args.max_actions, legality_backoff=True,
                principal_product=principal_product,
                first_event_target_focus=args.first_event_target_focus,
                principal_target_prompt=args.principal_target_prompt,
            )
            result.update({
                "principal_product_input": principal_product,
                "predicted_context_batch": list(predicted_batch),
                "inferred_final_mixture": inferred,
            })
            full_pred, structural_pred, replay_failure = replay_provenance(result)
            structural_hit = bool(result["completed"] and structural_exact(
                structural_pred, str(reference["structural_precursor"])
            ))
            result.update({
                "predicted_full_precursor_replayed": full_pred if result["completed"] else None,
                "predicted_structural_precursor": structural_pred if result["completed"] else None,
                "structural_exact": structural_hit,
                "replay_failure": replay_failure,
                "expected_structural_precursor": str(reference["structural_precursor"]),
            })
            counts["evaluated"] += 1
            counts["structural_exact"] += int(structural_hit)
            counts["completed"] += int(result["completed"])
            counts[f"terminal_{result['terminal']}"] += 1
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            handle.flush()
            if index % args.log_every == 0 or index == len(selected):
                print(json.dumps({"phase": "full_endpoint_rollout", "reactions": index,
                                  "total": len(selected),
                                  "structural_exact": counts["structural_exact"],
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    report = {
        "artifact_type": "system_one_pr81_complete_hf_principal_product_structural_endpoint_evaluation",
        "split": args.split,
        "input_contract": "principal_product_only_with_train_only_predicted_context",
        "product_source_field": product_field,
        "product_selection_is_proxy": True,
        "output_contract": "full_endpoint_structural_precursor_product_origin_projection",
        "policy_training_scope": "strict_executable_trace_view_10152_train_reactions",
        "full_endpoint_source": full_source,
        "full_endpoint_reaction_denominator": 3120,
        "strict_policy_source": strict_source,
        "train_import_source": train_source,
        "context_report_sha256": sha256(args.context_run / "report.json"),
        "context_cases_sha256": context_report["cases_sha256"],
        "context_method": context_report["method"],
        "selection": {"method": "sha256_seed_reaction_id", "seed": args.seed,
                      "limit": args.limit},
        "max_actions": args.max_actions,
        "legality_backoff": True,
        "first_event_target_focus": args.first_event_target_focus,
        "principal_target_prompt": args.principal_target_prompt,
        "evaluated_reactions": counts["evaluated"],
        "structural_exact": counts["structural_exact"],
        "structural_exact_rate": counts["structural_exact"] / counts["evaluated"],
        "completed": counts["completed"],
        "terminal_counts": {key.removeprefix("terminal_"): value for key, value in counts.items()
                            if key.startswith("terminal_")},
        "elapsed_s": time.perf_counter() - started,
        "cases_sha256": sha256(path),
        "evaluator_sha256": sha256(Path(__file__)),
        "provenance_scorer_sha256": sha256(ROOT / "scripts/score_system_one_structural_bridge.py"),
        "weights": {
            "route_adapter_sha256": lineage["route_manifest"]["adapter_model_sha256"],
            "route_head_sha256": lineage["route_report"]["head_sha256"],
            "typed_adapter_sha256": lineage["typed_manifest"]["adapter_model_sha256"],
            "typed_head_sha256": lineage["typed_manifest"]["decision_head_sha256"],
        },
        "limitations": [
            "context proposal is trained on full endpoint reaction pairs, while policy weights are trained only on strict executable traces",
            "full precursor is not MECH_PROOF-compiled; only each accepted electron event is executor-checked",
            "single predicted trajectory per principal product; no Top-K sampling",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "report": report}), flush=True)


if __name__ == "__main__":
    main()
