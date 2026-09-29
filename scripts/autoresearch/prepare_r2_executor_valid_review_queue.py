#!/usr/bin/env python3
"""Prepare, but do not label, executor-valid R2 counterfactual precursors.

The frozen R2 positives provide products and recorded precursor sets. For each
matching strict-executable FlowER trace, mutate the *last* electron event,
then replay only trailing reference imports/finish. The resulting structural
precursor is a chemical-review candidate, never an automatic negative: an
unrecorded route or non-reference intermediate can still be chemically sound.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.autoresearch.build_r3_corruptions import (
    corrupt, event_positions, in_closed_shell_two_electron_scope,
    states_closed_shell,
)
from scripts.autoresearch.stratified_manifest import digest, product_key
from scripts.earho_v2_protocol import decision_action, replay_reference
from scripts.run_natural_language_value_search import Action, execute, visible
from mechet.endpoints import split_precursor_endpoints


def _source_hash(manifest: Path, split: str) -> str:
    meta = json.loads(manifest.read_text())
    return str(meta["splits"][split].get("sha256")
               or meta["splits"][split].get("output_sha256") or "")


def _load_positives(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((path.parent / "manifest.json").read_text())
    status = json.loads((path.parent / "ARTIFACT_STATUS.json").read_text())
    source_hash = digest(path)
    if (manifest.get("cohort_sha256") != source_hash
            or manifest.get("positive_proposals") != 400
            or status.get("cohort_sha256") != source_hash
            or status.get("positive_source_allowed") is not True
            or status.get("evaluation_allowed") is not False):
        raise ValueError("R2 positive source is not the frozen 400-row component")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    positives: dict[str, dict[str, Any]] = {}
    for row in rows:
        product = row["model_input"]["product_smiles"]
        if (product_key(product) != product or product in positives
                or row["source_split"] != "test"):
            raise ValueError("R2 positive product is duplicate, noncanonical or not held out")
        positives[product] = row
    if len(positives) != 400:
        raise ValueError("R2 positive denominator changed")
    return positives


def _known_references(path: Path, products: set[str]) -> dict[str, set[str]]:
    references: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(str(row["target_smiles"]))
            if product in products:
                references[product].add(product_key(str(row["structural_precursor"])))
    if set(references) != products:
        raise ValueError("R2 products absent from official full-endpoint test")
    return references


def _priority(seed: int, product: str, reaction_id: str) -> str:
    return hashlib.sha256(f"{seed}:{product}:{reaction_id}".encode()).hexdigest()


def _terminal_counterfactual(source: dict[str, Any],
                             decisions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    if not in_closed_shell_two_electron_scope(source) or not states_closed_shell(source):
        return None, "out_of_closed_shell_scope"
    reference = replay_reference(source, decisions)
    events = event_positions(decisions)
    if not events:
        return None, "no_electron_event"
    index = events[-1]
    mutated = corrupt(reference, index)
    if mutated is None or mutated["failure_kind"] != "accepted_wrong_successor":
        return None, "no_accepted_final_event_mutation"
    child, error = execute(reference.nodes[index], Action(
        name=mutated["name"], arguments=mutated["arguments"],
        raw="r2_review_candidate", logprob=0.0, tokens=1), max_imports=64)
    if child is None:
        return None, "mutation_replay_rejected:" + error
    if visible(child.state) == visible(reference.nodes[index + 1].state):
        return None, "no_successor_divergence"
    for decision in decisions[index + 1:]:
        name, arguments, _ = decision_action(decision)
        if name == "apply_electron_flow":
            raise ValueError("selected event was not the final electron event")
        child, error = execute(child, Action(
            name=name, arguments=arguments, raw="r2_reference_tail",
            logprob=0.0, tokens=1), max_imports=64)
        if child is None:
            return None, "suffix_replay_rejected:" + error
    if not child.terminal:
        return None, "counterfactual_not_terminal"
    endpoints = split_precursor_endpoints(child.state, str(source["target_smiles"]))
    precursor = product_key(endpoints.structural)
    return {
        "proposed_precursors": precursor,
        "mapped_full_precursor_state": endpoints.full,
        "mapped_structural_precursor": endpoints.structural,
        "mapped_auxiliary_fragments": list(endpoints.auxiliary),
        "event_decision_index": index,
        "event_mutation": mutated["mutation"],
        "mutated_action": {"name": mutated["name"], "arguments": mutated["arguments"]},
        "reference_action": {"name": decision_action(decisions[index])[0],
                             "arguments": decision_action(decisions[index])[1]},
        "reference_successor": visible(reference.nodes[index + 1].state),
        "counterfactual_successor": mutated["observed_successor"],
    }, "candidate"


def build(positives: Path, official_test: Path, official_manifest: Path,
          trace_source: Path, history_source: Path, r3_manifest: Path,
          output: Path, *, seed: int = 17, max_traces_per_product: int = 5) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R2 review queue already exists: {output}")
    if max_traces_per_product < 1:
        raise ValueError("max_traces_per_product must be positive")
    if digest(official_test) != _source_hash(official_manifest, "test"):
        raise ValueError("R2 official full-endpoint test hash drifted")
    lineage = json.loads(r3_manifest.read_text())
    if (digest(trace_source) != lineage["trace_sha256"]
            or digest(history_source) != lineage["history_sha256"]):
        raise ValueError("R2 strict trace/history sources differ from frozen R3 lineage")
    by_product = _load_positives(positives)
    known = _known_references(official_test, set(by_product))
    candidates_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    with trace_source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(str(row["target_smiles"]))
            if product not in by_product:
                continue
            counts["matching_product_trace_rows"] += 1
            if product_key(str(row["structural_precursor"])) != by_product[product]["model_input"]["proposed_precursors"]:
                continue
            reaction_id = str(row.get("source_id") or row["id"])
            candidates_by_product[product].append(row)
    selected: dict[str, tuple[str, dict[str, Any]]] = {}
    for product, rows in candidates_by_product.items():
        ranked = sorted(rows, key=lambda row: _priority(
            seed, product, str(row.get("source_id") or row["id"])))
        for row in ranked[:max_traces_per_product]:
            reaction_id = str(row.get("source_id") or row["id"])
            if reaction_id in selected:
                raise ValueError("selected R2 trace reaction ID is duplicated")
            selected[reaction_id] = (product, row)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with history_source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            reaction_id = str(row.get("source_id") or "")
            if reaction_id in selected:
                histories[reaction_id].append(row)
    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for reaction_id, (product, source) in sorted(selected.items()):
        counts["attempted_traces"] += 1
        try:
            counterfactual, reason = _terminal_counterfactual(
                source, histories.get(reaction_id, []))
        except (ValueError, KeyError, IndexError, StopIteration) as exc:
            counts["replay_exception_" + type(exc).__name__] += 1
            continue
        counts[reason.split(":", 1)[0]] += 1
        if counterfactual is None:
            continue
        precursor = counterfactual["proposed_precursors"]
        if precursor in known[product]:
            counts["matched_another_recorded_reference"] += 1
            continue
        pair = (product, precursor)
        if pair in seen_pairs:
            counts["duplicate_product_precursor_pair"] += 1
            continue
        seen_pairs.add(pair)
        records.append({
            "artifact_type": "r2_executor_valid_counterfactual_review_candidate_v1",
            "candidate_id": hashlib.sha256(f"{product}\0{precursor}\0{reaction_id}".encode()).hexdigest(),
            "model_input": {"product_smiles": product, "proposed_precursors": precursor},
            "source_positive_proposal_id": by_product[product]["proposal_id"],
            "source_trace_reaction_id": reaction_id,
            "source_split": "test",
            "review_context": {**counterfactual,
                               "known_recorded_precursors": sorted(known[product]),
                               "original_recorded_precursors": by_product[product]["model_input"]["proposed_precursors"]},
            "review_status": "pending_independent_chemistry_review",
            "chemical_negative_label": None,
        })
    records.sort(key=lambda row: row["candidate_id"])
    output.mkdir(parents=True)
    cohort = output / "review_candidates.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "artifact_type": "r2_executor_valid_review_queue_manifest_v1",
        "candidate_count": len(records),
        "distinct_products": len({row["model_input"]["product_smiles"] for row in records}),
        "cohort": str(cohort), "cohort_sha256": digest(cohort),
        "positive_source_sha256": digest(positives),
        "official_test_sha256": digest(official_test),
        "trace_source_sha256": digest(trace_source),
        "history_source_sha256": digest(history_source),
        "r3_manifest_sha256": digest(r3_manifest),
        "seed": seed, "max_traces_per_product": max_traces_per_product,
        "audit": dict(sorted(counts.items())),
        "evaluation_allowed": False,
        "training_allowed": False,
        "claim_boundary": "Executor-accepted counterfactuals for independent review only; a non-reference endpoint is not a chemical negative.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r2_executor_valid_review_queue_status_v1",
        "cohort_sha256": report["cohort_sha256"],
        "evaluation_allowed": False, "training_allowed": False,
        "evidence_audited": False, "human_review_required": True,
    }, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", required=True, type=Path)
    parser.add_argument("--official-test", required=True, type=Path)
    parser.add_argument("--official-manifest", required=True, type=Path)
    parser.add_argument("--trace-source", required=True, type=Path)
    parser.add_argument("--history-source", required=True, type=Path)
    parser.add_argument("--r3-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-traces-per-product", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(build(args.positives, args.official_test, args.official_manifest,
                           args.trace_source, args.history_source, args.r3_manifest,
                           args.output, seed=args.seed,
                           max_traces_per_product=args.max_traces_per_product),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
