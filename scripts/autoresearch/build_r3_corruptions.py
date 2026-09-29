#!/usr/bin/env python3
"""Freeze replay-audited, single-decision corruptions for PR #69 R3.

The perturbation is generated without consulting a model.  A gold prefix is
replayed through the Stage-II executor, exactly one electron-flow argument is
changed, and the first rejected or wrong-successor decision is recorded.  The
gold suffix is private evaluation data, never part of a policy observation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest
from scripts.earho_v2_protocol import decision_action, replay_reference
from scripts.run_natural_language_value_search import Action, execute, visible


ALIASES = re.compile(r"\bA\d{2,}\b")
BUCKETS = ("one", "two", "three_plus")
DEPTHS = ("early", "middle", "late")


def coordination(row: dict[str, Any]) -> str:
    steps = row["metadata"]["trace_plan"]["steps"]
    maximum = max((len(step.get("moves") or ()) for step in steps), default=0)
    if maximum < 1:
        raise ValueError("trace has no electron moves")
    return "one" if maximum == 1 else "two" if maximum == 2 else "three_plus"


def event_coordination(step: dict[str, Any]) -> str:
    count = len(step.get("moves") or ())
    if count < 1:
        raise ValueError("electron event has no moves")
    return "one" if count == 1 else "two" if count == 2 else "three_plus"


def in_closed_shell_two_electron_scope(row: dict[str, Any]) -> bool:
    """Reject radical and aggregate-BE trajectories, not just the selected event."""
    for step in row["metadata"]["trace_plan"]["steps"]:
        for move in step.get("moves") or ():
            if move.get("mode") or int(move.get("electrons", 0)) != 2:
                return False
            if (move.get("source") or {}).get("kind") == "RADICAL_PAIR":
                return False
            if (move.get("sink") or {}).get("kind") == "RADICAL_PAIR":
                return False
    return True


def states_closed_shell(row: dict[str, Any]) -> bool:
    """Reject latent open-shell states even when arrows use ordinary containers."""
    from rdkit import Chem

    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    steps = row["metadata"]["trace_plan"]["steps"]
    states = [row["target_smiles"]]
    states.extend(state for step in steps
                  for state in (step["state_before"], step["state_after"]))
    for state in states:
        molecule = Chem.MolFromSmiles(state, parameters)
        if molecule is None or any(atom.GetNumRadicalElectrons()
                                   for atom in molecule.GetAtoms()):
            return False
    return True


def event_positions(decisions: list[dict[str, Any]]) -> list[int]:
    return [index for index, row in enumerate(decisions)
            if decision_action(row)[0] == "apply_electron_flow"]


def depth_positions(events: list[int]) -> dict[str, int]:
    if len(events) < 3:
        raise ValueError("R3 early/middle/late requires three distinct electron events")
    return {"early": events[0], "middle": events[len(events) // 2],
            "late": events[-1]}


def _replacement_arguments(arguments: dict[str, Any], field: str,
                           original: str, replacement: str, move_index: int) -> dict[str, Any]:
    changed = copy.deepcopy(arguments)
    moves = changed.get("electron_flow") or []
    if move_index >= len(moves):
        raise ValueError("event move index outside action")
    move = moves[move_index]
    value = str(move.get(field) or "")
    if not ALIASES.search(value) or original not in value:
        raise ValueError("mutation target alias is absent")
    move[field] = re.sub(rf"\b{re.escape(original)}\b", replacement, value)
    # This is descriptive text only, but keep the submitted action coherent.
    instruction = str(move.get("instruction") or "")
    if instruction:
        move["instruction"] = re.sub(
            rf"\b{re.escape(original)}\b", replacement, instruction)
    return changed


def corrupt(reference: Any, index: int) -> dict[str, Any] | None:
    name, arguments, _ = decision_action(reference.decisions[index])
    if name != "apply_electron_flow":
        raise ValueError("R3 only corrupts electron-flow decisions")
    node = reference.nodes[index]
    gold_successor = reference.nodes[index + 1]
    prompt = next(str(message.get("content") or "")
                  for message in reference.decisions[index]["messages"]
                  if message.get("role") == "user")
    available = sorted(set(ALIASES.findall(prompt)))
    moves = arguments.get("electron_flow") or []
    rejected: dict[str, Any] | None = None
    def check(candidate: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal rejected
        child, error = execute(node, Action(
            name=name, arguments=candidate, raw="r3_controlled_mutation",
            logprob=0.0, tokens=1), max_imports=64)
        record = {"name": name, "arguments": candidate, "mutation": mutation,
                  "first_failure_index": index, "prefix_length": index}
        if child is not None:
            if visible(child.state) == visible(gold_successor.state):
                return None
            record.update({"failure_kind": "accepted_wrong_successor",
                           "observed_successor": visible(child.state),
                           "executor_error": None})
            return record
        if rejected is None:
            record.update({"failure_kind": "rejected_action",
                           "observed_successor": None, "executor_error": error})
            rejected = record
        return None

    for move_index, move in enumerate(moves):
        for field in ("destination", "source"):
            value = str(move.get(field) or "")
            for original in sorted(set(ALIASES.findall(value))):
                for replacement in available:
                    if replacement == original or replacement in set(ALIASES.findall(value)):
                        continue
                    candidate = _replacement_arguments(
                        arguments, field, original, replacement, move_index)
                    found = check(candidate, {"move_index": move_index,
                                              "field": field, "from": original,
                                              "to": replacement})
                    if found is not None:
                        return found
    # Some FlowER events are faithfully encoded as aggregate BE deltas rather
    # than source/sink arrows.  Mutate the local bond address in those actions
    # as well; otherwise the one-move coordination stratum vanishes entirely.
    for change_index, change in enumerate(arguments.get("bond_order_changes") or []):
        atoms = list(change.get("atoms") or [])
        for atom_index, original in enumerate(atoms):
            if original not in available:
                continue
            for replacement in available:
                if replacement == original or replacement in atoms:
                    continue
                candidate = copy.deepcopy(arguments)
                altered = candidate["bond_order_changes"][change_index]
                altered["atoms"][atom_index] = replacement
                if altered.get("instruction"):
                    altered["instruction"] = re.sub(
                        rf"\b{re.escape(original)}\b", replacement,
                        str(altered["instruction"]))
                found = check(candidate, {"bond_change_index": change_index,
                                          "atom_index": atom_index,
                                          "field": "bond_order_changes.atoms",
                                          "from": original, "to": replacement})
                if found is not None:
                    return found
    return rejected


def _expected_hash(manifest: dict[str, Any], split: str) -> str:
    item = manifest["splits"][split]
    return str(item.get("sha256") or item.get("output_sha256") or "")


def public_executor_result(mutated: dict[str, Any]) -> dict[str, Any]:
    """Expose only real execution feedback, never reference-relative wrongness."""
    error = mutated["executor_error"]
    return {"ok": error is None, "code": "PASS" if error is None else "REJECTED",
            "current_state": mutated["observed_successor"], "error": error}


def build(trace_source: Path, history_source: Path, trace_manifest: Path,
          history_manifest: Path, output: Path, *, seed: int = 17,
          per_cell: int = 32, candidate_limit: int = 512) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3 output already exists: {output}")
    trace_meta = json.loads(trace_manifest.read_text())
    history_meta = json.loads(history_manifest.read_text())
    trace_hash, history_hash = digest(trace_source), digest(history_source)
    if trace_hash != _expected_hash(trace_meta, "test"):
        raise ValueError("R3 trace source differs from frozen test manifest")
    if history_hash != _expected_hash(history_meta, "test"):
        raise ValueError("R3 history source differs from frozen test manifest")
    if candidate_limit < per_cell:
        raise ValueError("candidate_limit must be at least per_cell")
    pools: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    audit = Counter()
    with trace_source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rid = str(row.get("source_id") or row["id"])
            steps = row["metadata"]["trace_plan"]["steps"]
            if len(steps) < 3:
                continue
            if not in_closed_shell_two_electron_scope(row):
                audit["out_of_scope_reactions"] += 1
                continue
            for depth, step in zip(DEPTHS, (steps[0], steps[len(steps) // 2], steps[-1])):
                bucket = event_coordination(step)
                cell = f"{depth}/{bucket}"
                priority = int.from_bytes(hashlib.sha256(
                    f"{seed}:{cell}:{rid}".encode()).digest(), "big")
                item = (-priority, rid, row)
                if len(pools[cell]) < candidate_limit:
                    heapq.heappush(pools[cell], item)
                elif item[:2] > pools[cell][0][:2]:
                    heapq.heapreplace(pools[cell], item)
    # Cap replay to deterministic per-cell candidates, after the chemistry gate.
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for depth in DEPTHS:
        for bucket in BUCKETS:
            cell = f"{depth}/{bucket}"
            ranked = sorted(pools[cell], key=lambda item: (-item[0], item[1]))
            candidates[cell] = {rid: row for _, rid, row in ranked}
    wanted = {rid for rows in candidates.values() for rid in rows}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with history_source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rid = str(row.get("source_id") or "")
            if rid in wanted:
                histories[rid].append(row)
    records: list[dict[str, Any]] = []
    replay_cache: dict[str, Any] = {}
    scope_cache: dict[str, bool] = {}
    for depth in DEPTHS:
        for bucket in BUCKETS:
            cell = f"{depth}/{bucket}"
            accepted = 0
            for rid, source in candidates[cell].items():
                if accepted >= per_cell:
                    break
                try:
                    if rid not in scope_cache:
                        scope_cache[rid] = states_closed_shell(source)
                    if not scope_cache[rid]:
                        audit[f"out_of_scope_radical_states_{cell}"] += 1
                        continue
                    if rid not in replay_cache:
                        replay_cache[rid] = replay_reference(source, histories.get(rid, []))
                    reference = replay_cache[rid]
                    events = event_positions(histories[rid])
                    steps = source["metadata"]["trace_plan"]["steps"]
                    if len(events) != len(steps):
                        raise ValueError("reference event count differs from trace steps")
                    index = depth_positions(events)[depth]
                    step_index = (0 if depth == "early" else
                                  len(steps) // 2 if depth == "middle" else len(steps) - 1)
                    if event_coordination(steps[step_index]) != bucket:
                        raise ValueError("event coordination cell drifted")
                    mutated = corrupt(reference, index)
                    if mutated is None:
                        raise ValueError(f"no controlled corruption at {depth}")
                    name, gold_arguments, _ = decision_action(reference.decisions[index])
                    if len(gold_arguments.get("electron_flow") or []) != len(steps[step_index]["moves"]):
                        raise ValueError("decision electron moves differ from trace step")
                    records.append({
                        "artifact_type": "r3_controlled_corruption_v1",
                        "reaction_id": rid,
                        "source_dataset": "FlowER strict executable test",
                        "source_split": "test",
                        "target_smiles": reference.target,
                        "strata": {"failure_depth": depth,
                                   "event_coordination": bucket,
                                   "reaction_max_coordination": coordination(source),
                                   "corrupted_event_moves": len(gold_arguments.get("electron_flow") or []),
                                   "reference_event_count": len(events)},
                        "model_visible": {
                            "target_smiles": reference.target,
                            "prefix_actions": [
                                {"name": action["name"], "arguments": action["arguments"],
                                 "result": action["result"]}
                                for action in reference.nodes[index].actions],
                            "corrupted_action": {"name": mutated["name"],
                                                 "arguments": mutated["arguments"]},
                            "executor_result": public_executor_result(mutated),
                        },
                        "private_reference": {
                            "first_failure_index": index,
                            "corruption_kind": mutated["failure_kind"],
                            "expected_precursor": reference.expected_precursor,
                            "expected_successor": visible(reference.nodes[index + 1].state),
                            "correct_action": {"name": name, "arguments": gold_arguments},
                            "suffix_actions": [
                                {"name": decision_action(item)[0],
                                 "arguments": decision_action(item)[1]}
                                for item in reference.decisions[index + 1:]],
                        },
                        "corruption": mutated["mutation"],
                    })
                    accepted += 1
                    audit[f"accepted_{cell}"] += 1
                except (ValueError, KeyError, StopIteration) as exc:
                    audit[f"quarantined_{cell}"] += 1
                    audit[f"reason_{type(exc).__name__}"] += 1
            if accepted < per_cell:
                audit[f"underfilled_{cell}"] = per_cell - accepted
    output.mkdir(parents=True)
    cohort = output / "r3_corruptions.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    cell_counts = {f"{depth}/{bucket}": sum(
        row["strata"]["failure_depth"] == depth
        and row["strata"]["event_coordination"] == bucket
        for row in records) for depth in DEPTHS for bucket in BUCKETS}
    report = {
        "artifact_type": "r3_controlled_corruption_manifest_v1",
        "trace_source": str(trace_source), "trace_sha256": trace_hash,
        "history_source": str(history_source), "history_sha256": history_hash,
        "trace_manifest_sha256": digest(trace_manifest),
        "history_manifest_sha256": digest(history_manifest),
        "cohort": str(cohort), "cohort_sha256": digest(cohort),
        "rows": len(records), "unique_reactions": len({row["reaction_id"] for row in records}),
        "cell_counts": cell_counts, "failure_kind_counts": dict(Counter(
            row["private_reference"]["corruption_kind"] for row in records)),
        "audit": dict(audit), "seed": seed, "per_cell_target": per_cell,
        "candidate_limit_per_bucket": candidate_limit,
        "independence_note": "Cells may share reaction IDs; cluster uncertainty by reaction.",
        "chemistry_scope": "all steps are explicit two-electron source/sink moves and all recorded states have zero RDKit radical electrons; radical and aggregate BE-delta traces excluded",
        "claim_boundary": "Controlled first divergence only; localization and repair model scores not yet measured.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-source", type=Path, required=True)
    parser.add_argument("--history-source", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--history-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-cell", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(build(args.trace_source, args.history_source,
                           args.trace_manifest, args.history_manifest,
                           args.output, seed=args.seed, per_cell=args.per_cell,
                           candidate_limit=args.candidate_limit),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
