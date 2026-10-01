#!/usr/bin/env python3
"""Reaction-level complexity audit with the declared source as denominator.

Predictions may cover only a subset; absent IDs count as unevaluated, never
as correct. The output separates chemistry-complexity strata from runtime
failures and does not reinterpret a trace view as a full benchmark split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from rdkit import Chem


def bond_table(smiles: str) -> dict[tuple[int, int], tuple[float, bool]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid mapped trajectory state")
    output = {}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if a and b:
            output[tuple(sorted((a, b)))] = (bond.GetBondTypeAsDouble(), bond.IsInRing())
    return output


def reaction_features(row: dict) -> dict[str, int | bool]:
    steps = list(((row.get("metadata") or {}).get("trace_plan") or {}).get("steps") or [])
    if not steps:
        raise ValueError(f"{row.get('source_id')}: missing frozen trace plan")
    atoms, changed = set(), set()
    flows = imports = 0
    ring_change = False
    for step in steps:
        for move in step.get("moves") or []:
            flows += 1
            atoms.update(int(atom) for side in ("source", "sink") for atom in move[side]["atoms"])
        imports += len(step.get("imports") or [])
        before = bond_table(str(step["state_before"]))
        after = bond_table(str(step["state_after"]))
        for pair in before.keys() | after.keys():
            lhs, rhs = before.get(pair), after.get(pair)
            if lhs is None or rhs is None or lhs[0] != rhs[0]:
                changed.add(pair)
                ring_change |= bool((lhs and lhs[1]) or (rhs and rhs[1]))
    return {"reacting_atoms": len(atoms), "changed_bonds": len(changed),
            "ring_change": ring_change, "electron_flows": flows,
            "fragment_imports": imports, "trajectory_length": len(steps)}


def category(value: int, *, edges: tuple[int, ...]) -> str:
    for upper in edges:
        if value <= upper:
            return str(upper) if upper == edges[0] else f"<={upper}"
    return f">{edges[-1]}"


def aggregate(source: list[dict], predictions: list[dict]) -> dict:
    by_id = {str(row["source_id"]): row for row in source}
    if len(by_id) != len(source):
        raise ValueError("source contains duplicate reaction IDs")
    predicted = {str(row["source_id"]): row for row in predictions}
    if len(predicted) != len(predictions) or predicted.keys() - by_id.keys():
        raise ValueError("prediction IDs duplicate or outside declared source")
    axes = {
        "reacting_atoms": (2, 4, 8), "changed_bonds": (1, 2, 4),
        "electron_flows": (1, 2, 3), "fragment_imports": (0, 1, 2),
        "trajectory_length": (1, 2, 3),
    }
    counters: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for rid, row in by_id.items():
        features = reaction_features(row)
        observation = predicted.get(rid)
        tags = {axis: category(int(features[axis]), edges=edges) for axis, edges in axes.items()}
        tags["ring_change"] = "yes" if features["ring_change"] else "no"
        tags["all"] = "all"
        for axis, group in tags.items():
            slot = counters[axis][group]
            slot["source_reactions"] += 1
            if observation is not None:
                slot["evaluated"] += 1
                slot["top1_endpoint_exact"] += int(bool(observation.get("top1_exact")))
                slot["top1_full_exact"] += int(bool(observation.get("top1_full_exact")))
                slot["pass_at_beam"] += int(bool(observation.get("pass_at_beam")))
                slot["top_terminal"] += int(bool(observation.get("top_terminal")))
    return {axis: {group: dict(counts) for group, counts in sorted(groups.items())}
            for axis, groups in sorted(counters.items())}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "prediction_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.prediction},
        "metric_contract": "reaction_level_missing_predictions_are_unevaluated",
        "strata": aggregate(read_jsonl(args.source),
                            [r for path in args.prediction for r in read_jsonl(path)]),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["strata"]["all"]["all"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
