#!/usr/bin/env python3
"""Audit strict-trace full precursors against the full-endpoint HF contract.

This is a reaction-ID join and reference-only diagnostic. It is not a policy
scorer and never supplies held-out precursor information to a policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.proof_program import sides_equal
from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.eval_system_one_product_start_pilot import canonical_visible
from scripts.train_system_one_electron_flow import verify_source


def components_without_stereo(smiles: str) -> tuple[str, ...]:
    out = []
    for fragment in smiles.split("."):
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            raise ValueError(f"unparseable precursor fragment: {fragment}")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        out.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
    return tuple(sorted(out))


def audit(strict_dir: Path, full_dir: Path, split: str) -> tuple[dict, list[dict]]:
    strict_path = strict_dir / f"{split}.jsonl"
    full_path = full_dir / f"{split}.jsonl"
    strict_source = verify_source(strict_path)
    full_manifest = json.loads((full_dir / "manifest.json").read_text())
    full_declared = full_manifest["splits"][split]
    if (full_manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or full_manifest["executor_filtering"] is not False
            or full_declared["rows"] != 3120
            or full_declared["endpoint_sha256"] != sha256(full_path)):
        raise ValueError("full endpoint manifest or split hash mismatch")
    full = {}
    for line in full_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in full:
            raise ValueError(f"duplicate full endpoint ID: {reaction_id}")
        full[reaction_id] = row
    if len(full) != 3120:
        raise ValueError("full endpoint denominator mismatch")
    strict = {}
    for line in strict_path.read_text().splitlines():
        row = json.loads(line)
        if row["metadata"]["decision_type"] != "finish":
            continue
        reaction_id = str(row["metadata"]["reaction_id"])
        if reaction_id in strict or reaction_id not in full:
            raise ValueError(f"duplicate/unmatched strict ID: {reaction_id}")
        strict[reaction_id] = row
    if len(strict) != strict_source["reaction_denominator"]:
        raise ValueError("strict trace reaction denominator mismatch")
    counts: Counter[str] = Counter()
    cases = []
    for reaction_id in sorted(strict, key=int):
        reference = strict[reaction_id]
        full_row = full[reaction_id]
        strict_precursor = str(reference["expected_precursor"])
        full_reactants = str(full_row["reactants_unmapped"])
        full_structural = str(full_row["precursor_unmapped"])
        exact_visible = canonical_visible(strict_precursor) == canonical_visible(full_reactants)
        chemical_full = sides_equal(strict_precursor, full_reactants, ignore_maps=True)
        no_stereo = components_without_stereo(strict_precursor) == components_without_stereo(full_reactants)
        chemical_structural = sides_equal(strict_precursor, full_structural, ignore_maps=True)
        if exact_visible and not chemical_full or chemical_full and not no_stereo:
            raise ValueError(f"endpoint comparison hierarchy violated: {reaction_id}")
        case = {
            "reaction_id": reaction_id,
            "strict_full_precursor": strict_precursor,
            "full_endpoint_reactants": full_reactants,
            "full_endpoint_structural_precursor": full_structural,
            "exact_visible_full_reactants": exact_visible,
            "chemical_full_reactants": chemical_full,
            "same_components_without_stereo": no_stereo,
            "chemical_full_equals_structural": chemical_structural,
        }
        cases.append(case)
        counts["strict_reactions"] += 1
        for key in ("exact_visible_full_reactants", "chemical_full_reactants",
                    "same_components_without_stereo", "chemical_full_equals_structural"):
            counts[key] += int(case[key])
    return {
        "artifact_type": "system_one_strict_to_full_endpoint_output_contract_audit",
        "split": split,
        "strict_source": strict_source,
        "full_endpoint_sha256": full_declared["endpoint_sha256"],
        "full_endpoint_reactions": 3120,
        "counts": dict(counts),
    }, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, cases = audit(args.strict_dir, args.full_endpoint_dir, args.split)
    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case, separators=(",", ":")) + "\n")
    report["cases_sha256"] = sha256(cases_path)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
