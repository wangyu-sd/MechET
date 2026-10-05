#!/usr/bin/env python3
"""Compare PR81 strict-trace targets with the full mech-USPTO endpoint inputs.

The strict natural-language trajectory target is the executor's final molecular
mixture. The independent 3,120-reaction endpoint benchmark asks for a selected
principal product. This audit prevents a mixture-start rollout being reported
as product-only benchmark performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.eval_system_one_product_start_pilot import canonical_visible
from scripts.train_system_one_electron_flow import verify_source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_counter(smiles: str, *, isomeric: bool = True) -> Counter[str]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError("invalid molecular mixture in input gap audit")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Counter(Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=isomeric
    ).split("."))


def audit(strict_rows: dict[str, dict], full_rows: dict[str, dict]) -> dict:
    if not set(strict_rows) <= set(full_rows):
        raise ValueError("strict trace reaction IDs absent from full endpoint split")
    counts: Counter[str] = Counter()
    residual_batches: Counter[tuple[str, ...]] = Counter()
    examples = []
    for reaction_id in sorted(strict_rows, key=lambda value: int(value)):
        strict, full = strict_rows[reaction_id], full_rows[reaction_id]
        strict_target = canonical_visible(strict["target_smiles"])
        principal_product = canonical_visible(full["product_unmapped"])
        strict_precursor = canonical_visible(strict["expected_precursor"])
        full_reactants = canonical_visible(full["reactants_unmapped"])
        structural_precursor = canonical_visible(full["precursor_unmapped"])
        counts["strict_reactions"] += 1
        counts["target_exact"] += int(strict_target == principal_product)
        counts["strict_target_more_components"] += int(
            sum(component_counter(strict_target).values())
            > sum(component_counter(principal_product).values())
        )
        counts["principal_product_components_contained_in_strict_target"] += int(
            not (component_counter(principal_product) - component_counter(strict_target))
        )
        strict_nostereo = component_counter(strict_target, isomeric=False)
        product_nostereo = component_counter(principal_product, isomeric=False)
        no_stereo_contained = not (product_nostereo - strict_nostereo)
        counts["principal_product_components_contained_without_stereo"] += int(
            no_stereo_contained
        )
        if no_stereo_contained:
            batch = tuple(sorted((strict_nostereo - product_nostereo).elements()))
            residual_batches[batch] += 1
            counts["nonempty_residual_context_batch_without_stereo"] += int(bool(batch))
        counts["strict_precursor_equals_full_reactants"] += int(
            strict_precursor == full_reactants
        )
        counts["strict_precursor_equals_full_structural"] += int(
            strict_precursor == structural_precursor
        )
        if strict_target != principal_product and len(examples) < 5:
            examples.append({
                "reaction_id": reaction_id,
                "strict_final_mixture_target": strict_target,
                "full_endpoint_principal_product": principal_product,
            })
    return {
        "counts": dict(sorted(counts.items())),
        "residual_context_batches_without_stereo": [
            {"components": list(batch), "reactions": count}
            for batch, count in residual_batches.most_common()
        ],
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--cases", type=Path,
                        help="optional completed PR81 rollout cases.jsonl to bind its input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    strict_path = args.strict_dir / f"{args.split}.jsonl"
    full_path = args.full_endpoint_dir / f"{args.split}.jsonl"
    strict_source = verify_source(strict_path)
    manifest = json.loads((args.full_endpoint_dir / "manifest.json").read_text())
    declared = manifest["splits"][args.split]
    expected_full_rows = 24959 if args.split == "train" else 3120
    if (manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or manifest["executor_filtering"] is not False
            or declared["rows"] != expected_full_rows
            or sha256(full_path) != declared["endpoint_sha256"]):
        raise ValueError("full endpoint benchmark manifest/hash/denominator mismatch")
    strict_rows = {}
    for line in strict_path.read_text().splitlines():
        row = json.loads(line)
        if row["metadata"]["decision_type"] != "finish":
            continue
        reaction_id = str(row["metadata"]["reaction_id"])
        if reaction_id in strict_rows:
            raise ValueError(f"duplicate strict terminal: {reaction_id}")
        strict_rows[reaction_id] = row
    if len(strict_rows) != strict_source["reaction_denominator"]:
        raise ValueError("strict terminal reaction count mismatch")
    full_rows = {}
    for line in full_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in full_rows:
            raise ValueError(f"duplicate full endpoint reaction: {reaction_id}")
        full_rows[reaction_id] = row
    if len(full_rows) != expected_full_rows:
        raise ValueError("full endpoint reaction count mismatch")
    result = {
        "artifact_type": "system_one_strict_mixture_vs_full_principal_product_audit",
        "split": args.split,
        "strict_source_sha256": strict_source["sha256"],
        "strict_reaction_denominator": strict_source["reaction_denominator"],
        "full_endpoint_sha256": declared["endpoint_sha256"],
        "full_endpoint_denominator": expected_full_rows,
        **audit(strict_rows, full_rows),
    }
    if args.cases:
        seen = set()
        for line in args.cases.read_text().splitlines():
            row = json.loads(line)
            reaction_id = str(row["id"])
            if reaction_id in seen or reaction_id not in strict_rows:
                raise ValueError(f"invalid rollout case ID: {reaction_id}")
            if canonical_visible(row["target"]) != canonical_visible(
                strict_rows[reaction_id]["target_smiles"]
            ):
                raise ValueError(f"rollout target differs from strict final mixture: {reaction_id}")
            seen.add(reaction_id)
        if len(seen) != len(strict_rows):
            raise ValueError("rollout cases do not cover strict reaction denominator")
        result["rollout_cases_sha256"] = sha256(args.cases)
        result["rollout_targets_equal_strict_final_mixture"] = len(seen)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"examples", "residual_context_batches_without_stereo"}}, indent=2))


if __name__ == "__main__":
    main()
