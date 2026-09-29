#!/usr/bin/env python3
"""Build R1 products with independently recorded held-out precursor sets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


def mapped_bonds(smiles: str) -> set[tuple[int, int]]:
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid mapped molecule in R1 source")
    result: set[tuple[int, int]] = set()
    for bond in molecule.GetBonds():
        left, right = (atom.GetAtomMapNum() for atom in (bond.GetBeginAtom(), bond.GetEndAtom()))
        if left and right:
            result.add(tuple(sorted((left, right))))
    return result


def disconnection_signature(product: str, precursor: str) -> set[tuple[int, int]]:
    return mapped_bonds(product) - mapped_bonds(precursor)


def disconnection_stratum(signatures: list[set[tuple[int, int]]]) -> str:
    if not signatures or any(not signature for signature in signatures):
        return "unavailable"
    return "similar" if len({tuple(sorted(signature)) for signature in signatures}) == 1 else "different"


def structural_overlap(product: str, precursor_sets: list[str]) -> str:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target = Chem.MolFromSmiles(product)
    if target is None:
        raise ValueError("invalid R1 product")
    target_fp = fingerprint.GetFingerprint(target)
    scores = []
    for precursor in precursor_sets:
        components = [Chem.MolFromSmiles(item) for item in precursor.split(".")]
        if not components or any(component is None for component in components):
            raise ValueError("invalid R1 precursor component")
        scores.append(max(DataStructs.TanimotoSimilarity(
            target_fp, fingerprint.GetFingerprint(component)) for component in components))
    return "high" if min(scores) >= 0.6 else "low"


def build(source: Path, official_manifest: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"R1 cohort already frozen: {output}")
    manifest = json.loads(official_manifest.read_text())
    expected = manifest["splits"]["test"]["output_sha256"]
    source_hash = digest(source)
    if source_hash != expected:
        raise ValueError("R1 source differs from the official full-endpoint test manifest")
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(row["target_smiles"])
            precursor = product_key(row["structural_precursor"])
            groups[product][precursor].append({
                "record_id": str(row.get("source_id") or row["id"]),
                "mapped_product": row["target_smiles"],
                "mapped_precursor": row["structural_precursor"],
            })
    selected = []
    for product, references in sorted(groups.items()):
        if len(references) < 2:
            continue
        alternatives = []
        signatures = []
        for precursor, support in sorted(references.items()):
            support = sorted(support, key=lambda item: item["record_id"])
            signatures.append(disconnection_signature(support[0]["mapped_product"],
                                                        support[0]["mapped_precursor"]))
            alternatives.append({"precursor_smiles": precursor,
                                 "support_record_ids": [item["record_id"] for item in support]})
        if len({item["record_id"] for refs in references.values() for item in refs}) < 2:
            raise ValueError("multi-reference product lacks distinct held-out records")
        selected.append({
            "artifact_type": "r1_independent_recorded_alternatives_v1",
            "product_smiles": product, "source_dataset": "FlowER official full endpoint",
            "source_split": "test", "reference_count": len(alternatives),
            "references": alternatives,
            "strata": {
                "reference_count": "exactly_two" if len(alternatives) == 2 else "three_or_more",
                "disconnection": disconnection_stratum(signatures),
                "reaction_family": "unavailable",
                "structural_overlap": structural_overlap(
                    product, [item["precursor_smiles"] for item in alternatives]),
            },
        })
    output.mkdir(parents=True)
    cohort = output / "r1_multi_reference.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"artifact_type": "r1_frozen_cohort_manifest_v1",
              "source": str(source), "source_sha256": source_hash,
              "official_manifest": str(official_manifest),
              "official_manifest_sha256": digest(official_manifest),
              "cohort": str(cohort), "cohort_sha256": digest(cohort),
              "products": len(selected),
              "exactly_two": sum(item["reference_count"] == 2 for item in selected),
              "three_or_more": sum(item["reference_count"] >= 3 for item in selected),
              "stratum_counts": {name: {
                  value: sum(item["strata"][name] == value for item in selected)
                  for value in sorted({item["strata"][name] for item in selected})}
                  for name in ("reference_count", "disconnection", "reaction_family", "structural_overlap")},
              "limitation": "Recorded alternatives, not automatically established laboratory equivalence."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.official_manifest, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
