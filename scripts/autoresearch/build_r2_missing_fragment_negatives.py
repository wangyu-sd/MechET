#!/usr/bin/env python3
"""Build R2 closed-inventory atom-conservation negatives from held-out records.

Only an omission causing a heavy-element inventory deficit is accepted. The
resulting proposal cannot produce the product *from the stated precursor set*
under atom conservation. This does not claim that adding an unlisted reagent
could never produce the same product.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key, stable_hash


def omission_candidates(mapped_product: str, mapped_precursors: str) -> list[dict[str, Any]]:
    from rdkit import Chem

    product = Chem.MolFromSmiles(mapped_product)
    precursor = Chem.MolFromSmiles(mapped_precursors)
    if product is None or precursor is None:
        raise ValueError("invalid mapped FlowER reaction in R2 source")
    product_maps = [atom.GetAtomMapNum() for atom in product.GetAtoms()]
    if not product_maps or 0 in product_maps or len(product_maps) != len(set(product_maps)):
        raise ValueError("R2 atom-conservation witness requires unique mapped product atoms")
    fragments = Chem.GetMolFrags(precursor, asMols=True, sanitizeFrags=True)
    if len(fragments) < 2:
        return []
    fragment_maps = [{atom.GetAtomMapNum() for atom in fragment.GetAtoms()
                      if atom.GetAtomMapNum()} for fragment in fragments]
    if not set(product_maps) <= set().union(*fragment_maps):
        # The source row itself is not an atom-conserving positive for this proof.
        return []
    product_elements = Counter(atom.GetAtomicNum() for atom in product.GetAtoms()
                               if atom.GetAtomicNum() > 1)
    results = []
    for index, fragment in enumerate(fragments):
        remaining = [other for j, other in enumerate(fragments) if j != index]
        retained_maps = set().union(*(fragment_maps[j] for j in range(len(fragments)) if j != index))
        missing_maps = sorted(set(product_maps) - retained_maps)
        if not missing_maps:
            continue
        retained_elements = Counter(atom.GetAtomicNum() for other in remaining
                                    for atom in other.GetAtoms() if atom.GetAtomicNum() > 1)
        missing_elements = {Chem.GetPeriodicTable().GetElementSymbol(number): quantity - retained_elements[number]
                            for number, quantity in sorted(product_elements.items())
                            if quantity > retained_elements[number]}
        if not missing_elements:
            # Map labels alone do not prove an *unmapped* proposal is impossible:
            # a different atom correspondence may use an equivalent atom.
            continue
        candidate = product_key(".".join(Chem.MolToSmiles(other) for other in remaining))
        if not candidate:
            continue
        results.append({
            "proposed_precursors": candidate,
            "removed_fragment_mapped": Chem.MolToSmiles(fragment),
            "missing_product_atom_maps": missing_maps,
            "missing_element_counts": missing_elements,
        })
    return results


def build(source: Path, official_manifest: Path, positives: Path,
          positive_manifest: Path, output: Path, *, seed: int = 17,
          count: int = 50) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R2 negative cohort already frozen: {output}")
    if count <= 0 or count % 2:
        raise ValueError("R2 missing-fragment count must be positive and even")
    official = json.loads(official_manifest.read_text())
    source_hash = digest(source)
    if source_hash != official["splits"]["test"]["output_sha256"]:
        raise ValueError("R2 source differs from official full-endpoint test manifest")
    positive_meta = json.loads(positive_manifest.read_text())
    positive_hash = digest(positives)
    if positive_hash != positive_meta["cohort_sha256"]:
        raise ValueError("R2 positive source hash drifted")
    positive_rows = [json.loads(line) for line in positives.open(encoding="utf-8")]
    if len(positive_rows) != positive_meta["positive_proposals"]:
        raise ValueError("R2 positive count differs from frozen manifest")
    by_id: dict[str, dict[str, Any]] = {}
    for positive in positive_rows:
        # One independently recorded witness per selected proposal is enough.
        # Retaining every supporting duplicate would overweight that product.
        record_id = min(positive["private_label"]["proposal_record_ids"])
        if record_id in by_id:
            raise ValueError("duplicate R2 positive source record ID")
        by_id[record_id] = positive
    matched: dict[str, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            record_id = str(row.get("source_id") or row["id"])
            if record_id in by_id:
                matched[record_id] = row
    if set(matched) != set(by_id):
        raise ValueError("R2 positive source records are missing from official test")

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_id, row in matched.items():
        positive = by_id[record_id]
        if product_key(row["target_smiles"]) != positive["product_smiles"]:
            raise ValueError("R2 positive product/source record mismatch")
        if product_key(row["structural_precursor"]) != positive["model_input"]["proposed_precursors"]:
            raise ValueError("R2 positive precursor/source record mismatch")
        choices = omission_candidates(row["target_smiles"], row["structural_precursor"])
        if not choices:
            continue
        chosen = min(choices, key=lambda item: (
            stable_hash(seed, record_id, item["proposed_precursors"]),
            item["proposed_precursors"]))
        eligible[positive["strata"]["positive_type"]].append({
            "artifact_type": "r2_closed_inventory_missing_fragment_negative_v2",
            "proposal_id": f"r2:missing_fragment:{record_id}",
            "source_dataset": "FlowER official full endpoint",
            "source_split": "test",
            "product_smiles": positive["product_smiles"],
            "model_input": {"product_smiles": positive["product_smiles"],
                            "proposed_precursors": chosen["proposed_precursors"]},
            "private_label": {
                "negative_class": "missing_necessary_fragment",
                "evidence_kind": "product_element_inventory_deficit_after_fragment_omission",
                "scope": "closed_stated_precursor_inventory_atom_conservation",
                "source_record_id": record_id,
                "original_precursors": positive["model_input"]["proposed_precursors"],
                "removed_fragment_mapped": chosen["removed_fragment_mapped"],
                "missing_product_atom_maps": chosen["missing_product_atom_maps"],
                "missing_element_counts": chosen["missing_element_counts"],
            },
            "strata": {"negative_class": "missing_necessary_fragment",
                       "positive_source_type": positive["strata"]["positive_type"],
                       "heavy_atom_quartile": positive["strata"]["heavy_atom_quartile"]},
        })
    selected = []
    for kind in ("recorded_precursor", "documented_alternative"):
        ordered = sorted(eligible[kind], key=lambda row: (
            stable_hash(seed, kind, row["proposal_id"]), row["proposal_id"]))
        if len(ordered) < count // 2:
            raise ValueError(f"R2 {kind} missing-fragment cell underfilled: {len(ordered)}")
        selected.extend(ordered[:count // 2])
    selected.sort(key=lambda row: row["proposal_id"])
    if len({row["product_smiles"] for row in selected}) != len(selected):
        raise ValueError("R2 negative cohort has duplicate products")
    output.mkdir(parents=True)
    cohort = output / "r2_missing_fragment_negatives.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "artifact_type": "r2_missing_fragment_negative_manifest_v2",
        "source": str(source), "source_sha256": source_hash,
        "official_manifest": str(official_manifest),
        "official_manifest_sha256": digest(official_manifest),
        "positives": str(positives), "positives_sha256": positive_hash,
        "positive_manifest_sha256": digest(positive_manifest),
        "cohort": str(cohort), "cohort_sha256": digest(cohort),
        "seed": seed, "negative_class": "missing_necessary_fragment",
        "negative_proposals": len(selected),
        "eligible_by_positive_type": {name: len(eligible[name]) for name in sorted(eligible)},
        "selected_by_positive_type": {name: sum(row["strata"]["positive_source_type"] == name
                                               for row in selected) for name in sorted(eligible)},
        "claim_boundary": "Unmapped elemental-inventory deficit for the stated precursor set; not an open-world impossibility claim or complete R2 evaluator.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r2_missing_fragment_status_v2",
        "cohort_sha256": report["cohort_sha256"],
        "evidence_audited": True,
        "evaluation_allowed": False,
        "training_allowed": False,
        "reason": "One formal contradiction class only; R2 requires all hard-negative classes and an executor-valid class",
    }, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.official_manifest,
                           args.positives, args.positive_manifest,
                           args.output, seed=args.seed, count=args.count),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
