"""Structural overlap audits for composition-OOD mechanistic splits."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from statistics import median
from typing import Any, Iterable, Mapping

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .proof_to_trace import ProofTracePlan, execution_primitive_signatures


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _clear_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    value = Chem.Mol(mol)
    for atom in value.GetAtoms():
        atom.SetAtomMapNum(0)
    return value


def canonical_unmapped_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ValueError("INVALID_STRUCTURAL_SMILES")
    return Chem.MolToSmiles(_clear_atom_maps(mol), canonical=True, isomericSmiles=True)


def _largest_fragment(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ValueError("INVALID_STRUCTURAL_SMILES")
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError("EMPTY_STRUCTURAL_SMILES")
    return max(fragments, key=lambda item: item.GetNumHeavyAtoms())


def murcko_scaffold_key(smiles: str) -> str:
    fragment = _clear_atom_maps(_largest_fragment(smiles))
    scaffold = MurckoScaffold.GetScaffoldForMol(fragment)
    if scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=False)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def _fingerprint(smiles: str):
    return _MORGAN.GetFingerprint(_clear_atom_maps(_largest_fragment(smiles)))


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row.get("metadata") or {})


def target_smiles(row: Mapping[str, Any]) -> str:
    return str(row.get("target_smiles") or _metadata(row).get("target_smiles") or "")


def structural_precursor_smiles(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    for value in (
        row.get("structural_precursor"),
        row.get("expected_structural_precursor"),
        metadata.get("structural_precursor"),
        metadata.get("expected_structural_precursor"),
        metadata.get("structural_endpoint"),
        row.get("expected_precursor"),
    ):
        if value:
            return str(value)
    raise ValueError("STRUCTURAL_PRECURSOR_MISSING")


def family_label(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    for key in (
        "reaction_family",
        "mechanism_family",
        "mechanism_class",
        "reaction_class",
        "name_reaction",
    ):
        value = metadata.get(key, row.get(key))
        if value not in (None, ""):
            return str(value)
    return ""


def _trace_plan(row: Mapping[str, Any]) -> ProofTracePlan:
    metadata = _metadata(row)
    value = metadata.get("trace_plan") or row.get("trace_plan")
    if not isinstance(value, Mapping):
        raise ValueError("TRACE_PLAN_MISSING")
    return ProofTracePlan.from_dict(value)


def reaction_center_context_signature(row: Mapping[str, Any]) -> str:
    """Hash a map-independent local target context plus move primitives."""

    plan = _trace_plan(row)
    mol = Chem.MolFromSmiles(plan.target_smiles)
    if mol is None:
        raise ValueError("REACTION_CENTER_TARGET_INVALID")
    map_to_index = {
        int(atom.GetAtomMapNum()): atom.GetIdx()
        for atom in mol.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    involved_maps: set[int] = set()
    move_topology: list[dict[str, Any]] = []
    for step in plan.steps:
        for move in step.moves:
            source = dict(move.get("source") or {})
            sink = dict(move.get("sink") or {})
            source_maps = [int(item) for item in source.get("atoms") or []]
            sink_maps = [int(item) for item in sink.get("atoms") or []]
            involved_maps.update(source_maps)
            involved_maps.update(sink_maps)
            move_topology.append(
                {
                    "source_kind": str(source.get("kind") or ""),
                    "sink_kind": str(sink.get("kind") or ""),
                    "source_size": len(source_maps),
                    "sink_size": len(sink_maps),
                    "electrons": int(move.get("electrons", 2)),
                }
            )
    selected_indices = {
        map_to_index[item] for item in involved_maps if item in map_to_index
    }
    for index in list(selected_indices):
        selected_indices.update(
            neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
        )
    if not selected_indices:
        raise ValueError("REACTION_CENTER_ATOMS_MISSING")
    local = Chem.MolFragmentToSmiles(
        _clear_atom_maps(mol),
        atomsToUse=sorted(selected_indices),
        canonical=True,
        isomericSmiles=True,
        allBondsExplicit=True,
    )
    payload = {
        "local_target_context": local,
        "move_topology": move_topology,
        "execution_primitives": list(execution_primitive_signatures(plan)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def row_overlap_features(row: Mapping[str, Any]) -> dict[str, Any]:
    product = canonical_unmapped_smiles(target_smiles(row))
    precursor = canonical_unmapped_smiles(structural_precursor_smiles(row))
    return {
        "product_key": product,
        "precursor_key": precursor,
        "reaction_key": f"{product}>>{precursor}",
        "scaffold_key": murcko_scaffold_key(product),
        "reaction_center_key": reaction_center_context_signature(row),
        "family": family_label(row),
        "product_fingerprint": _fingerprint(product),
    }


def audit_structural_overlap(
    train_rows: Iterable[Mapping[str, Any]],
    heldout_rows: Iterable[Mapping[str, Any]],
    *,
    similarity_threshold: float = 0.90,
    compute_similarity: bool = True,
    max_near_duplicate_rate: float = 0.0,
) -> dict[str, Any]:
    """Audit one held-out split against the training structural universe."""

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1]")
    train_values = list(train_rows)
    heldout_values = list(heldout_rows)
    train_features = [row_overlap_features(row) for row in train_values]
    heldout_features = [row_overlap_features(row) for row in heldout_values]

    product_keys = {item["product_key"] for item in train_features}
    precursor_keys = {item["precursor_key"] for item in train_features}
    reaction_keys = {item["reaction_key"] for item in train_features}
    scaffold_keys = {item["scaffold_key"] for item in train_features}
    center_keys = {item["reaction_center_key"] for item in train_features}
    family_keys = {item["family"] for item in train_features if item["family"]}
    train_fingerprints = [item["product_fingerprint"] for item in train_features]

    annotations: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    similarities: list[float] = []
    family_available = 0
    for index, (row, feature) in enumerate(zip(heldout_values, heldout_features)):
        product_seen = feature["product_key"] in product_keys
        precursor_seen = feature["precursor_key"] in precursor_keys
        reaction_seen = feature["reaction_key"] in reaction_keys
        scaffold_seen = feature["scaffold_key"] in scaffold_keys
        center_seen = feature["reaction_center_key"] in center_keys
        family = str(feature["family"] or "")
        family_seen = family in family_keys if family else None
        family_available += int(bool(family))

        maximum_similarity = None
        near_duplicate = False
        if compute_similarity and train_fingerprints:
            values = DataStructs.BulkTanimotoSimilarity(
                feature["product_fingerprint"], train_fingerprints
            )
            maximum_similarity = max(values, default=0.0)
            similarities.append(float(maximum_similarity))
            near_duplicate = maximum_similarity >= similarity_threshold

        counts.update(
            {
                "exact_product_seen": int(product_seen),
                "exact_precursor_seen": int(precursor_seen),
                "exact_reaction_seen": int(reaction_seen),
                "scaffold_seen": int(scaffold_seen),
                "reaction_center_seen": int(center_seen),
                "family_seen": int(family_seen is True),
                "family_unseen": int(family_seen is False),
                "near_duplicate": int(near_duplicate),
                "composition_ood_scaffold_seen": int(scaffold_seen),
                "composition_ood_scaffold_unseen": int(not scaffold_seen),
                "composition_ood_center_seen": int(center_seen),
                "composition_ood_center_unseen": int(not center_seen),
            }
        )
        identifier = str(row.get("id") or f"row_{index}")
        annotations[identifier] = {
            "exact_product_seen_in_train": product_seen,
            "exact_structural_precursor_seen_in_train": precursor_seen,
            "exact_reaction_seen_in_train": reaction_seen,
            "murcko_scaffold_seen_in_train": scaffold_seen,
            "reaction_center_context_seen_in_train": center_seen,
            "family_label": family or None,
            "family_seen_in_train": family_seen,
            "maximum_product_tanimoto_to_train": maximum_similarity,
            "near_duplicate_at_threshold": near_duplicate,
            "similarity_threshold": similarity_threshold if compute_similarity else None,
        }

    n = len(heldout_values)
    denominator = max(n, 1)
    near_duplicate_rate = counts["near_duplicate"] / denominator
    report = {
        "n_train": len(train_values),
        "n_heldout": n,
        "similarity_metric": (
            "Morgan radius=2, 2048-bit Tanimoto on the largest product fragment"
            if compute_similarity
            else "disabled"
        ),
        "similarity_threshold": similarity_threshold if compute_similarity else None,
        "exact_product_overlap_count": counts["exact_product_seen"],
        "exact_product_overlap_rate": counts["exact_product_seen"] / denominator,
        "exact_structural_precursor_overlap_count": counts["exact_precursor_seen"],
        "exact_structural_precursor_overlap_rate": counts["exact_precursor_seen"]
        / denominator,
        "exact_reaction_overlap_count": counts["exact_reaction_seen"],
        "exact_reaction_overlap_rate": counts["exact_reaction_seen"] / denominator,
        "murcko_scaffold_seen_count": counts["scaffold_seen"],
        "murcko_scaffold_seen_rate": counts["scaffold_seen"] / denominator,
        "reaction_center_context_seen_count": counts["reaction_center_seen"],
        "reaction_center_context_seen_rate": counts["reaction_center_seen"]
        / denominator,
        "family_labels_available": family_available,
        "family_seen_count": counts["family_seen"],
        "family_unseen_count": counts["family_unseen"],
        "near_duplicate_count": counts["near_duplicate"],
        "near_duplicate_rate": near_duplicate_rate,
        "maximum_product_tanimoto_summary": (
            {
                "minimum": min(similarities),
                "median": median(similarities),
                "maximum": max(similarities),
            }
            if similarities
            else None
        ),
        "composition_ood_strata": {
            "scaffold_seen": counts["composition_ood_scaffold_seen"],
            "scaffold_unseen": counts["composition_ood_scaffold_unseen"],
            "reaction_center_seen": counts["composition_ood_center_seen"],
            "reaction_center_unseen": counts["composition_ood_center_unseen"],
            "family_seen": counts["family_seen"],
            "family_unseen": counts["family_unseen"],
        },
        "claim_gate": {
            "nonempty_heldout": n > 0,
            "zero_exact_reaction_overlap": counts["exact_reaction_seen"] == 0,
            "near_duplicate_rate_at_or_below_tolerance": near_duplicate_rate
            <= max_near_duplicate_rate,
            "max_near_duplicate_rate": max_near_duplicate_rate,
        },
        "row_annotations": annotations,
    }
    return report


def annotate_rows_with_overlap(
    rows: Iterable[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        value = dict(row)
        metadata = dict(value.get("metadata") or {})
        identifier = str(value.get("id") or f"row_{index}")
        metadata["mechcomp_structural_overlap"] = dict(
            annotations.get(identifier) or {}
        )
        value["metadata"] = metadata
        output.append(value)
    return output
