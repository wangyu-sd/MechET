#!/usr/bin/env python3
"""Classify first product-only remap replay divergences without changing the executor."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rdkit import Chem

from mechet.forward_expert import ElectronMove
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.analyze_reliable_product_start import _sha256
from scripts.earho_v2_protocol import decision_action, replay_reference
from scripts.run_natural_language_value_search import (
    Action, Node, execute, mapped_atom_numbers, product_only_private_state,
    visible,
)


def heavy_atom_skeleton(smiles: str) -> str:
    """Return a connectivity-only heavy-atom graph, ignoring bond order/charge."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError("invalid successor SMILES")
    graph = Chem.RWMol()
    retained = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        replacement = Chem.Atom(atom.GetAtomicNum())
        replacement.SetIsotope(atom.GetIsotope())
        replacement.SetNoImplicit(True)
        retained[atom.GetIdx()] = graph.AddAtom(replacement)
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in retained and right in retained:
            graph.AddBond(retained[left], retained[right], Chem.BondType.SINGLE)
    return Chem.MolToSmiles(graph.GetMol(), canonical=True)


def source_bond_context(state: str, arguments: dict) -> list[dict]:
    """Read each electron-source bond before and after the executor's Kekulization."""
    moves = compile_event_arguments(state, arguments)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    original = Chem.MolFromSmiles(state, params)
    if original is None:
        raise ValueError("invalid pre-event state")
    kekule = Chem.Mol(original)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:
        pass
    indices = {atom.GetAtomMapNum(): atom.GetIdx() for atom in original.GetAtoms()}
    output = []
    for raw in moves:
        if raw.get("mode") == "BE_DELTA":
            continue
        move = ElectronMove.parse(raw)
        if move.source.kind != "BOND":
            continue
        left, right = (indices[value] for value in move.source.atoms)
        bond = original.GetBondBetweenAtoms(left, right)
        selected = kekule.GetBondBetweenAtoms(left, right)
        output.append({
            "source_is_aromatic": bool(bond.GetIsAromatic()) if bond else False,
            "source_kekule_order": (
                int(round(selected.GetBondTypeAsDouble())) if selected else None
            ),
        })
    return output


def classify_first_divergence(source: dict, decisions: list[dict]) -> dict:
    state = product_only_private_state(str(source["target_smiles"]))
    original_state = str(source["target_smiles"])

    def root(mapped: str) -> Node:
        return Node(
            target=visible(mapped), state=mapped,
            next_map=max(mapped_atom_numbers(mapped), default=0) + 1,
            visited={visible(mapped)},
        )

    node, original_node = root(state), root(original_state)
    for index, row in enumerate(decisions):
        if int(row["metadata"]["decision_index"]) != index:
            raise ValueError("noncontiguous decision indices")
        name, arguments, result = decision_action(row)
        original_child, original_error = execute(
            original_node, Action(name, arguments, "reference", 0.0, 1), max_imports=64,
        )
        if original_child is None:
            raise ValueError(f"original-map reference replay failed: {original_error}")
        source_bonds_original = (
            source_bond_context(original_node.state, arguments)
            if name == "apply_electron_flow" else []
        )
        source_bonds_remapped = (
            source_bond_context(node.state, arguments)
            if name == "apply_electron_flow" else []
        )
        child, error = execute(
            node, Action(name, arguments, "reference", 0.0, 1), max_imports=64,
        )
        if child is None:
            return {"decision_index": index, "decision_type": name,
                    "class": "execution_failure", "error": error}
        expected = str(result.get("derived_precursor") or result.get("current_state") or "")
        if visible(original_child.state) != expected:
            raise ValueError("original-map successor disagrees with frozen reference")
        actual = visible(child.state)
        if actual != expected:
            expected_skeleton = heavy_atom_skeleton(expected)
            actual_skeleton = heavy_atom_skeleton(actual)
            return {
                "decision_index": index, "decision_type": name,
                "class": (
                    "same_heavy_atom_connectivity_different_state"
                    if expected_skeleton == actual_skeleton
                    else "different_heavy_atom_connectivity"
                ),
                "reference_successor": expected,
                "remapped_successor": actual,
                "reference_skeleton": expected_skeleton,
                "remapped_skeleton": actual_skeleton,
                "source_bonds_original": source_bonds_original,
                "source_bonds_remapped": source_bonds_remapped,
            }
        node, original_node = child, original_child
    raise ValueError("audit-listed failure replayed without a divergence")


def audit_all_original_source_orders(
    source: Path, decisions: Path, *, compact_history: bool,
) -> dict:
    """Count source-bond Kekulé orders over every frozen original-map trace."""
    by_source = defaultdict(list)
    with decisions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            by_source[str(row["source_id"])].append(row)
    counts: Counter[str] = Counter()
    single_source_ids: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            reaction = json.loads(line)
            source_id = str(reaction["source_id"])
            rows = sorted(
                by_source[source_id],
                key=lambda row: int(row["metadata"]["decision_index"]),
            )
            reference = replay_reference(
                reaction, rows, compact_history=compact_history,
            )
            counts["reactions"] += 1
            for index, row in enumerate(rows):
                name, arguments, _ = decision_action(row)
                if name != "apply_electron_flow":
                    continue
                for item in source_bond_context(reference.nodes[index].state, arguments):
                    if not item["source_is_aromatic"]:
                        continue
                    counts[f"aromatic_source_order_{item['source_kekule_order']}"] += 1
                    if item["source_kekule_order"] == 1:
                        single_source_ids.add(source_id)
    if counts["reactions"] != len(by_source):
        raise ValueError("source/decision reaction denominator mismatch")
    return {
        "counts": dict(counts),
        "single_source_ids": sorted(single_source_ids),
        "interpretation": (
            "aromatic source-bond occurrences in original-map reference events; "
            "a universal aromatic-double rule would change valid single-bond cases"
        ),
    }


def classify(audit_path: Path, *, all_source_orders: bool = False) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("artifact_type") != "reliable_mechet_product_only_private_mapping_audit_v1":
        raise ValueError("unrecognized parity audit")
    source, decisions = Path(audit["source"]), Path(audit["decisions"])
    if _sha256(source) != audit["source_sha256"] or _sha256(decisions) != audit["decisions_sha256"]:
        raise ValueError("audit source/decision SHA mismatch")
    failures = list(audit.get("failures") or [])
    wanted = {str(item["source_id"]) for item in failures}
    if len(wanted) != len(failures) or any(
        item.get("kind") != "product_only_remap_replay_failed" for item in failures
    ):
        raise ValueError("audit failure IDs or kinds are invalid")
    sources = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row["source_id"])
            if key in wanted:
                if key in sources:
                    raise ValueError("duplicate source ID")
                sources[key] = row
    by_source = defaultdict(list)
    with decisions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row["source_id"])
            if key in wanted:
                by_source[key].append(row)
    if set(sources) != wanted or set(by_source) != wanted:
        raise ValueError("audit failure IDs lack source or decision rows")
    cases = []
    for source_id in sorted(wanted):
        rows = sorted(by_source[source_id], key=lambda row: int(row["metadata"]["decision_index"]))
        cases.append({"source_id": source_id, **classify_first_divergence(sources[source_id], rows)})
    report = {
        "artifact_type": "reliable_mechet_mapping_failure_connectivity_v1",
        "audit": str(audit_path), "audit_sha256": _sha256(audit_path),
        "source_sha256": audit["source_sha256"],
        "decisions_sha256": audit["decisions_sha256"],
        "denominator": len(cases),
        "class_counts": dict(Counter(item["class"] for item in cases)),
        "first_decision_counts": dict(Counter(item["decision_index"] for item in cases)),
        "aromatic_source_kekule_order_changed": sum(
            any(
                left["source_is_aromatic"] and right["source_is_aromatic"]
                and left["source_kekule_order"] != right["source_kekule_order"]
                for left, right in zip(
                    item.get("source_bonds_original", []),
                    item.get("source_bonds_remapped", []), strict=True,
                )
            ) for item in cases
        ),
        "scope": (
            "only the audit-listed product-only remap reference failures; "
            "same heavy-atom topology does not prove chemical or resonance equivalence"
        ),
        "cases": cases,
    }
    if all_source_orders:
        report["all_original_source_orders"] = audit_all_original_source_orders(
            source, decisions,
            compact_history=audit.get("observation_contract") == "compressed_history",
        )
        if report["all_original_source_orders"]["counts"]["reactions"] != audit["n_reactions"]:
            raise ValueError("full source-order audit denominator differs from mapping audit")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-source-orders", action="store_true")
    args = parser.parse_args()
    report = classify(args.audit, all_source_orders=args.all_source_orders)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "denominator": report["denominator"],
        "class_counts": report["class_counts"],
        "first_decision_counts": report["first_decision_counts"],
        "aromatic_source_kekule_order_changed": report["aromatic_source_kekule_order_changed"],
        "all_original_source_orders": report.get("all_original_source_orders", {}).get("counts"),
        "output": str(args.output),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
