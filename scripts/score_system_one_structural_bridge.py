#!/usr/bin/env python3
"""Replay frozen bridge actions with private atom provenance for structural scoring.

The model never sees these private maps. Product-origin atoms are marked at
reset, new context/IMPORT atoms are marked as auxiliary, and accepted actions
are replayed without a reference trajectory. Full-endpoint structural
precursors enter only after the predicted trajectory has ended.
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

from mechet.electron_pointer import parse_pointer_observation
from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.in_place_grounded_flow import deterministic_unmapped_state
from mechet.system_one_replay import execute_pair_indices, pair_indices_to_arguments
from mechet.trajectory_history import TrajectoryHistory
from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.audit_system_one_observation_parity import mapped_from_visible, runtime_prompt
from scripts.eval_system_one_product_start_pilot import canonical_visible


def parse_with_hydrogens(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError(f"invalid SMILES for provenance replay: {smiles[:120]}")
    return molecule


def initialize_provenance(product: str, mixture: str) -> tuple[str, str, int]:
    """Return private mapped mixture/product with uniquely tagged product atoms."""
    molecule = parse_with_hydrogens(mixture)
    fragments = Chem.GetMolFrags(molecule)
    fragment_molecules = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    product_key = canonical_visible(product)
    matches = [index for index, part in enumerate(fragment_molecules)
               if canonical_visible(Chem.MolToSmiles(part)) == product_key]
    if len(matches) != 1:
        raise ValueError(f"principal product occurs {len(matches)} times in inferred mixture")
    product_atoms = fragments[matches[0]]
    for map_id, atom_index in enumerate(product_atoms, 1):
        molecule.GetAtomWithIdx(atom_index).SetAtomMapNum(map_id)
    next_map = len(product_atoms) + 1
    for index, atom in enumerate(molecule.GetAtoms()):
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(next_map)
            next_map += 1
    product_mol = Chem.Mol(fragment_molecules[matches[0]])
    for map_id, atom in enumerate(product_mol.GetAtoms(), 1):
        atom.SetAtomMapNum(map_id)
    mapped = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)
    target = Chem.MolToSmiles(product_mol, canonical=False, isomericSmiles=True)
    if canonical_visible(deterministic_unmapped_state(mapped).text) != canonical_visible(mixture):
        raise ValueError("initial private maps changed input chemistry")
    return mapped, target, next_map


def append_private_import(mapped: str, batch: list[list], next_map: int) -> tuple[str, int]:
    combined = parse_with_hydrogens(mapped)
    for smiles, count in batch:
        fragment = parse_with_hydrogens(str(smiles))
        if int(count) < 1:
            raise ValueError("invalid import multiplicity")
        for _ in range(int(count)):
            fresh = Chem.Mol(fragment)
            for atom in fresh.GetAtoms():
                atom.SetAtomMapNum(next_map)
                next_map += 1
            combined = Chem.CombineMols(combined, fresh)
    return Chem.MolToSmiles(combined, canonical=False, isomericSmiles=True), next_map


def replay_provenance(case: dict) -> tuple[str, str, str | None]:
    """Return predicted full/structural SMILES and failure code, if any."""
    mapped, mapped_product, next_map = initialize_provenance(
        str(case["principal_product_input"]), str(case["target"])
    )
    history = TrajectoryHistory()
    for step, action in enumerate(case["actions"]):
        visible = deterministic_unmapped_state(mapped).text
        if canonical_visible(visible) != canonical_visible(str(action["state_before"])):
            raise ValueError(f"step {step}: private replay before-state mismatch")
        kind = str(action["action"])
        if kind == "finish_trace":
            if action["accepted"]:
                history = history.accept(kind, {}, {"ok": True, "code": "PASS"})
            continue
        if kind == "import_fragments":
            batch = action["batch"]
            mapped, next_map = append_private_import(mapped, batch, next_map)
            successor = deterministic_unmapped_state(mapped).text
            if canonical_visible(successor) != canonical_visible(str(action["state_after"])):
                raise ValueError(f"step {step}: private import successor mismatch")
            arguments = {"fragments": [
                {"smiles": smiles, "count": count, "purpose": "electron_participant"}
                for smiles, count in batch
            ]}
            history = history.accept(kind, arguments, {
                "ok": True, "code": "PASS", "current_state": successor,
                "imported_fragments": sum(int(count) for _smiles, count in batch),
            })
            continue
        if kind != "apply_electron_flow":
            raise ValueError(f"step {step}: unsupported action {kind}")
        prompt = runtime_prompt(str(case["target"]), visible, history)
        observation = parse_pointer_observation(prompt, row_id=f"{case['id']}::provenance_{step}")
        local_mapped = mapped_from_visible(visible)
        selected = list(action["selected_pairs"])
        result = execute_pair_indices(local_mapped, observation, selected)
        if bool(result.get("ok")) != bool(action["execute_ok"]):
            raise ValueError(f"step {step}: executor result differs from frozen rollout")
        if not result.get("ok"):
            if case["completed"] or case["terminal"] != "ELECTRON_EXECUTION_FAILED":
                raise ValueError(f"step {step}: frozen terminal differs from replay failure")
            return visible, "", "ELECTRON_EXECUTION_FAILED"
        origin_maps = deterministic_unmapped_state(mapped).atom_maps
        successor_mol = parse_with_hydrogens(str(result["state_smiles"]))
        for atom in successor_mol.GetAtoms():
            local_map = int(atom.GetAtomMapNum())
            if not 1 <= local_map <= len(origin_maps):
                raise ValueError(f"step {step}: untracked electron successor atom")
            atom.SetAtomMapNum(origin_maps[local_map - 1])
        mapped = Chem.MolToSmiles(successor_mol, canonical=False, isomericSmiles=True)
        successor = deterministic_unmapped_state(mapped).text
        if canonical_visible(successor) != canonical_visible(str(action["state_after"])):
            raise ValueError(f"step {step}: private electron successor mismatch")
        history = history.accept(kind, pair_indices_to_arguments(observation, selected), {
            "ok": True, "code": result.get("code") or "PASS", "current_state": successor,
        })
    final_visible = deterministic_unmapped_state(mapped).text
    if case["completed"]:
        if canonical_visible(final_visible) != canonical_visible(str(case["predicted_precursor"])):
            raise ValueError("private endpoint differs from frozen prediction")
        structural = split_precursor_endpoints(mapped, mapped_product).structural
        return final_visible, structural, None
    return final_visible, "", str(case["terminal"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bridge_report = json.loads((args.bridge_dir / "report.json").read_text())
    bridge_cases = args.bridge_dir / "cases.jsonl"
    if (bridge_report["artifact_type"] != "system_one_pr81_principal_product_context_bridge_diagnostic"
            or bridge_report["cases_sha256"] != sha256(bridge_cases)):
        raise ValueError("invalid frozen bridge report/cases")
    split = bridge_report["split"]
    full_path = args.full_endpoint_dir / f"{split}.jsonl"
    full_manifest = json.loads((args.full_endpoint_dir / "manifest.json").read_text())
    declared = full_manifest["splits"][split]
    if (full_manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or full_manifest["executor_filtering"] is not False
            or declared["rows"] != 3120
            or declared["endpoint_sha256"] != sha256(full_path)
            or bridge_report["full_endpoint_source_sha256"] != declared["endpoint_sha256"]):
        raise ValueError("full endpoint benchmark source mismatch")
    full = {}
    for line in full_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in full:
            raise ValueError(f"duplicate full endpoint reaction ID: {reaction_id}")
        full[reaction_id] = row
    if len(full) != 3120:
        raise ValueError("full endpoint ID denominator mismatch")
    cases = [json.loads(line) for line in bridge_cases.read_text().splitlines()]
    if len(cases) != bridge_report["evaluated_reactions"]:
        raise ValueError("bridge case denominator mismatch")
    selected = cases[:args.limit] if args.limit else cases
    if args.limit < 0 or not selected:
        raise ValueError("invalid/nonempty structural replay selection")
    args.output.mkdir(parents=True)
    counts: Counter[str] = Counter()
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("w") as handle:
        for index, case in enumerate(selected, 1):
            reaction_id = str(case["id"])
            reference = full[reaction_id]
            if canonical_visible(case["principal_product_input"]) != canonical_visible(reference["product_unmapped"]):
                raise ValueError(f"{reaction_id}: principal-product source mismatch")
            full_pred, structural_pred, failure = replay_provenance(case)
            exact = bool(case["completed"] and structural_exact(
                structural_pred, reference["structural_precursor"]
            ))
            result = {
                "reaction_id": reaction_id,
                "principal_product_input": case["principal_product_input"],
                "completed": bool(case["completed"]),
                "replay_failure": failure,
                "predicted_full_precursor": full_pred if case["completed"] else None,
                "predicted_structural_precursor": structural_pred if case["completed"] else None,
                "expected_structural_precursor": reference["structural_precursor"],
                "structural_exact": exact,
                "strict_full_precursor_exact": case["endpoint_exact_strict_full_precursor"],
            }
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            counts["evaluated"] += 1
            counts["structural_exact"] += int(exact)
            counts["strict_full_precursor_exact"] += int(case["endpoint_exact_strict_full_precursor"])
            counts["completed"] += int(case["completed"])
            if index % 100 == 0:
                print(json.dumps({"phase": "structural_replay", "cases": index,
                                  "structural_exact": counts["structural_exact"]}), flush=True)
    report = {
        "artifact_type": "system_one_pr81_provenance_structural_endpoint_diagnostic",
        "scope": "strict_trace_overlap_principal_product_input_structural_scorer_not_full_3120_benchmark",
        "split": split,
        "bridge_report_sha256": sha256(args.bridge_dir / "report.json"),
        "bridge_cases_sha256": sha256(bridge_cases),
        "full_endpoint_sha256": declared["endpoint_sha256"],
        "selected": len(selected),
        "counts": dict(counts),
        "cases_sha256": sha256(cases_path),
        "evaluator_sha256": sha256(Path(__file__)),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
