"""Descriptive symbolic graph changes and explicitly exposure-scoped support.

These features do not turn executor acceptance into laboratory feasibility.
The v1 signature abstraction intentionally retains its historical limitations.
"""
from functools import lru_cache
import hashlib
import json


@lru_cache(maxsize=4096)
def _state_graph(state):
    from rdkit import Chem
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(state, params)
    if mol is None:
        raise ValueError("symbolic state unavailable to the pinned RDKit parser")
    atoms = {a.GetAtomMapNum(): {"element": a.GetAtomicNum(), "charge": a.GetFormalCharge(),
        "aromatic": a.GetIsAromatic(), "chiral_tag": str(a.GetChiralTag())} for a in mol.GetAtoms()}
    if 0 in atoms or len(atoms) != mol.GetNumAtoms():
        raise ValueError("graph changes require unique private atom maps")
    bonds = {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))): b.GetBondTypeAsDouble() for b in mol.GetBonds()}
    components = len(Chem.GetMolFrags(mol))
    return atoms, bonds, components, mol.GetNumBonds() - mol.GetNumAtoms() + components


def molecular_delta(before, after):
    a0, b0, f0, c0 = _state_graph(before)
    a1, b1, f1, c1 = _state_graph(after)
    changed = [{"atoms": list(pair), "before": b0.get(pair, 0.), "after": b1.get(pair, 0.)}
        for pair in sorted(b0.keys() | b1.keys()) if b0.get(pair, 0.) != b1.get(pair, 0.)]
    changed_atoms = {m for b in changed for m in b["atoms"]}
    changed_atoms.update(m for m in a0.keys() | a1.keys() if a0.get(m) != a1.get(m))
    return {"changed_atom_maps": sorted(changed_atoms), "changed_bonds": changed,
        "fragment_count_delta": f1 - f0, "cycle_rank_delta": c1 - c0,
        "net_charge_delta": sum(a["charge"] for a in a1.values()) - sum(a["charge"] for a in a0.values()),
        "atom_inventory_preserved": a0.keys() == a1.keys(),
        "interpretation": "native symbolic graph differences, not an inferred mechanism",
        "experimental_feasibility_established": False}


def program_support(actions, exposure):
    from mechet.natural_language_electron_flow import compile_event_arguments
    from mechet.proof_to_trace import execution_primitive_signature
    complete = bool(actions) and actions[-1].get("name") == "finish_trace"
    sequence, counts = [], []
    ordinary = True
    for index, action in enumerate(actions):
        if action.get("name") != "apply_electron_flow":
            continue
        args = action["arguments"]
        if args.get("bond_order_changes") or args.get("charge_changes"):
            ordinary = False
            continue
        moves = compile_event_arguments(action["state_before"], args)
        signatures = [execution_primitive_signature(m, action["state_before"]) for m in moves]
        sequence.append(signatures)
        counts.append({"action_index": index, "primitive_signatures": signatures,
            "training_reaction_frequencies": [exposure["primitive_counts"].get(s, 0) for s in signatures]})
    signature = hashlib.sha256(json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if complete and ordinary and sequence else None
    frequencies = [n for event in counts for n in event["training_reaction_frequencies"]]
    return {"program_complete": complete, "ordinary_source_sink_program": ordinary,
        "complete_signature": signature,
        "complete_signature_training_count": exposure["program_counts"].get(signature, 0) if signature else None,
        "events": counts, "minimum_primitive_training_frequency": min(frequencies) if frequencies else None,
        "all_primitives_seen_at_least_five": all(n >= 5 for n in frequencies) if frequencies else None,
        "basis": "source_to_sink_execution_moves_v1",
        "limitations": ["imports omitted from whole-program signature", "coupled-move ordering can affect signatures", "v1 atom descriptors inherit historical explicit-H handling"],
        "route_C1_claim": False, "independent_chemical_precedent": False}
