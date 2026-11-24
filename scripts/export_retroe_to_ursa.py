"""Export Retro-E scored_group.json routes to official URSA (RetroCast schema 2) JSON and flat step-level JSONL for independent FlowER forward cross-validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import inchi as rd_inchi
except Exception:  # pragma: no cover
    Chem = None
    rd_inchi = None


def _canon_and_inchikey(smiles: str):
    smiles = str(smiles or "").strip()
    if not smiles:
        return "", ""
    if Chem is None:
        return smiles, ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, ""
    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        ik = rd_inchi.MolToInchiKey(mol)
    except Exception:
        ik = ""
    return canon, ik


def _build_mol_node(smiles: str, occ: str, rxn_by_occ: dict, stock_set: set):
    canon, ik = _canon_and_inchikey(smiles)
    node = {
        "smiles": canon,
        "inchikey": ik,
        "is_leaf": occ not in rxn_by_occ,
        "in_stock": canon in stock_set or smiles in stock_set,
    }
    rxn = rxn_by_occ.get(occ)
    if rxn is not None:
        precs = [str(p) for p in (rxn.get("precursors") or []) if p]
        children = []
        for idx, p_smi in enumerate(precs):
            child_occ = f"{occ}.{idx}"
            children.append(_build_mol_node(p_smi, child_occ, rxn_by_occ, stock_set))
        node["is_leaf"] = False
        node["product_of"] = {
            "reactants": children,
            "mapped_smiles": str(rxn.get("reaction_smiles") or ""),
            "occurrence": occ,
        }
    return node


def convert_report_to_ursa_route(report: dict):
    """Convert one Retro-E scored_group report dict into RetroCast/URSA schema 2 route dict + flat reaction steps."""
    rxns = report.get("reactions") or []
    ev = report.get("reward_evidence") or {}
    is_closed = (report.get("status") == "stock_solved" or bool(ev.get("structural_stock_closed"))) and len(rxns) > 0
    if not is_closed:
        return None, []

    rxn_by_occ = {}
    for idx, r in enumerate(rxns):
        occ = str(r.get("occurrence") if r.get("occurrence") is not None else idx)
        rxn_by_occ[occ] = r

    root_occ = "0" if "0" in rxn_by_occ else sorted(rxn_by_occ.keys())[0]
    root_smi = str(report.get("root") or rxn_by_occ[root_occ].get("product") or "")
    stock_set = set(str(s) for s in (report.get("stock_leaves") or []) if s)

    target_node = _build_mol_node(root_smi, root_occ, rxn_by_occ, stock_set)
    ursa_route = {
        "task_id": str(report.get("task_id") or ""),
        "schema_version": "2",
        "target": target_node,
        "annotations": {
            "status": report.get("status"),
            "num_reactions": len(rxns),
        },
    }

    # Build flat step list for independent FlowER forward cross-validation
    flat_steps = []
    for occ, r in sorted(rxn_by_occ.items()):
        prod_c, _ = _canon_and_inchikey(str(r.get("product") or ""))
        prec_c = [_canon_and_inchikey(str(p))[0] for p in (r.get("precursors") or []) if p]
        flat_steps.append({
            "task_id": str(report.get("task_id") or ""),
            "occurrence": occ,
            "product_smiles": prod_c,
            "reactants_smiles": ".".join(prec_c),
            "forward_reaction_smiles": f"{'.'.join(prec_c)}>>{prod_c}",
            "precursors": prec_c,
        })
    return ursa_route, flat_steps


def _load_one(sub: Path):
    f = sub / "scored_group.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    return data[0] if isinstance(data, list) else data


def export_groups_dir(groups_dir: Path, out_dir: Path, label: str = "retroe"):
    out_dir.mkdir(parents=True, exist_ok=True)
    subdirs = sorted([d for d in groups_dir.iterdir() if d.is_dir()])
    with ThreadPoolExecutor(max_workers=24) as pool:
        reports = [r for r in pool.map(_load_one, subdirs) if r is not None]

    routes = {}
    all_steps = []
    for rep in reports:
        ursa_route, flat_steps = convert_report_to_ursa_route(rep)
        if ursa_route is not None:
            tid = ursa_route["task_id"] or f"target_{len(routes)}"
            routes[tid] = [ursa_route]
            all_steps.extend(flat_steps)

    ursa_json_path = out_dir / f"{label}_ursa_routes.json"
    ursa_json_path.write_text(json.dumps(routes, indent=2) + "\n")

    steps_jsonl_path = out_dir / f"{label}_flower_forward_steps.jsonl"
    steps_jsonl_path.write_text(
        "\n".join(json.dumps(s, sort_keys=True) for s in all_steps) + ("\n" if all_steps else "")
    )
    summary = {
        "label": label,
        "total_reports": len(reports),
        "exported_closed_routes": len(routes),
        "exported_forward_steps": len(all_steps),
        "ursa_routes_json": str(ursa_json_path),
        "flower_forward_steps_jsonl": str(steps_jsonl_path),
    }
    (out_dir / f"{label}_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="retroe")
    args = parser.parse_args()
    summary = export_groups_dir(args.groups_dir, args.out_dir, label=args.label)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
