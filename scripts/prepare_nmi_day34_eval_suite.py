"""Prepare Day 3-4 NMI evaluation panels (failure recovery, supply disruption) and score six-dimensional synthesis practicality."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import statistics

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None

HAZARDOUS_SUBSTRUCTS = ("[Cr]", "[Os]", "[Sn]", "[Pb]", "[Hg]", "[N-]=[N+]=[N-]")


def _mol_mw_and_heavy(smiles: str):
    smiles = str(smiles or "").strip()
    if not smiles:
        return 0.0, 0
    if Chem is None:
        return float(len(smiles) * 12), len(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0
    return float(Descriptors.MolWt(mol)), int(mol.GetNumHeavyAtoms())


def _route_multi_objective(report: dict):
    """Compute Exact Route Atom Economy, Green Reagent Compliance, and Starting-Material Simplicity."""
    target_mw, target_heavy = _mol_mw_and_heavy(report.get("root", ""))
    leaves = [str(s) for s in (report.get("stock_leaves") or []) if s]
    if not leaves:
        for r in report.get("reactions") or []:
            leaves.extend(str(p) for p in (r.get("precursors") or []) if p)
    leaf_stats = [_mol_mw_and_heavy(s) for s in leaves]
    total_leaf_mw = sum(mw for mw, _ in leaf_stats)
    atom_economy = min(1.0, target_mw / total_leaf_mw) if total_leaf_mw > 0 and target_mw > 0 else 0.0
    max_leaf_heavy = max((h for _, h in leaf_stats), default=0)
    # Simplicity: 1 - (heaviest_leaf / target_heavy), higher means genuine multi-step disassembly from simple BBs
    leaf_simplicity = max(0.0, 1.0 - (max_leaf_heavy / max(target_heavy, 1))) if target_heavy > 0 else 0.0

    # Green compliance: check no hazardous heavy-metal or azide species in precursors/leaves
    green_ok = True
    if Chem is not None:
        patterns = [Chem.MolFromSmarts(s) for s in HAZARDOUS_SUBSTRUCTS]
        all_smi = list(leaves)
        for r in report.get("reactions") or []:
            all_smi.extend(str(p) for p in (r.get("precursors") or []) if p)
        for smi in set(all_smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if any(pat is not None and mol.HasSubstructMatch(pat) for pat in patterns):
                green_ok = False
                break
    return round(atom_economy, 4), round(leaf_simplicity, 4), green_ok


def pareto_preference_score(
    *,
    min_pr18: float,
    lls: int,
    atom_economy: float,
    leaf_simplicity: float,
    green_ok: bool,
    mode: str = "medchem",
) -> float:
    """Inference-time multi-objective Pareto router (no retraining required): MedChem vs Process mode."""
    mech_norm = max(0.0, min(1.0, (min_pr18 + 0.5) / 0.8))
    lls_norm = max(0.0, 1.0 - max(0, lls - 1) / 8.0)
    green_val = 1.0 if green_ok else 0.0
    if mode == "process":
        return round(0.40 * mech_norm + 0.25 * atom_economy + 0.20 * green_val + 0.15 * leaf_simplicity, 4)
    # Default: MedChem (fast LLS + high mechanistic reliability + simple starting materials)
    return round(0.35 * mech_norm + 0.40 * lls_norm + 0.15 * leaf_simplicity + 0.10 * atom_economy, 4)


def _load_group_report(sub: Path):
    f = sub / "scored_group.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    return data[0] if isinstance(data, list) else data


def _extract_selected_pr18(stepwise):
    sel = set(stepwise.get("selected_path") or [])
    bidir, fwd = [], []
    for sl in stepwise.get("slates") or []:
        for c in sl.get("candidates") or []:
            if c.get("request_id") not in sel:
                continue
            if c.get("teacher_score") is not None:
                bidir.append(float(c["teacher_score"]))
            fz = (c.get("teacher_z") or {}).get("chemical_forward")
            if fz is not None:
                fwd.append(float(fz))
    return bidir, fwd


def _stereo_preserved(reactions):
    for rxn in reactions or []:
        prod = str(rxn.get("product") or "")
        precs = [str(p) for p in (rxn.get("precursors") or [])]
        if "@" in prod and not any("@" in p for p in precs):
            return False
    return True


def _route_lls(reactions):
    if not reactions:
        return 0
    return max((str(r.get("occurrence", "0")).count(".") + 1) for r in reactions)


def _nontrivial_stock_leaf(report):
    leaves = [str(s) for s in (report.get("stock_leaves") or []) if s]
    if not leaves:
        # fallback to final reaction precursors
        for r in report.get("reactions") or []:
            for p in r.get("precursors") or []:
                if p:
                    leaves.append(str(p))
    # pick the heaviest/longest SMILES leaf (avoid H2O, Br-, etc.)
    candidates = [s for s in dict.fromkeys(leaves) if len(s) >= 4]
    if not candidates:
        candidates = list(dict.fromkeys(leaves))
    if not candidates:
        return None
    candidates.sort(key=lambda s: (-len(s), s))
    return candidates[0]


def score_six_dimensions(groups_dir: Path, label: str, mech_threshold: float = -0.25):
    subdirs = sorted([d for d in groups_dir.iterdir() if d.is_dir()])
    with ThreadPoolExecutor(max_workers=24) as pool:
        reports = [r for r in pool.map(_load_group_report, subdirs) if r is not None]

    n_total = len(reports)
    n_already = sum(1 for r in reports if r.get("status") == "already_stock")
    need_synth = [r for r in reports if r.get("status") != "already_stock"]
    n_synth = len(need_synth)

    closed = []
    mech_closed = []
    stereo_ok = 0
    lls_vals = []
    ae_vals = []
    simplicity_vals = []
    green_ok_count = 0
    medchem_scores = []
    process_scores = []
    convergent = 0
    backtrack_closed = 0
    first_disconnections = set()
    cycle_failures = 0

    for r in need_synth:
        st = r.get("status", "")
        ev = r.get("reward_evidence") or {}
        sw = r.get("stepwise") or {}
        rxns = r.get("reactions") or []
        is_closed = (st == "stock_solved" or bool(ev.get("structural_stock_closed"))) and len(rxns) > 0
        if is_closed:
            closed.append(r)
            bidir, fwd = _extract_selected_pr18(sw)
            all_s = bidir + fwd
            min_s = min(all_s) if all_s else 0.0
            if min_s >= mech_threshold:
                mech_closed.append(r)
            if _stereo_preserved(rxns):
                stereo_ok += 1
            lls = _route_lls(rxns)
            lls_vals.append(lls)
            ae, simp, g_ok = _route_multi_objective(r)
            ae_vals.append(ae)
            simplicity_vals.append(simp)
            if g_ok:
                green_ok_count += 1
            medchem_scores.append(
                pareto_preference_score(
                    min_pr18=min_s, lls=lls, atom_economy=ae, leaf_simplicity=simp, green_ok=g_ok, mode="medchem"
                )
            )
            process_scores.append(
                pareto_preference_score(
                    min_pr18=min_s, lls=lls, atom_economy=ae, leaf_simplicity=simp, green_ok=g_ok, mode="process"
                )
            )
            if len(rxns) > lls:
                convergent += 1
            if int(sw.get("backtracks", 0) or 0) > 0:
                backtrack_closed += 1
            root_rxn = next((x for x in rxns if str(x.get("occurrence")) == "0"), rxns[0])
            key = (str(root_rxn.get("product", "")), tuple(sorted(str(p) for p in root_rxn.get("precursors") or [])))
            first_disconnections.add(key)
        else:
            if any("ROUTE_CYCLE" in str(c.get("error") or "")
                   for sl in (sw.get("slates") or []) for c in (sl.get("candidates") or [])):
                cycle_failures += 1

    n_closed = len(closed)
    n_mech = len(mech_closed)
    return dict(
        label=label,
        total_targets=n_total,
        already_stock=n_already,
        synthesis_needed=n_synth,
        nonzero_stock_closed=n_closed,
        stock_closure_rate=round(n_closed / max(n_synth, 1), 4),
        mech_gated_closed=n_mech,
        mech_gated_closure_rate=round(n_mech / max(n_synth, 1), 4),
        mech_retention_ratio=round(n_mech / max(n_closed, 1), 4),
        stereo_preservation_rate=round(stereo_ok / max(n_closed, 1), 4),
        mean_lls=round(statistics.mean(lls_vals), 3) if lls_vals else None,
        mean_atom_economy=round(statistics.mean(ae_vals), 4) if ae_vals else None,
        mean_leaf_simplicity=round(statistics.mean(simplicity_vals), 4) if simplicity_vals else None,
        green_compliance_rate=round(green_ok_count / max(n_closed, 1), 4),
        mean_pareto_medchem=round(statistics.mean(medchem_scores), 4) if medchem_scores else None,
        mean_pareto_process=round(statistics.mean(process_scores), 4) if process_scores else None,
        convergent_closed=convergent,
        convergent_fraction=round(convergent / max(n_closed, 1), 4),
        backtrack_closed=backtrack_closed,
        cycle_failed_targets=cycle_failures,
        unique_root_disconnections=len(first_disconnections),
    ), reports


def freeze_panels(reports, out_dir: Path, *, split_role: str = "public_test"):
    if split_role != "public_test":
        raise ValueError(
            f"Refusing to freeze main-paper stress/fork panels from split_role={split_role!r}; "
            "must use an independent public test archive (e.g. Reference 1084 / URSA 200)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. Full failure recovery panel (all ROUTE_CYCLE failures on the public test set, no arbitrary truncation)
    cycle_rows = []
    for r in reports:
        if r.get("status") != "rejected_action":
            continue
        sw = r.get("stepwise") or {}
        err_list = [str(c.get("error") or "") for sl in (sw.get("slates") or [])
                    for c in (sl.get("candidates") or []) if not c.get("accepted") and c.get("error")]
        if not any("ROUTE_CYCLE" in e for e in err_list):
            continue
        cycle_rows.append(dict(
            task_id=r.get("task_id", ""),
            root=r.get("root", ""),
            status=r.get("status", ""),
            failure_kind="ROUTE_CYCLE",
            last_error=err_list[-1] if err_list else "",
            committed_reactions=len(r.get("reactions") or []),
        ))
    cycle_rows.sort(key=lambda x: x["task_id"])
    fail_path = out_dir / "public_cycle_recovery_full.jsonl"
    fail_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in cycle_rows) + ("\n" if cycle_rows else ""))

    # 2. Full supply disruption panel (all nonzero stock-closed targets with a nontrivial leaf blocked)
    disrupt_rows = []
    for r in reports:
        ev = r.get("reward_evidence") or {}
        rxns = r.get("reactions") or []
        if not ((r.get("status") == "stock_solved" or ev.get("structural_stock_closed")) and len(rxns) > 0):
            continue
        blocked = _nontrivial_stock_leaf(r)
        if not blocked:
            continue
        disrupt_rows.append(dict(
            task_id=r.get("task_id", ""),
            root=r.get("root", ""),
            original_reactions=len(rxns),
            original_lls=_route_lls(rxns),
            blocked_stock_smiles=blocked,
        ))
    disrupt_rows.sort(key=lambda x: x["task_id"])
    disrupt_path = out_dir / "public_supply_disruption_full.jsonl"
    disrupt_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in disrupt_rows) + ("\n" if disrupt_rows else ""))

    # 3. Full competing electronic forks observed on the public test set (for Fig 3d probability shift)
    fork_rows = []
    for r in reports:
        sw = r.get("stepwise") or {}
        for idx, sl in enumerate(sw.get("slates") or []):
            outs = sl.get("outcomes") or {}
            vals = set(outs.values())
            if "closed" in vals and "failed" in vals:
                fork_rows.append(dict(
                    task_id=r.get("task_id", ""),
                    root=r.get("root", ""),
                    slate_index=idx,
                    slate_key=sl.get("key", ""),
                    candidate_count=len(sl.get("candidates") or []),
                ))
    fork_rows.sort(key=lambda x: (x["task_id"], x["slate_index"]))
    fork_path = out_dir / "public_competing_forks_full.jsonl"
    fork_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in fork_rows) + ("\n" if fork_rows else ""))

    manifest = dict(
        split_role=split_role,
        failure_recovery_count=len(cycle_rows),
        failure_recovery_sha256=hashlib.sha256(fail_path.read_bytes()).hexdigest(),
        supply_disruption_count=len(disrupt_rows),
        supply_disruption_sha256=hashlib.sha256(disrupt_path.read_bytes()).hexdigest(),
        competing_forks_count=len(fork_rows),
        competing_forks_sha256=hashlib.sha256(fork_path.read_bytes()).hexdigest(),
    )
    (out_dir / "panel_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=/path/to/groups")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-role", choices=("public_test", "internal_valid"), default="public_test")
    parser.add_argument("--freeze-from", default="", help="arm label to freeze public failure/disruption/fork panels from")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    freeze_reports = None
    for spec in args.arm:
        label, path_str = spec.split("=", 1)
        metrics, reports = score_six_dimensions(Path(path_str), label)
        metrics["split_role"] = args.split_role
        rows.append(metrics)
        if label == args.freeze_from:
            freeze_reports = reports

    out_name = "public_six_dim_metrics" if args.split_role == "public_test" else "internal_valid_selection_log"
    (args.out_dir / f"{out_name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (args.out_dir / f"{out_name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = freeze_panels(freeze_reports, args.out_dir, split_role=args.split_role) if freeze_reports is not None else None
    print(json.dumps(dict(split_role=args.split_role, metrics=rows, panels=manifest), indent=2))


if __name__ == "__main__":
    main()
