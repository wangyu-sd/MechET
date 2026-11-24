"""Elevation (1) Test-Time Compute Scaling Law (B=8..256 prefix truncation) and Elevation (2) Emergent Synthetic Strategy Quantitative Decoding from scored_group.json archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import statistics

try:
    from rdkit import Chem
    from rdkit.Chem import GraphDescriptors, rdMolDescriptors
except Exception:  # pragma: no cover
    Chem = None
    GraphDescriptors = None
    rdMolDescriptors = None

DEFAULT_BUDGETS = (8, 16, 32, 64, 128, 256)


def _load_report(sub: Path):
    f = sub / "scored_group.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    return data[0] if isinstance(data, list) else data


def _mol_complexity(smiles: str):
    smiles = str(smiles or "").strip()
    if not smiles:
        return 0.0, 0, 0
    if Chem is None:
        return float(len(smiles)), len(smiles), smiles.count("1") + smiles.count("2")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float(len(smiles)), len(smiles), 0
    bertz = float(GraphDescriptors.BertzCT(mol))
    heavy = int(mol.GetNumHeavyAtoms())
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    return bertz, heavy, rings


def _route_lls(reactions):
    if not reactions:
        return 0
    return max((str(r.get("occurrence", "0")).count(".") + 1) for r in reactions)


def _selected_min_pr18(stepwise):
    sel = set(stepwise.get("selected_path") or [])
    scores = []
    for sl in stepwise.get("slates") or []:
        for c in sl.get("candidates") or []:
            if c.get("request_id") not in sel:
                continue
            if c.get("teacher_score") is not None:
                scores.append(float(c["teacher_score"]))
            fz = (c.get("teacher_z") or {}).get("chemical_forward")
            if fz is not None:
                scores.append(float(fz))
    return min(scores) if scores else 0.0


def _calls_used_to_solve(report: dict) -> int:
    """Estimate cumulative model calls used along the search up to route closure."""
    sw = report.get("stepwise") or {}
    explicit = sw.get("total_calls") or sw.get("model_calls") or report.get("model_calls")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    # Count candidates across slates up to the last selected candidate
    sel = set(sw.get("selected_path") or [])
    total = 0
    last_sel_total = 0
    for sl in sw.get("slates") or []:
        cands = sl.get("candidates") or []
        for c in cands:
            total += 1
            if c.get("request_id") in sel:
                last_sel_total = total
    if last_sel_total > 0:
        return last_sel_total
    return max(total, len(report.get("reactions") or []))


def compute_test_time_scaling(reports, label: str, budgets=DEFAULT_BUDGETS, mech_threshold: float = -0.25):
    """Elevation (1): Prefix-truncated Solved@B and MV-Solved@B curves from full search logs."""
    need_synth = [r for r in reports if r.get("status") != "already_stock"]
    n = max(len(need_synth), 1)
    rows = []
    for b in budgets:
        solv0 = 0
        mv_solv = 0
        for r in need_synth:
            ev = r.get("reward_evidence") or {}
            rxns = r.get("reactions") or []
            is_closed = (r.get("status") == "stock_solved" or bool(ev.get("structural_stock_closed"))) and len(rxns) > 0
            if not is_closed:
                continue
            if _calls_used_to_solve(r) <= b:
                solv0 += 1
                if _selected_min_pr18(r.get("stepwise") or {}) >= mech_threshold:
                    mv_solv += 1
        rows.append({
            "label": label,
            "budget": int(b),
            "synthesis_needed": len(need_synth),
            "solv0_count": solv0,
            "solv0_rate": round(solv0 / n, 4),
            "mv_solv_count": mv_solv,
            "mv_solv_rate": round(mv_solv / n, 4),
        })
    return rows


def decode_emergent_strategies(reports, label: str):
    """Elevation (2): Quantify non-monotonic preparatory steps, convergency, ring disconnections, and electron-push depth."""
    need_synth = [r for r in reports if r.get("status") != "already_stock"]
    closed = [
        r for r in need_synth
        if (r.get("status") == "stock_solved" or bool((r.get("reward_evidence") or {}).get("structural_stock_closed")))
        and len(r.get("reactions") or []) > 0
    ]

    non_mono_routes = 0
    multistep_routes = 0
    convergent_routes = 0
    convergency_ratios = []
    ring_disconnect_routes = 0
    ring_disconnect_steps = 0
    total_steps = 0
    flow_counts = []

    for r in closed:
        rxns = r.get("reactions") or []
        lls = _route_lls(rxns)
        n_rxn = len(rxns)
        if n_rxn >= 2:
            multistep_routes += 1
        if n_rxn > lls:
            convergent_routes += 1
        convergency_ratios.append(n_rxn / max(lls, 1))

        has_non_mono = False
        has_ring_disc = False
        for rxn in rxns:
            total_steps += 1
            p_bertz, p_heavy, p_rings = _mol_complexity(rxn.get("product", ""))
            precs = [str(p) for p in (rxn.get("precursors") or []) if p]
            prec_stats = [_mol_complexity(p) for p in precs]
            if prec_stats:
                max_prec_bertz = max(s[0] for s in prec_stats)
                max_prec_heavy = max(s[1] for s in prec_stats)
                sum_prec_rings = sum(s[2] for s in prec_stats)
                # Non-monotonic preparatory step: precursor is temporarily more complex than product
                if max_prec_bertz > p_bertz * 1.05 or max_prec_heavy > p_heavy:
                    has_non_mono = True
                if sum_prec_rings < p_rings:
                    has_ring_disc = True
                    ring_disconnect_steps += 1
            actions = rxn.get("actions") or []
            if actions:
                flow_counts.append(sum(1 for a in actions if "FLOW" in str(a).upper()))

        if has_non_mono and n_rxn >= 2:
            non_mono_routes += 1
        if has_ring_disc:
            ring_disconnect_routes += 1

    n_closed = max(len(closed), 1)
    n_multi = max(multistep_routes, 1)
    return {
        "label": label,
        "closed_routes": len(closed),
        "multistep_routes": multistep_routes,
        "non_monotonic_multistep_count": non_mono_routes,
        "non_monotonic_multistep_rate": round(non_mono_routes / n_multi, 4),
        "convergent_routes": convergent_routes,
        "convergent_fraction": round(convergent_routes / n_closed, 4),
        "mean_convergency_index": round(statistics.mean(convergency_ratios), 4) if convergency_ratios else None,
        "ring_disconnection_routes": ring_disconnect_routes,
        "ring_disconnection_route_rate": round(ring_disconnect_routes / n_closed, 4),
        "ring_disconnection_step_rate": round(ring_disconnect_steps / max(total_steps, 1), 4),
        "mean_flow_actions_per_step": round(statistics.mean(flow_counts), 3) if flow_counts else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=/path/to/groups")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="8,16,32,64,128,256")
    args = parser.parse_args()

    budgets = tuple(int(x) for x in args.budgets.split(",") if x.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scaling_rows = []
    strategy_rows = []
    for spec in args.arm:
        label, path_str = spec.split("=", 1)
        subdirs = sorted([d for d in Path(path_str).iterdir() if d.is_dir()])
        with ThreadPoolExecutor(max_workers=24) as pool:
            reports = [r for r in pool.map(_load_report, subdirs) if r is not None]
        scaling_rows.extend(compute_test_time_scaling(reports, label, budgets=budgets))
        strategy_rows.append(decode_emergent_strategies(reports, label))

    (args.out_dir / "test_time_scaling.json").write_text(json.dumps(scaling_rows, indent=2) + "\n")
    (args.out_dir / "emergent_strategy.json").write_text(json.dumps(strategy_rows, indent=2) + "\n")
    for fname, rows in (("test_time_scaling.csv", scaling_rows), ("emergent_strategy.csv", strategy_rows)):
        if rows:
            with (args.out_dir / fname).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(json.dumps({"scaling": scaling_rows, "strategy": strategy_rows}, indent=2))


if __name__ == "__main__":
    main()
