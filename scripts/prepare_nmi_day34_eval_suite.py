"""Prepare Day 3-4 NMI evaluation panels (failure recovery, supply disruption) and score six-dimensional synthesis practicality."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import statistics


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
        convergent_closed=convergent,
        convergent_fraction=round(convergent / max(n_closed, 1), 4),
        backtrack_closed=backtrack_closed,
        cycle_failed_targets=cycle_failures,
        unique_root_disconnections=len(first_disconnections),
    ), reports


def freeze_panels(reports, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. Failure recovery 100 panel (prioritize ROUTE_CYCLE then rejected_action)
    cycle_rows, other_fail_rows = [], []
    for r in reports:
        if r.get("status") != "rejected_action":
            continue
        sw = r.get("stepwise") or {}
        err_list = [str(c.get("error") or "") for sl in (sw.get("slates") or [])
                    for c in (sl.get("candidates") or []) if not c.get("accepted") and c.get("error")]
        is_cycle = any("ROUTE_CYCLE" in e for e in err_list)
        entry = dict(
            task_id=r.get("task_id", ""),
            root=r.get("root", ""),
            status=r.get("status", ""),
            failure_kind="ROUTE_CYCLE" if is_cycle else "REJECTED_ACTION",
            last_error=err_list[-1] if err_list else "",
            committed_reactions=len(r.get("reactions") or []),
        )
        (cycle_rows if is_cycle else other_fail_rows).append(entry)
    cycle_rows.sort(key=lambda x: x["task_id"])
    other_fail_rows.sort(key=lambda x: x["task_id"])
    failure_100 = (cycle_rows + other_fail_rows)[:100]
    fail_path = out_dir / "failure_recovery_100.jsonl"
    fail_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in failure_100) + ("\n" if failure_100 else ""))

    # 2. Supply disruption panel (all nonzero stock-closed targets with a nontrivial leaf blocked)
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
    disrupt_path = out_dir / "supply_disruption_panel.jsonl"
    disrupt_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in disrupt_rows) + ("\n" if disrupt_rows else ""))

    manifest = dict(
        failure_recovery_count=len(failure_100),
        failure_recovery_cycle_count=sum(1 for x in failure_100 if x["failure_kind"] == "ROUTE_CYCLE"),
        failure_recovery_sha256=hashlib.sha256(fail_path.read_bytes()).hexdigest(),
        supply_disruption_count=len(disrupt_rows),
        supply_disruption_sha256=hashlib.sha256(disrupt_path.read_bytes()).hexdigest(),
    )
    (out_dir / "panel_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=/path/to/groups")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--freeze-from", default="", help="arm label to freeze failure/disruption panels from")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    freeze_reports = None
    for spec in args.arm:
        label, path_str = spec.split("=", 1)
        metrics, reports = score_six_dimensions(Path(path_str), label)
        rows.append(metrics)
        if label == args.freeze_from:
            freeze_reports = reports

    (args.out_dir / "baseline_six_dim_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (args.out_dir / "baseline_six_dim_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = freeze_panels(freeze_reports, args.out_dir) if freeze_reports is not None else None
    print(json.dumps(dict(metrics=rows, panels=manifest), indent=2))


if __name__ == "__main__":
    main()
