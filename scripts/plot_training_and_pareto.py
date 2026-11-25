"""Plot Nature/NMI-grade 4-panel training dynamics, 11-dim multi-objective profile, and route-level Pareto frontier."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from scripts.prepare_nmi_day34_eval_suite import (
    _extract_selected_pr18,
    _load_group_report,
    _route_lls,
    _route_multi_objective,
    _stereo_preserved,
    pareto_preference_score,
)

# Nature Publishing Group (NPG) inspired editorial palette
INK = "#1B2A4A"
SPINE = "#64748B"
GRID = "#E2E8F0"
BG_PANEL = "#FAFBFD"
TEAL = "#00A087"
BLUE = "#3C5488"
CORAL = "#E64B35"
AMBER = "#F39B7F"
SLATE = "#94A3B8"
GOLD = "#D97706"


def _pareto_front_indices(xs, ys):
    """Return indices of non-dominated points maximizing both x and y, sorted by x ascending."""
    pts = sorted(range(len(xs)), key=lambda i: (-xs[i], -ys[i]))
    front = []
    best_y = -1e18
    for idx in pts:
        if ys[idx] > best_y:
            front.append(idx)
            best_y = ys[idx]
    front.sort(key=lambda i: xs[i])
    return front


def _load_route_points(groups_dir: Path, label: str):
    rows = []
    for sub in sorted(d for d in groups_dir.iterdir() if d.is_dir()):
        r = _load_group_report(sub)
        if not r or r.get("status") == "already_stock":
            continue
        ev = r.get("reward_evidence") or {}
        rxns = r.get("reactions") or []
        if not ((r.get("status") == "stock_solved" or bool(ev.get("structural_stock_closed"))) and len(rxns) > 0):
            continue
        bidir, fwd = _extract_selected_pr18(r.get("stepwise") or {})
        all_s = bidir + fwd
        min_pr18 = min(all_s) if all_s else 0.0
        lls = _route_lls(rxns)
        ae, simp, g_ok = _route_multi_objective(r)
        stereo = _stereo_preserved(rxns)
        med = pareto_preference_score(min_pr18=min_pr18, lls=lls, atom_economy=ae, leaf_simplicity=simp, green_ok=g_ok, mode="medchem")
        proc = pareto_preference_score(min_pr18=min_pr18, lls=lls, atom_economy=ae, leaf_simplicity=simp, green_ok=g_ok, mode="process")
        rows.append({
            "label": label,
            "task_id": r.get("task_id", ""),
            "root": r.get("root", ""),
            "num_reactions": len(rxns),
            "lls": lls,
            "convergent": int(len(rxns) > lls),
            "min_pr18": round(min_pr18, 4),
            "atom_economy": ae,
            "leaf_simplicity": simp,
            "stereo_ok": int(stereo),
            "green_ok": int(g_ok),
            "medchem_score": med,
            "process_score": proc,
        })
    return rows


def _load_training_trace(progress_jsonl: Path):
    steps, grad_norms, kls, route_losses, window_closed = [], [], [], [], []
    for line in progress_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event") != "external-route-round-complete":
            continue
        metrics_list = obj.get("metrics") or []
        if not metrics_list:
            continue
        m0 = metrics_list[0]
        total_closed = sum(int(m.get("native_stock_solved", 0) or 0) for m in metrics_list)
        total_targets = sum(int(m.get("targets", 0) or 0) for m in metrics_list)
        w_rate = total_closed / max(total_targets, 1)
        for wm in m0.get("window_update_metrics") or []:
            steps.append(int(wm.get("learner_unit", 0)) + 1)
            grad_norms.append(float(wm.get("grad_norm", 0.0)))
            kls.append(float(wm.get("reference_kl", 0.0)))
            route_losses.append(float(wm.get("joint_route_pg_loss", 0.0)))
            window_closed.append(w_rate * 100.0)
    return steps, grad_norms, kls, route_losses, window_closed


def _style_ax(ax, letter: str, title: str):
    ax.set_facecolor(BG_PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(colors=INK, labelsize=9.2, width=0.8)
    ax.grid(True, color=GRID, linestyle="-", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.11, 1.07, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK, va="top")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-jsonl", type=Path, required=True)
    parser.add_argument("--step0-groups", type=Path, required=True)
    parser.add_argument("--step32-groups", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps, grad_norms, kls, route_losses, window_closed = _load_training_trace(args.progress_jsonl)
    pts0 = _load_route_points(args.step0_groups, "step0_round204")
    pts32 = _load_route_points(args.step32_groups, "step32_plan05")
    summary = json.loads(args.summary_json.read_text())
    m0, m32 = summary[0], summary[1]

    pf_mech_ae = _pareto_front_indices([p["min_pr18"] for p in pts32], [p["atom_economy"] for p in pts32])
    multi_idx = [i for i, p in enumerate(pts32) if p["lls"] >= 2]
    pf_multi_local = _pareto_front_indices([pts32[i]["min_pr18"] for i in multi_idx], [pts32[i]["atom_economy"] for i in multi_idx])
    pf_multi = [multi_idx[j] for j in pf_multi_local]
    pf_med_proc = _pareto_front_indices([p["medchem_score"] for p in pts32], [p["process_score"] for p in pts32])

    pareto_rows = []
    for idx in sorted(set(pf_mech_ae) | set(pf_multi) | set(pf_med_proc),
                      key=lambda i: (-pts32[i]["lls"], -pts32[i]["process_score"])):
        row = dict(pts32[idx])
        row["on_global_mech_ae_front"] = int(idx in pf_mech_ae)
        row["on_multistep_mech_ae_front"] = int(idx in pf_multi)
        row["on_medchem_process_front"] = int(idx in pf_med_proc)
        pareto_rows.append(row)

    csv_path = args.out_dir / "pareto_frontier_routes_step32.csv"
    if pareto_rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()))
            w.writeheader()
            w.writerows(pareto_rows)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": SPINE,
        "text.color": INK,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.6), dpi=300, facecolor="white")
    fig.suptitle(
        "Retro-E Long-Horizon Mechanistic RL (Run 05) — Training Dynamics & Multi-Objective Pareto Frontier",
        fontsize=13.5, fontweight="bold", color=INK, y=0.985
    )

    # Panel a: Training Trajectory
    ax = axes[0, 0]
    _style_ax(ax, "a", "Live RL Trajectory (Steps 1–36 / 96, 100% Non-Stock Full Routes)")
    ax.fill_between(steps, min(window_closed) - 1.5, window_closed, color=BLUE, alpha=0.10, zorder=1)
    ax.plot(steps, window_closed, color=BLUE, lw=2.4, marker="o", ms=4.2, mfc="white", mew=1.5,
            label="Window Multi-Step Closure (%)", zorder=3)
    ax.set_xlabel("Optimizer Step", fontsize=9.8, fontweight="bold", color=INK)
    ax.set_ylabel("Window Non-Stock Closure (%)", fontsize=9.8, fontweight="bold", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(min(window_closed) - 1.5, max(window_closed) + 2.5)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(CORAL)
    ax2.spines["right"].set_linewidth(0.85)
    ax2.plot(steps, grad_norms, color=CORAL, lw=1.7, ls="--", alpha=0.9, label="Policy Grad Norm", zorder=2)
    ax2.set_ylabel("Policy Grad Norm", fontsize=9.8, fontweight="bold", color=CORAL)
    ax2.tick_params(axis="y", labelcolor=CORAL, labelsize=9.2)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper left", frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.5)

    # Panel b: Multi-Objective Profile
    ax = axes[0, 1]
    _style_ax(ax, "b", "Multi-Objective Evaluation Profile (512 Clean VALID, N = 499)")
    metric_keys = [
        ("stock_closure_rate", "Stock Closure"),
        ("mech_gated_closure_rate", "Mech-Gated Closure"),
        ("mean_leaf_simplicity", "Leaf Simplicity"),
        ("mean_pareto_process", "Process Pareto"),
        ("mean_pareto_medchem", "MedChem Pareto"),
        ("mean_atom_economy", "Exact Atom Economy"),
        ("stereo_preservation_rate", "Stereo Preservation"),
        ("green_compliance_rate", "Green Compliance"),
    ]
    y_pos = np.arange(len(metric_keys))
    v0 = [m0[k] * 100.0 for k, _ in metric_keys]
    v32 = [m32[k] * 100.0 for k, _ in metric_keys]
    h = 0.36
    ax.barh(y_pos - h / 2, v0, height=h, color=SLATE, alpha=0.82, edgecolor="white", lw=0.6,
            label="step 0 (round-204, T=0.50)", zorder=2)
    ax.barh(y_pos + h / 2, v32, height=h, color=TEAL, alpha=0.92, edgecolor="white", lw=0.6,
            label="step 32 (05 RL, T=0.80)", zorder=3)
    for i, (a0, a32) in enumerate(zip(v0, v32)):
        ax.text(a32 + 1.0, i + h / 2, f"{a32:.1f}%", va="center", ha="left", fontsize=8.2, fontweight="bold", color=INK)
        ax.text(a0 + 1.0, i - h / 2, f"{a0:.1f}%", va="center", ha="left", fontsize=7.8, color=SPINE)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in metric_keys], fontsize=9.2, color=INK)
    ax.set_xlabel("Score / Rate (%)", fontsize=9.8, fontweight="bold", color=INK)
    ax.set_xlim(0, 114)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.5)

    # Panel c: Pareto Frontier (min PR18 vs Exact Atom Economy, global + LLS>=2 multistep)
    ax = axes[1, 0]
    _style_ax(ax, "c", "Route Pareto Frontier: Mechanistic Reliability vs Exact Atom Economy")
    cmap = LinearSegmentedColormap.from_list("npg_lls", ["#00A087", "#3C5488", "#1B2A4A"])
    xs32 = [p["min_pr18"] for p in pts32]
    ys32 = [p["atom_economy"] for p in pts32]
    lls32 = [p["lls"] for p in pts32]
    sc = ax.scatter(xs32, ys32, c=lls32, cmap=cmap, vmin=1, vmax=max(lls32), s=52, alpha=0.78,
                    edgecolors="white", lw=0.6, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=26)
    cbar.set_label("Longest Linear Sequence (LLS)", fontsize=9, fontweight="bold", color=INK)
    cbar.outline.set_edgecolor(SPINE)

    # Global Pareto envelope
    fx = [xs32[i] for i in pf_mech_ae]
    fy = [ys32[i] for i in pf_mech_ae]
    ax.plot(fx, fy, color=CORAL, lw=2.3, ls="-", zorder=4, label=f"Global Pareto Front ({len(pf_mech_ae)} routes)")
    ax.scatter(fx, fy, color=CORAL, marker="*", s=165, edgecolors="white", lw=0.7, zorder=5)

    # Multistep (LLS >= 2) Pareto envelope
    if pf_multi:
        mx = [xs32[i] for i in pf_multi]
        my = [ys32[i] for i in pf_multi]
        ax.plot(mx, my, color=GOLD, lw=2.0, ls="--", zorder=4, label=f"Multi-Step (LLS≥2) Front ({len(pf_multi)} routes)")
        ax.scatter(mx, my, color=GOLD, marker="D", s=58, edgecolors="white", lw=0.7, zorder=5)

    ax.axvline(-0.25, color=SPINE, ls=":", lw=1.2, zorder=2, label="Mechanistic Gate (-0.25)")
    ax.set_xlabel("Bottleneck Mechanistic Score (min PR18)", fontsize=9.8, fontweight="bold", color=INK)
    ax.set_ylabel("Exact Route Atom Economy", fontsize=9.8, fontweight="bold", color=INK)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.3)

    # Panel d: Inference-Time MedChem vs Process Pareto Frontier
    ax = axes[1, 1]
    _style_ax(ax, "d", "Multi-Objective Pareto Trade-off: MedChem vs Process Mode")
    ax.scatter([p["medchem_score"] for p in pts0], [p["process_score"] for p in pts0],
               color=SLATE, alpha=0.45, s=42, edgecolors="white", lw=0.4, label=f"step 0 solved (n={len(pts0)})", zorder=2)
    ax.scatter([p["medchem_score"] for p in pts32], [p["process_score"] for p in pts32],
               color=TEAL, alpha=0.76, s=50, edgecolors="white", lw=0.5, label=f"step 32 solved (n={len(pts32)})", zorder=3)
    pf0 = _pareto_front_indices([p["medchem_score"] for p in pts0], [p["process_score"] for p in pts0])
    ax.plot([pts0[i]["medchem_score"] for i in pf0], [pts0[i]["process_score"] for i in pf0],
            color=SPINE, lw=1.8, ls="--", marker="s", ms=5.5, mfc="white", label=f"step 0 Front ({len(pf0)})", zorder=4)
    ax.plot([pts32[i]["medchem_score"] for i in pf_med_proc], [pts32[i]["process_score"] for i in pf_med_proc],
            color=CORAL, lw=2.4, ls="-", label=f"step 32 Front ({len(pf_med_proc)})", zorder=5)
    ax.scatter([pts32[i]["medchem_score"] for i in pf_med_proc], [pts32[i]["process_score"] for i in pf_med_proc],
               color=CORAL, marker="*", s=180, edgecolors="white", lw=0.7, zorder=6)
    ax.set_xlabel("MedChem Preference Score (LLS + Reliability)", fontsize=9.8, fontweight="bold", color=INK)
    ax.set_ylabel("Process Preference Score (Reliability + Exact AE + Green)", fontsize=9.8, fontweight="bold", color=INK)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = args.out_dir / "training_curves_and_pareto.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    out_summary = {
        "png_path": str(png_path),
        "pareto_csv_path": str(csv_path),
        "global_mech_ae_front": len(pf_mech_ae),
        "multistep_lls_ge2_front": len(pf_multi),
        "medchem_process_front": len(pf_med_proc),
        "multistep_pareto_routes": [pts32[i] for i in pf_multi],
    }
    (args.out_dir / "pareto_summary.json").write_text(json.dumps(out_summary, indent=2) + "\n")
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()
