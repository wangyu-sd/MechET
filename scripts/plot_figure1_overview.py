"""Render Nature/NMI Figure 1 v1: borderless, asymmetric 7-panel (a-g) low-opacity editorial architecture with real 1:1 undistorted ACS monochrome RDKit mechanism & crimson curved arrows."""
import argparse
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from scripts.plot_acs_mechanism import _compute_curved_arrows, _draw_bracket, _draw_retro_double_arrow
from scripts.plot_electron_flow_states import _changed_maps, _extract_subreaction_states, _keep_reacting_frags
from scripts.prepare_nmi_day34_eval_suite import _load_group_report

# Nature / NPG Low-Opacity Editorial Palette
INK = "#0F172A"
MUTED = "#475569"
TEAL = "#00A087"
BLUE = "#3C5488"
CORAL = "#E64B35"
CRIMSON = "#C81E1E"
AMBER = "#D97706"
SLATE = "#64748B"


def _wash_box(ax, x, y, w, h, color, alpha=0.08, radius=0.14, zorder=1):
    """Borderless soft low-opacity background wash (no harsh outlines!)."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder
    )
    ax.add_patch(p)
    return p


def _soft_pill(ax, cx, cy, w, h, text, color, fill_alpha=0.13, text_color=None, fontsize=8.2, bold=True, zorder=5):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor=color, edgecolor="none", alpha=fill_alpha, zorder=zorder
    )
    ax.add_patch(p)
    ax.text(
        cx, cy, text, color=text_color or color, fontsize=fontsize,
        fontweight="bold" if bold else "normal", ha="center", va="center", zorder=zorder + 1
    )


def _panel_header(ax, x, y, letter, title, subtitle=""):
    ax.text(x, y, letter, fontsize=15.5, fontweight="bold", color=INK, va="baseline", ha="left", zorder=10)
    ax.text(x + 0.34, y, title, fontsize=10.8, fontweight="bold", color=INK, va="baseline", ha="left", zorder=10)
    if subtitle:
        ax.text(x + 0.34, y - 0.27, subtitle, fontsize=8.0, color=MUTED, va="baseline", ha="left", zorder=10)


def _render_mol_exact(mapped_smi: str, active_maps: set, width_px: int, height_px: int):
    mol = _keep_reacting_frags(mapped_smi, active_maps)
    if mol is None:
        return Image.new("RGBA", (width_px, height_px), (255, 255, 255, 0)), {}
    idx_to_map = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DCairo(width_px, height_px)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.9
    opts.minFontSize = 13
    opts.maxFontSize = 16
    opts.padding = 0.12
    opts.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    coords = {mnum: (float(drawer.GetDrawCoords(idx).x), float(drawer.GetDrawCoords(idx).y)) for idx, mnum in idx_to_map.items()}
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    return img, coords


def _pick_hero_reaction(groups_dir: Path):
    best = None
    for sub in sorted(d for d in groups_dir.iterdir() if d.is_dir()):
        r = _load_group_report(sub)
        if not r or r.get("status") != "stock_solved":
            continue
        for rxn in r.get("reactions") or []:
            st, tr = _extract_subreaction_states(rxn)
            n_flows = sum(1 for t in tr if t[0].startswith("FLOW"))
            if len(st) == 4 and n_flows >= 2:
                n_arr = sum(len(_compute_curved_arrows(st[i][0], st[i + 1][0])) for i in range(len(tr)) if tr[i][0].startswith("FLOW"))
                mol = Chem.MolFromSmiles(str(rxn.get("product", "")))
                ha = mol.GetNumHeavyAtoms() if mol else 999
                if 13 <= ha <= 21 and n_arr >= 4:
                    score = (n_arr, -abs(ha - 16))
                    if best is None or score > best[0]:
                        best = (score, rxn, st, tr)
    return best[1:] if best else None


def build_figure1(groups_dir: Path, out_png: Path, out_pdf: Path):
    FIG_W, FIG_H = 18.2, 12.0  # True physical inches, 1:1 aspect ratio
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "text.color": INK,
    })

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================================
    # PANEL a (Top-Left Wide Hero: x=0.35..11.85, y=7.45..11.65)
    # Paradigm Shift: Reaction-Level Black-Box vs Proof-Carrying Electron Flow
    # =========================================================================
    _panel_header(ax, 0.40, 11.40, "a", "From reaction-level black-box guessing to executable electron-flow compiling",
                  "Physical conservation laws compiled directly into sub-reaction planning states (Chemical Lean)")

    # Upper muted strip: Reaction-Level Black-Box
    _wash_box(ax, 0.42, 10.22, 11.25, 0.78, SLATE, alpha=0.07, radius=0.14)
    ax.text(0.62, 10.61, "Conventional\nReaction-Level AI", fontsize=8.3, fontweight="bold", color=MUTED, va="center")
    _soft_pill(ax, 3.15, 10.61, 1.55, 0.42, "Target SMILES", SLATE, fill_alpha=0.14, text_color=INK, fontsize=8.2)
    ax.add_patch(FancyArrowPatch((4.00, 10.61), (5.25, 10.61), arrowstyle="-|>,head_length=5.5,head_width=3.2", color=SLATE, lw=1.4, ls="--", zorder=4))
    _soft_pill(ax, 6.40, 10.61, 2.10, 0.44, "Black-Box Leap (Template / LLM)", CORAL, fill_alpha=0.11, text_color=CORAL, fontsize=8.0)
    ax.add_patch(FancyArrowPatch((7.52, 10.61), (8.65, 10.61), arrowstyle="-|>,head_length=5.5,head_width=3.2", color=SLATE, lw=1.4, ls="--", zorder=4))
    _soft_pill(ax, 9.60, 10.61, 1.70, 0.42, "Precursors (?)", SLATE, fill_alpha=0.14, text_color=INK, fontsize=8.2)
    ax.text(11.50, 10.61, "✗ No conservation\n✗ Implicit side-rxns", fontsize=7.5, color=CORAL, ha="right", va="center", linespacing=1.35)

    # Lower hero wash: Retro-E Real ACS Mechanism
    _wash_box(ax, 0.42, 7.42, 11.25, 2.66, TEAL, alpha=0.055, radius=0.16)
    hero = _pick_hero_reaction(groups_dir)
    if hero is not None:
        rxn, states, transitions = hero
        all_active = set()
        for i in range(len(states) - 1):
            all_active.update(_changed_maps(states[i][0], states[i + 1][0])[0])
        n_st = len(states)
        slot_w, slot_h = 2.18, 1.62
        gap_x = 0.68
        sx0 = 0.70
        y_bot, y_top = 8.10, 8.10 + slot_h
        y_mid = (y_bot + y_top) / 2.0
        w_px, h_px = int(slot_w * 240), int(slot_h * 240)

        for i, (smi, _, kind) in enumerate(states):
            x0 = sx0 + i * (slot_w + gap_x)
            x1 = x0 + slot_w
            xc = (x0 + x1) / 2.0
            if kind == "interm":
                _wash_box(ax, x0 + 0.04, y_bot + 0.05, slot_w - 0.08, slot_h - 0.10, AMBER, alpha=0.09, radius=0.12, zorder=1)
                _draw_bracket(ax, x0 + 0.06, x1 - 0.06, y_bot + 0.08, y_top - 0.08, lw=1.3, color=INK)
            elif kind == "final":
                _wash_box(ax, x0 + 0.04, y_bot + 0.05, slot_w - 0.08, slot_h - 0.10, TEAL, alpha=0.10, radius=0.12, zorder=1)

            img, coords = _render_mol_exact(smi, all_active, w_px, h_px)
            ax.imshow(img, extent=[x0, x1, y_bot, y_top], aspect="equal", zorder=2)

            if i < len(transitions) and transitions[i][0].startswith("FLOW"):
                arrows = _compute_curved_arrows(states[i][0], states[i + 1][0])
                for ar_idx, ar in enumerate(arrows):
                    def px_to_in(pt):
                        return (x0 + (pt[0] / w_px) * slot_w, y_top - (pt[1] / h_px) * slot_h)
                    if ar["src_type"] == "atom" and ar["src"] in coords:
                        p_src = px_to_in(coords[ar["src"]])
                        ax.scatter([p_src[0]], [p_src[1] + 0.06], color=CRIMSON, s=11, zorder=8)
                    elif ar["src_type"] == "bond" and ar["src"][0] in coords and ar["src"][1] in coords:
                        c1, c2 = coords[ar["src"][0]], coords[ar["src"][1]]
                        p_src = px_to_in(((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2))
                    else:
                        continue
                    if ar["dst_type"] == "atom" and ar["dst"] in coords:
                        p_dst = px_to_in(coords[ar["dst"]])
                    elif ar["dst_type"] == "bond" and ar["dst"][0] in coords and ar["dst"][1] in coords:
                        c1, c2 = coords[ar["dst"][0]], coords[ar["dst"][1]]
                        p_dst = px_to_in(((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2))
                    else:
                        continue
                    if math.hypot(p_dst[0] - p_src[0], p_dst[1] - p_src[1]) < 0.04:
                        continue
                    rad = 0.40 if ar_idx % 2 == 0 else -0.40
                    ax.add_patch(FancyArrowPatch(
                        p_src, p_dst, connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-|>,head_length=4.5,head_width=2.6",
                        color=CRIMSON, lw=1.45, shrinkA=3, shrinkB=4, zorder=9
                    ))

            cap = "S₀ Target" if kind == "target" else ("S₃ In-Stock Precursors" if kind == "final" else f"[S{i}] Transient State")
            ax.text(xc, 7.86, cap, fontsize=7.9, fontweight="bold", color=TEAL if kind == "final" else INK, ha="center")

            if i < len(transitions):
                op_b, op_d, _ = transitions[i]
                al, ar_x = x1 + 0.08, x1 + gap_x - 0.08
                xm = (al + ar_x) / 2.0
                if op_b == "IMPORT":
                    ax.add_patch(FancyArrowPatch((al, y_mid), (ar_x, y_mid), arrowstyle="-|>,head_length=5,head_width=3", color=BLUE, lw=1.35, zorder=5))
                    ax.text(xm, y_mid + 0.14, op_d, fontsize=7.6, fontweight="bold", color=BLUE, ha="center", va="bottom")
                    ax.text(xm, y_mid - 0.14, "IMPORT", fontsize=6.9, color=MUTED, ha="center", va="top")
                else:
                    _draw_retro_double_arrow(ax, al, ar_x, y_mid, lw=1.25, color=INK)
                    ax.text(xm, y_mid + 0.14, op_b.replace("FLOW", "e⁻-Flow"), fontsize=7.5, fontweight="bold", color=CRIMSON, ha="center", va="bottom")

        # Bottom proof certificate bar (borderless soft pill)
        _soft_pill(ax, 6.05, 7.58, 10.6, 0.26,
                   "Proof-Carrying Synthesis Certificate:    ΔMass = 0   ·   ΔCharge = 0   ·   Δe⁻ = 0   ·   Machine-Verified Electron Program",
                   TEAL, fill_alpha=0.14, text_color=TEAL, fontsize=7.8)

    # =========================================================================
    # PANEL b (Top-Right: x=12.15..17.85, y=7.45..11.65)
    # Online Bidirectional PR18 Guardrail + Independent Dual Cross-Validation
    # =========================================================================
    _panel_header(ax, 12.15, 11.40, "b", "Bidirectional guardrail & dual audit",
                  "Online PR18 reversibility + zero-self-judging eval")
    _wash_box(ax, 12.15, 7.42, 5.65, 3.58, BLUE, alpha=0.055, radius=0.16)

    _soft_pill(ax, 13.40, 10.15, 1.85, 0.50, "State  S_before", BLUE, fill_alpha=0.14, text_color=INK, fontsize=8.4)
    _soft_pill(ax, 16.55, 10.15, 1.85, 0.50, "State  S_after", TEAL, fill_alpha=0.15, text_color=INK, fontsize=8.4)
    # Upper inverse arrow
    ax.add_patch(FancyArrowPatch((14.36, 10.32), (15.58, 10.32), connectionstyle="arc3,rad=-0.20",
                                 arrowstyle="-|>,head_length=5.5,head_width=3.2", color=BLUE, lw=1.7, zorder=6))
    ax.text(14.97, 10.66, "Inverse  s_inv", fontsize=8.0, fontweight="bold", color=BLUE, ha="center")
    # Lower forward replay arrow
    ax.add_patch(FancyArrowPatch((15.58, 9.98), (14.36, 9.98), connectionstyle="arc3,rad=-0.20",
                                 arrowstyle="-|>,head_length=5.5,head_width=3.2", color=TEAL, lw=1.7, zorder=6))
    ax.text(14.97, 9.54, "Forward Replay  s_fwd", fontsize=8.0, fontweight="bold", color=TEAL, ha="center")

    # Online decision gate
    ax.add_patch(FancyArrowPatch((14.97, 9.36), (14.97, 8.88), arrowstyle="-|>,head_length=5,head_width=3", color=MUTED, lw=1.3, zorder=5))
    _soft_pill(ax, 14.97, 8.62, 4.95, 0.44, "Online CRN-JEPA (PR18):  Intercept irreversible / side-rxn steps", AMBER, fill_alpha=0.14, text_color=INK, fontsize=7.9)

    # Independent audit strip
    ax.add_patch(FancyArrowPatch((14.97, 8.36), (14.97, 7.98), arrowstyle="-|>,head_length=5,head_width=3", color=TEAL, lw=1.3, ls="--", zorder=5))
    _soft_pill(ax, 13.65, 7.72, 2.35, 0.40, "URSA ChemCensor-U3 (Solv-2)", TEAL, fill_alpha=0.15, text_color=TEAL, fontsize=7.6)
    _soft_pill(ax, 16.30, 7.72, 2.35, 0.40, "Frozen FlowER Forward Audit", BLUE, fill_alpha=0.14, text_color=BLUE, fontsize=7.6)

    # =========================================================================
    # PANEL c (Mid-Left: x=0.40..7.70, y=3.60..7.05)
    # Critical Electronic Bifurcation & Half-Step Backjump Debugger
    # =========================================================================
    _panel_header(ax, 0.40, 6.85, "c", "Electronic bifurcation & half-step backjump",
                  "One electron-pair shift flips global route fate; surgical recovery without full step discard")
    _wash_box(ax, 0.42, 3.55, 7.35, 2.92, AMBER, alpha=0.055, radius=0.16)

    # Tree nodes: Root -> [S*] -> Branch A (cycle) & Branch B (closed)
    _soft_pill(ax, 1.35, 5.05, 1.40, 0.48, "Target S₀", INK, fill_alpha=0.10, text_color=INK, fontsize=8.4)
    ax.add_patch(FancyArrowPatch((2.08, 5.05), (2.78, 5.05), arrowstyle="-|>,head_length=5.5,head_width=3.2", color=INK, lw=1.5, zorder=5))
    _soft_pill(ax, 3.65, 5.05, 1.65, 0.56, "Critical Fork\n[S*] (1/2 step)", AMBER, fill_alpha=0.18, text_color=AMBER, fontsize=8.1)

    # Upper Branch A (FAIL)
    ax.add_patch(FancyArrowPatch((4.50, 5.20), (5.38, 5.88), connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>,head_length=5,head_width=3", color=CORAL, lw=1.6, zorder=5))
    ax.text(4.82, 5.76, "e⁻-move A", fontsize=7.5, fontweight="bold", color=CORAL, ha="center")
    _soft_pill(ax, 6.45, 5.95, 2.05, 0.46, "✗ ROUTE_CYCLE / Dead-End", CORAL, fill_alpha=0.14, text_color=CORAL, fontsize=7.9)

    # Surgical Backjump Arc from Branch A back to [S*]
    ax.add_patch(FancyArrowPatch((6.10, 5.68), (3.85, 5.38), connectionstyle="arc3,rad=0.32",
                                 arrowstyle="-|>,head_length=6,head_width=3.5", color=AMBER, lw=2.0, ls="--", zorder=7))
    ax.text(5.12, 5.08, "Electronic Backjump\n(keep valid prefix)", fontsize=7.5, fontweight="bold", color=AMBER, ha="center", va="center")

    # Lower Branch B (WIN)
    ax.add_patch(FancyArrowPatch((4.50, 4.88), (5.38, 4.22), connectionstyle="arc3,rad=0.12",
                                 arrowstyle="-|>,head_length=5.5,head_width=3.2", color=TEAL, lw=1.9, zorder=5))
    ax.text(4.80, 4.24, "e⁻-move B", fontsize=7.5, fontweight="bold", color=TEAL, ha="center")
    _soft_pill(ax, 6.45, 4.15, 2.05, 0.48, "✓ All-Stock Closed Tree", TEAL, fill_alpha=0.16, text_color=TEAL, fontsize=8.0)

    # =========================================================================
    # PANEL d (Mid-Center Compact: x=8.05..12.55, y=3.60..7.05)
    # Multi-Objective Bottleneck Gating
    # =========================================================================
    _panel_header(ax, 8.05, 6.85, "d", "Bottleneck reward gating",
                  "Eliminates stock-shortcut hacking")
    _wash_box(ax, 8.05, 3.55, 4.55, 2.92, CORAL, alpha=0.048, radius=0.16)

    gates = [
        (6.05, "G_mech:  Weakest-Link PR18  min_e s(e)", TEAL),
        (5.42, "G_stereo:  Spectator Stereocenter Lock", BLUE),
        (4.79, "φ_LLS:  Convergent Bonus & Step Economy", AMBER),
        (4.16, "− C_conflict:  2.5× Cycle / Repeat Penalty", CORAL),
    ]
    for gy, gtxt, gcol in gates:
        _soft_pill(ax, 10.32, gy, 4.15, 0.44, gtxt, gcol, fill_alpha=0.13, text_color=INK, fontsize=7.9)
    ax.text(10.32, 3.74, "R(τ) = I[closed] · G_mech · G_stereo · φ_LLS − C_conflict",
            fontsize=7.6, fontweight="bold", color=MUTED, ha="center")

    # =========================================================================
    # PANEL e (Mid-Right: x=12.90..17.85, y=3.60..7.05)
    # Long-Horizon ReFIT-PO & Three-Stage Curriculum
    # =========================================================================
    _panel_header(ax, 12.90, 6.85, "e", "Long-horizon ReFIT-PO",
                  "100% full non-stock depth ≥ 2 RL")
    _wash_box(ax, 12.90, 3.55, 4.90, 2.92, BLUE, alpha=0.055, radius=0.16)

    _soft_pill(ax, 14.05, 5.95, 1.80, 0.44, "Stage-II SFT", SLATE, fill_alpha=0.15, text_color=INK, fontsize=8.0)
    ax.add_patch(FancyArrowPatch((14.98, 5.95), (15.55, 5.95), arrowstyle="-|>,head_length=5,head_width=3", color=MUTED, lw=1.4))
    _soft_pill(ax, 16.58, 5.95, 1.98, 0.44, "Short-Horizon (204)", AMBER, fill_alpha=0.14, text_color=INK, fontsize=8.0)
    ax.add_patch(FancyArrowPatch((16.58, 5.70), (16.58, 5.18), arrowstyle="-|>,head_length=5.5,head_width=3.2", color=TEAL, lw=1.7))
    _soft_pill(ax, 15.35, 4.86, 4.35, 0.54, "Retro-E-Plan  (124k Non-Stock · 1:1:1 Depth 2/3/4+ · T=0.80)", TEAL, fill_alpha=0.18, text_color=TEAL, fontsize=7.8)
    # KL anchor back to SFT
    ax.add_patch(FancyArrowPatch((13.65, 5.14), (13.65, 5.70), arrowstyle="-|>,head_length=4.5,head_width=2.8", color=BLUE, lw=1.3, ls=":"))
    ax.text(13.98, 5.42, "KL Anchor", fontsize=7.2, color=BLUE, ha="left", va="center")
    # Same-state contrast bar
    _soft_pill(ax, 15.35, 4.02, 4.35, 0.44, "Same-State Contrast:  A_ReFIT(s, a_win) > A_ReFIT(s, a_lose)", BLUE, fill_alpha=0.12, text_color=INK, fontsize=7.6)

    # =========================================================================
    # PANEL f (Bottom-Left Wide: x=0.40..10.55, y=0.35..3.25)
    # Training-Free Inference-Time Multi-Objective Pareto Steering
    # =========================================================================
    _panel_header(ax, 0.40, 3.05, "f", "Training-free multi-objective Pareto steering",
                  "One frozen Retro-E-Plan model adapts across MedChem, Process, and Supply-Chain constraints")
    _wash_box(ax, 0.42, 0.32, 10.15, 2.32, TEAL, alpha=0.055, radius=0.16)

    # Left: Frozen model hub -> 3 application branches
    _soft_pill(ax, 1.85, 1.48, 2.25, 0.62, "Frozen Retro-E-Plan\n+ C2 Electronic Tree", INK, fill_alpha=0.11, text_color=INK, fontsize=8.3)
    branches = [
        (2.15, "MedChem Mode:  Min LLS + SAR-40 Common Intermediate", TEAL),
        (1.48, "Process Mode:  Exact Atom Economy + Green Compliance", BLUE),
        (0.81, "Supply Resilience:  Blocked-BB Automatic Rerouting", AMBER),
    ]
    for by, btxt, bcol in branches:
        ax.add_patch(FancyArrowPatch((3.02, 1.48), (3.72, by), connectionstyle="arc3,rad=0.0",
                                     arrowstyle="-|>,head_length=5,head_width=3", color=bcol, lw=1.5))
        _soft_pill(ax, 5.75, by, 3.95, 0.44, btxt, bcol, fill_alpha=0.13, text_color=INK, fontsize=7.7)

    # Mini stylized Pareto curve on right side of Panel f
    px0, py0, pw, ph = 8.15, 0.65, 2.15, 1.65
    ax.plot([px0, px0 + pw], [py0, py0], color=SLATE, lw=0.9)
    ax.plot([px0, px0], [py0, py0 + ph], color=SLATE, lw=0.9)
    t = np.linspace(0.15, 0.92, 30)
    cx = px0 + t * pw
    cy = py0 + (1.0 - 0.78 * (t ** 1.7)) * ph
    ax.plot(cx, cy, color=CORAL, lw=2.0, zorder=5)
    ax.scatter([px0 + 0.25 * pw, px0 + 0.80 * pw], [py0 + 0.88 * ph, py0 + 0.42 * ph],
               color=[TEAL, BLUE], s=55, edgecolors="white", lw=0.8, zorder=6)
    ax.text(px0 + 0.30 * pw, py0 + 0.90 * ph, "Process", fontsize=7.2, fontweight="bold", color=TEAL)
    ax.text(px0 + 0.66 * pw, py0 + 0.26 * ph, "MedChem", fontsize=7.2, fontweight="bold", color=BLUE)
    ax.text(px0 + pw / 2, py0 - 0.16, "Step Economy (1/LLS)", fontsize=7.1, color=MUTED, ha="center")
    ax.text(px0 - 0.14, py0 + ph / 2, "Exact AE & PR18", fontsize=7.1, color=MUTED, va="center", ha="center", rotation=90)

    # =========================================================================
    # PANEL g (Bottom-Right: x=10.90..17.85, y=0.35..3.25)
    # Mechanistic Synthesizability Surgeon (AI Dark Space Co-Design)
    # =========================================================================
    _panel_header(ax, 10.90, 3.05, "g", "Mechanistic synthesizability surgeon",
                  "Diagnose blocked electronic step → minimal edit → Solv-2 closure")
    _wash_box(ax, 10.90, 0.32, 6.90, 2.32, CORAL, alpha=0.052, radius=0.16)

    _soft_pill(ax, 12.15, 1.52, 2.05, 0.78, "AI De Novo Molecule\n(Dark Space · Unsolved)", CORAL, fill_alpha=0.13, text_color=CORAL, fontsize=7.9)
    ax.add_patch(FancyArrowPatch((13.22, 1.52), (13.78, 1.52), arrowstyle="-|>,head_length=5,head_width=3", color=AMBER, lw=1.6))
    _soft_pill(ax, 14.70, 1.52, 1.75, 0.78, "Pinpoint Blocked\nElectron Step\n(Minimal 1-Atom Edit)", AMBER, fill_alpha=0.15, text_color=INK, fontsize=7.5)
    ax.add_patch(FancyArrowPatch((15.62, 1.52), (16.12, 1.52), arrowstyle="-|>,head_length=5.5,head_width=3.2", color=TEAL, lw=1.7))
    _soft_pill(ax, 16.92, 1.52, 1.48, 0.78, "✓ 3-Step\nSolv-2 Route\n(Repaired)", TEAL, fill_alpha=0.16, text_color=TEAL, fontsize=7.8)
    ax.text(14.35, 0.62, "Closes the loop between de novo generative drug design and executable synthesis planning",
            fontsize=7.6, style="italic", color=MUTED, ha="center")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(out_png), "pdf": str(out_pdf)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, required=True)
    args = parser.parse_args()
    build_figure1(args.groups_dir, args.out_png, args.out_pdf)


if __name__ == "__main__":
    main()
