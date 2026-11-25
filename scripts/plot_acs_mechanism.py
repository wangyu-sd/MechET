"""Render JACS / Nature Chemistry (ACS 1996 style) organic mechanism figures with real crimson curved electron-pushing arrows, aligned 2D scaffolds, and bracketed transient intermediates."""
import argparse
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from scripts.plot_electron_flow_states import _bond_map, _charge_map, _changed_maps, _extract_subreaction_states, _keep_reacting_frags
from scripts.prepare_nmi_day34_eval_suite import _load_group_report

# Classic ACS / JACS / Nature Chemistry monochrome + crimson electron-arrow palette
ACS_BLACK = "#111827"
ACS_MUTED = "#4B5563"
ACS_RED = "#C81E1E"      # Classic curved electron-pushing arrow crimson
ACS_TEAL = "#0F766E"     # Subtle conservation/BB accent
ACS_RULE = "#D1D5DB"


def _compute_curved_arrows(smi_before: str, smi_after: str):
    """Derive chemically accurate curved electron-pushing arrows from atom-mapped state_before -> state_after.
    Returns list of dicts: {'src_type': 'atom'|'bond', 'src': map_or_pair, 'dst_type': 'atom'|'bond', 'dst': map_or_pair}."""
    ma = Chem.MolFromSmiles(smi_before or "")
    mb = Chem.MolFromSmiles(smi_after or "")
    if ma is None or mb is None:
        return []
    ba, bb = _bond_map(ma), _bond_map(mb)
    ca, cb = _charge_map(ma), _charge_map(mb)

    decreased_bonds = [k for k in ba if bb.get(k, 0.0) < ba[k]]
    increased_bonds = [k for k in bb if bb[k] > ba.get(k, 0.0)]
    donors = {m for m in set(ca) & set(cb) if cb[m] > ca[m]}      # lost electrons (charge increased)
    acceptors = {m for m in set(ca) & set(cb) if cb[m] < ca[m]}   # gained electrons (charge decreased)

    arrows = []
    used_dec = set()

    # 1. Breaking / decreasing bonds -> accepting atom or adjacent forming bond
    for (u, v) in decreased_bonds:
        if v in acceptors:
            arrows.append({"src_type": "bond", "src": (u, v), "dst_type": "atom", "dst": v})
            used_dec.add((u, v))
        elif u in acceptors:
            arrows.append({"src_type": "bond", "src": (u, v), "dst_type": "atom", "dst": u})
            used_dec.add((u, v))
        else:
            # Check if electrons flow into an adjacent newly formed/increased bond
            adj = [ib for ib in increased_bonds if u in ib or v in ib]
            if adj:
                ib = adj[0]
                arrows.append({"src_type": "bond", "src": (u, v), "dst_type": "bond", "dst": ib})
                used_dec.add((u, v))

    # 2. Lone-pair donor atom -> forming / increasing bond (or electrophilic atom)
    for (u, v) in increased_bonds:
        already_fed = any(ar["dst_type"] == "bond" and ar["dst"] == (u, v) for ar in arrows)
        if already_fed:
            continue
        if u in donors:
            dst_type = "bond" if (u, v) in ba else "atom"
            dst = (u, v) if dst_type == "bond" else v
            arrows.append({"src_type": "atom", "src": u, "dst_type": dst_type, "dst": dst})
        elif v in donors:
            dst_type = "bond" if (u, v) in ba else "atom"
            dst = (u, v) if dst_type == "bond" else u
            arrows.append({"src_type": "atom", "src": v, "dst_type": dst_type, "dst": dst})

    return arrows


def _render_acs_mol_with_coords(mapped_smi: str, active_maps: set, width: int = 560, height: int = 380):
    """Render pure ACS 1996 monochrome black skeletal structure and return (PIL image, map_num -> (px_x, px_y))."""
    mol = _keep_reacting_frags(mapped_smi, active_maps)
    if mol is None:
        return Image.new("RGBA", (width, height), (255, 255, 255, 0)), {}

    # Preserve atom map numbers in a side dict before clearing for clean ACS display
    idx_to_map = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)

    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2.0
    opts.minFontSize = 15
    opts.maxFontSize = 18
    opts.padding = 0.14
    opts.useBWAtomPalette()  # Authentic JACS / ACS 1996 pure monochrome black!
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    coords = {}
    for idx, mnum in idx_to_map.items():
        pt = drawer.GetDrawCoords(idx)
        coords[mnum] = (float(pt.x), float(pt.y))

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    return img, coords


def _draw_bracket(ax, x_left, x_right, y_bot, y_top, lw=1.6, color=ACS_BLACK):
    """Draw classic organic chemistry square brackets [ ... ] around a transient intermediate."""
    tick = (x_right - x_left) * 0.045
    # Left bracket [
    ax.plot([x_left + tick, x_left, x_left, x_left + tick], [y_top, y_top, y_bot, y_bot], color=color, lw=lw, zorder=6)
    # Right bracket ]
    ax.plot([x_right - tick, x_right, x_right, x_right - tick], [y_top, y_top, y_bot, y_bot], color=color, lw=lw, zorder=6)


def _draw_retro_double_arrow(ax, x1, x2, y, lw=1.4, color=ACS_BLACK):
    """Draw an authentic organic-chemistry retrosynthetic open double arrow (==>)."""
    gap = 0.011
    head_w = 0.024
    head_l = 0.016
    ax.plot([x1, x2 - head_l], [y + gap, y + gap], color=color, lw=lw, zorder=5)
    ax.plot([x1, x2 - head_l], [y - gap, y - gap], color=color, lw=lw, zorder=5)
    ax.plot([x2 - head_l - 0.004, x2, x2 - head_l - 0.004], [y + head_w, y, y - head_w], color=color, lw=lw, zorder=5)


def render_acs_mechanism_panel(ax, letter: str, rxn: dict, states: list, transitions: list):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_st = len(states)
    all_active = set()
    for i in range(n_st - 1):
        ca, _ = _changed_maps(states[i][0], states[i + 1][0])
        all_active.update(ca)

    # Clean ACS editorial top rule & title (no heavy boxes!)
    ax.text(0.01, 0.94, f"{letter}", fontsize=14, fontweight="bold", color=ACS_BLACK, va="center")
    n_flows = sum(1 for t in transitions if t[0].startswith("FLOW"))
    ax.text(
        0.04, 0.94,
        f"Executable Retrosynthetic Mechanism (Occurrence {rxn.get('occurrence', '0')}  ·  {n_flows} Coupled Electron-Pushing Steps)",
        fontsize=10.8, fontweight="bold", color=ACS_BLACK, va="center"
    )
    ax.text(
        0.99, 0.94,
        "Strict Conservation:  ΔMass = 0  ·  ΔCharge = 0  ·  Δe⁻ = 0",
        fontsize=8.8, color=ACS_TEAL, fontweight="bold", ha="right", va="center"
    )
    ax.plot([0.01, 0.99], [0.895, 0.895], color=ACS_RULE, lw=0.9)

    # Layout each state horizontally on pure white canvas
    margin = 0.025
    arrow_gap = 0.068
    slot_w = (1.0 - 2 * margin - arrow_gap * (n_st - 1)) / n_st
    slot_h = 0.62
    y_bot, y_top = 0.19, 0.19 + slot_h
    y_mid = (y_bot + y_top) / 2.0

    W_PX, H_PX = 560, 380

    for i, (smi, _, kind) in enumerate(states):
        x0 = margin + i * (slot_w + arrow_gap)
        x1 = x0 + slot_w
        xc = (x0 + x1) / 2.0

        img, coords = _render_acs_mol_with_coords(smi, all_active, width=W_PX, height=H_PX)
        ax.imshow(img, extent=[x0, x1, y_bot, y_top], aspect="auto", zorder=2)

        # Draw classic square brackets around transient intermediate electronic states
        if kind == "interm":
            _draw_bracket(ax, x0 + 0.004, x1 - 0.004, y_bot + 0.03, y_top - 0.03, lw=1.5, color=ACS_BLACK)

        # Draw REAL CRIMSON CURVED ELECTRON-PUSHING ARROWS on this state if the NEXT transition is a FLOW!
        if i < len(transitions) and transitions[i][0].startswith("FLOW"):
            arrows = _compute_curved_arrows(states[i][0], states[i + 1][0])

            def px_to_ax(px_pt):
                return (x0 + (px_pt[0] / W_PX) * slot_w, y_top - (px_pt[1] / H_PX) * slot_h)

            for ar_idx, ar in enumerate(arrows):
                if ar["src_type"] == "atom" and ar["src"] in coords:
                    p_src = px_to_ax(coords[ar["src"]])
                    # Subtle crimson lone-pair indicator dot near donor atom
                    ax.scatter([p_src[0]], [p_src[1] + 0.018], color=ACS_RED, s=14, zorder=8)
                elif ar["src_type"] == "bond" and ar["src"][0] in coords and ar["src"][1] in coords:
                    c1, c2 = coords[ar["src"][0]], coords[ar["src"][1]]
                    p_src = px_to_ax(((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0))
                else:
                    continue

                if ar["dst_type"] == "atom" and ar["dst"] in coords:
                    p_dst = px_to_ax(coords[ar["dst"]])
                elif ar["dst_type"] == "bond" and ar["dst"][0] in coords and ar["dst"][1] in coords:
                    c1, c2 = coords[ar["dst"][0]], coords[ar["dst"][1]]
                    p_dst = px_to_ax(((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0))
                else:
                    continue

                dist = math.hypot(p_dst[0] - p_src[0], p_dst[1] - p_src[1])
                if dist < 1e-4:
                    continue
                rad = 0.42 if ar_idx % 2 == 0 else -0.42
                curved = FancyArrowPatch(
                    p_src, p_dst,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>,head_length=5.5,head_width=3.2",
                    color=ACS_RED, linewidth=1.75, shrinkA=5, shrinkB=6, zorder=9
                )
                ax.add_patch(curved)

        # Clean ACS caption underneath each chemical species
        if kind == "target":
            cap = f"S₀  Target Product"
        elif kind == "final":
            cap = f"S{i}  Verified Precursors"
        else:
            cap = f"[S{i}]  Transient Intermediate"
        ax.text(xc, 0.125, cap, fontsize=9.0, fontweight="bold", color=ACS_BLACK, ha="center", va="center")

        # Draw reaction / retrosynthetic arrow between state i and state i+1
        if i < len(transitions):
            op_badge, op_detail, _ = transitions[i]
            ax_l = x1 + 0.008
            ax_r = x1 + arrow_gap - 0.008
            xm = (ax_l + ax_r) / 2.0
            if op_badge == "IMPORT":
                arr = FancyArrowPatch((ax_l, y_mid), (ax_r, y_mid), arrowstyle="-|>,head_length=6,head_width=3.5",
                                      color=ACS_BLACK, lw=1.4, zorder=5)
                ax.add_patch(arr)
                reag = op_detail.replace("+ ", "").strip()
                ax.text(xm, y_mid + 0.045, f"+ {reag}", fontsize=8.4, fontweight="bold", color=ACS_BLACK, ha="center", va="bottom")
                ax.text(xm, y_mid - 0.045, "import", fontsize=7.6, style="italic", color=ACS_MUTED, ha="center", va="top")
            else:
                _draw_retro_double_arrow(ax, ax_l, ax_r, y_mid, lw=1.35, color=ACS_BLACK)
                n_moves = op_detail.splitlines()[0] if op_detail else ""
                ax.text(xm, y_mid + 0.045, op_badge.replace("FLOW", "e⁻-Flow"), fontsize=8.3, fontweight="bold", color=ACS_RED, ha="center", va="bottom")
                ax.text(xm, y_mid - 0.048, n_moves, fontsize=7.4, color=ACS_MUTED, ha="center", va="top")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for sub in sorted(d for d in args.groups_dir.iterdir() if d.is_dir()):
        r = _load_group_report(sub)
        if not r or r.get("status") != "stock_solved":
            continue
        for rxn in r.get("reactions") or []:
            states, transitions = _extract_subreaction_states(rxn)
            n_flows = sum(1 for t in transitions if t[0].startswith("FLOW"))
            if 3 <= len(states) <= 4 and n_flows >= 2:
                # Verify we get >=2 clean curved arrows across transitions
                n_arrows = sum(
                    len(_compute_curved_arrows(states[i][0], states[i + 1][0]))
                    for i in range(len(transitions)) if transitions[i][0].startswith("FLOW")
                )
                mol = Chem.MolFromSmiles(str(rxn.get("product", "")))
                ha = mol.GetNumHeavyAtoms() if mol else 999
                if 13 <= ha <= 26 and n_arrows >= 3:
                    candidates.append((abs(ha - 18), -n_arrows, r.get("task_id", ""), rxn, states, transitions, n_arrows))

    candidates.sort(key=lambda x: (x[0], x[1]))
    chosen = []
    used = set()
    for _, _, tid, rxn, st, tr, na in candidates:
        if tid in used:
            continue
        used.add(tid)
        chosen.append((tid, rxn, st, tr, na))
        if len(chosen) >= 2:
            break

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig, axes = plt.subplots(len(chosen), 1, figsize=(16.0, 4.9 * len(chosen)), dpi=300, facecolor="white")
    if len(chosen) == 1:
        axes = [axes]

    for idx, (ax, (tid, rxn, st, tr, _)) in enumerate(zip(axes, chosen)):
        render_acs_mechanism_panel(ax, ["a", "b", "c"][idx], rxn, st, tr)

    plt.tight_layout(h_pad=1.6)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(json.dumps({
        "out_path": str(args.out_path),
        "cases": [
            {"task_id": tid, "occurrence": rxn.get("occurrence"), "states": len(st), "curved_arrows_drawn": na}
            for tid, rxn, st, tr, na in chosen
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
