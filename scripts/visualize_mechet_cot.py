#!/usr/bin/env python3
"""Publication-style MechET CoT figure (RDKit + Pillow).

Example:
  PYTHONPATH=src python scripts/visualize_mechet_cot.py \\
    --id flower_mech_et_val_496 \\
    --out docs/mechet_cot_example.png
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.mech_et import parse_mech_et_body
from mechet.sft import parse_mech_cot_output

# Publication palette
INK = (22, 22, 22)
MUTED = (95, 95, 95)
RULE = (210, 210, 210)
PANEL = (246, 247, 249)
ACCENT_RGB = (196, 72, 48)
TEAL = (16, 110, 110)


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = {
        "serif": [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        ],
        "serif_b": [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        ],
        "sans": [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "sans_b": [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "mono": [
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ],
    }
    for p in paths.get(kind, paths["sans"]):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _mol_from_mapped(smi: str):
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = True
    mol = Chem.MolFromSmiles(smi, ps)
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    return mol


def _frags_by_weight(smi: str) -> list[str]:
    frags = [p for p in smi.split(".") if p.strip()]
    scored = []
    for frag in frags:
        mol = _mol_from_mapped(frag)
        sc = mol.GetNumHeavyAtoms() if mol is not None else len(frag)
        scored.append((sc, frag))
    scored.sort(reverse=True)
    return [f for _, f in scored]


def _atom_idx_by_maps(mol, maps: set[int]) -> list[int]:
    if mol is None:
        return []
    out = []
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber") and int(atom.GetIntProp("molAtomMapNumber")) in maps:
            out.append(atom.GetIdx())
    return out


def _rgba_to_png(drawer) -> Image.Image:
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def _whitish_to_transparent(img: Image.Image, thresh: int = 248) -> Image.Image:
    """Keep structure pixels; punch out near-white background."""
    img = img.convert("RGBA")
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if r >= thresh and g >= thresh and b >= thresh:
                px[x, y] = (255, 255, 255, 0)
    return img


def _draw_mol(
    mol,
    size=(380, 280),
    highlight_atoms=None,
    highlight_bonds=None,
    show_maps: bool = False,
    # 透明度大一些：alpha 更低
    highlight_alpha: int = 75,
):
    w, h = size
    if mol is None:
        img = Image.new("RGB", size, (255, 255, 255))
        ImageDraw.Draw(img).text((16, h // 2), "parse failed", fill=ACCENT_RGB, font=_font("sans", 14))
        return img

    highlight_atoms = list(highlight_atoms or [])
    highlight_bonds = list(highlight_bonds or [])

    # Draw on a copy so we can strip map numbers without breaking callers
    draw_mol = Chem.Mol(mol)
    for atom in draw_mol.GetAtoms():
        if atom.HasProp("atomNote"):
            atom.ClearProp("atomNote")
        if atom.HasProp("molAtomMapNumber"):
            if show_maps:
                atom.SetProp("atomNote", str(atom.GetIntProp("molAtomMapNumber")))
            atom.ClearProp("molAtomMapNumber")

    # Pass 1: draw structure (no highlight) and collect atom screen coords
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 7.0  # bold bonds
    opts.scaleBondWidth = False
    opts.multipleBondOffset = 0.24
    opts.additionalAtomLabelPadding = 0.02
    opts.fixedBondLength = 26
    opts.minFontSize = 18
    opts.maxFontSize = 26
    opts.baseFontSize = 0.85
    bold_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if Path(bold_font).exists():
        opts.fontFile = bold_font
    opts.noAtomLabels = False
    opts.includeAtomTags = False
    opts.addAtomIndices = False
    opts.isotopeLabels = False
    opts.dummyIsotopeLabels = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, draw_mol)
    drawer.FinishDrawing()
    struct = _whitish_to_transparent(_rgba_to_png(drawer))

    # coords from draw_mol (same atom indices as mol)
    atom_xy = {}
    for idx in range(draw_mol.GetNumAtoms()):
        p = drawer.GetDrawCoords(idx)
        atom_xy[idx] = (p.x, p.y)

    # Pass 2: soft highlight underlay (higher transparency)
    under = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    ud = ImageDraw.Draw(under)
    fill = (*ACCENT_RGB, min(255, highlight_alpha))
    fill_bond = (170, 45, 35, min(255, highlight_alpha))

    for bidx in highlight_bonds:
        bond = mol.GetBondWithIdx(bidx)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in atom_xy and b in atom_xy:
            ud.line([atom_xy[a], atom_xy[b]], fill=fill_bond, width=20)

    r = 17
    for aidx in highlight_atoms:
        if aidx not in atom_xy:
            continue
        x, y = atom_xy[aidx]
        ud.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    composed = Image.alpha_composite(under, struct)
    return composed.convert("RGB")


def _bond_idxs(mol, map_pairs: list[tuple[int, int]]) -> list[int]:
    if mol is None:
        return []
    map_to_idx = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            map_to_idx[int(atom.GetIntProp("molAtomMapNumber"))] = atom.GetIdx()
    out = []
    for a, b in map_pairs:
        if a not in map_to_idx or b not in map_to_idx:
            continue
        bond = mol.GetBondBetweenAtoms(map_to_idx[a], map_to_idx[b])
        if bond is not None:
            out.append(bond.GetIdx())
    return out


def _panel(draw: ImageDraw.ImageDraw, box, label: str, title: str):
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=RULE, width=1)
    # panel tag
    tag = f"({label})"
    draw.text((x0 + 14, y0 + 10), tag, fill=INK, font=_font("serif_b", 18))
    draw.text((x0 + 52, y0 + 12), title, fill=INK, font=_font("serif", 17))


def _hline(draw, x0, x1, y, color=RULE, width=1):
    draw.line((x0, y, x1, y), fill=color, width=width)


def _arrow(draw, x0, y, x1, color=INK, width=2):
    draw.line((x0, y, x1, y), fill=color, width=width)
    draw.polygon([(x1, y), (x1 - 10, y - 6), (x1 - 10, y + 6)], fill=color)


def _fmt_delta(delta) -> list[str]:
    rows = []
    for i, j, d in delta.bonds:
        sign = f"+{d}" if d > 0 else str(d)
        rows.append(("BOND", f"{i}–{j}", sign))
    for i, d in delta.lone_pairs:
        sign = f"+{d}" if d > 0 else str(d)
        rows.append(("LP", str(i), sign))
    for i, q0, q1 in delta.charges:
        rows.append(("CHARGE", str(i), f"{q0} → {q1}"))
    return rows


def _pick_row(path: Path, sample_id: str = "") -> dict:
    if sample_id:
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") == sample_id or sample_id in line:
                return row
        raise SystemExit(f"id not found: {sample_id}")

    scored = []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        body = parse_mech_cot_output(row["messages"][-1]["content"])["mechanism"]
        graph = parse_mech_et_body(body)
        if not graph.get("ok"):
            continue
        sig = (graph.get("et_signature") or "unknown").lower()
        n_s, n_e = len(graph["states"]), len(graph["retro_edges"])
        score = (
            0 if sig in ("unknown", "", "none") else 1,
            1 if n_s == 2 and n_e == 1 else 0,
            1 if graph.get("endpoints") else 0,
            -n_s,
            -n_e,
        )
        scored.append((score, row))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def render(row: dict, out: Path, dpi_note: bool = True) -> Path:
    user = row["messages"][1]["content"]
    asst = row["messages"][2]["content"]
    product = user.split("\n", 1)[0].replace("TARGET:", "").strip()
    parsed = parse_mech_cot_output(asst)
    graph = parse_mech_et_body(parsed["mechanism"])
    if not graph.get("ok"):
        raise SystemExit(f"bad MECH_ET body: {graph.get('diagnostics')}")

    answer = parsed["answer"]
    src, dst = graph["retro_edges"][0]
    delta = graph["edge_deltas"][(src, dst)]
    active_maps = set()
    bond_pairs = []
    for i, j, _ in delta.bonds:
        active_maps.update((i, j))
        bond_pairs.append((i, j))
    for i, _ in delta.lone_pairs:
        active_maps.add(i)
    for i, _, _ in delta.charges:
        active_maps.add(i)

    shared = graph.get("shared") or ""
    topo = (row.get("metadata") or {}).get("topology", "linear")
    sig = graph.get("et_signature") or "unknown"
    demand = graph.get("et_demand") or "unknown"
    endpoints = graph.get("endpoints") or []
    centers = graph.get("centers") or []

    # Molecules: product + both edge endpoints + answer components
    mol_prod = _mol_from_mapped(_frags_by_weight(product)[0])
    mol_s0 = _mol_from_mapped(_frags_by_weight(graph["states"][src])[0])
    # For s1, draw the two heaviest organic frags if present
    s1_frags = _frags_by_weight(graph["states"][dst])
    mol_s1a = _mol_from_mapped(s1_frags[0]) if s1_frags else None
    mol_s1b = _mol_from_mapped(s1_frags[1]) if len(s1_frags) > 1 else None

    ans_frags = _frags_by_weight(answer)
    # Prefer organic reactants (skip tiny salts if we have space)
    ans_mols = []
    for frag in ans_frags:
        mol = _mol_from_mapped(frag)
        if mol is None:
            continue
        # skip lone Na+
        if mol.GetNumHeavyAtoms() <= 1 and mol.GetAtomWithIdx(0).GetSymbol() in ("Na", "K", "Li"):
            continue
        ans_mols.append(mol)
        if len(ans_mols) >= 3:
            break

    hl_prod = _atom_idx_by_maps(mol_prod, active_maps)
    hl_s0 = _atom_idx_by_maps(mol_s0, active_maps)
    hb_s0 = _bond_idxs(mol_s0, bond_pairs)
    hl_s1a = _atom_idx_by_maps(mol_s1a, active_maps)
    hl_s1b = _atom_idx_by_maps(mol_s1b, active_maps)

    img_prod = _draw_mol(mol_prod, size=(420, 300), highlight_atoms=hl_prod, highlight_bonds=_bond_idxs(mol_prod, bond_pairs))
    img_s0 = _draw_mol(mol_s0, size=(360, 260), highlight_atoms=hl_s0, highlight_bonds=hb_s0)
    img_s1a = _draw_mol(mol_s1a, size=(280, 220), highlight_atoms=hl_s1a)
    img_s1b = _draw_mol(mol_s1b, size=(280, 220), highlight_atoms=hl_s1b) if mol_s1b else None
    img_ans = [_draw_mol(m, size=(300, 220)) for m in ans_mols]

    # Canvas
    W, H = 1680, 1180
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title block
    draw.text((48, 28), "MechET", fill=TEAL, font=_font("sans_b", 22))
    draw.text(
        (140, 30),
        "Structured mechanism chain-of-thought for retrosynthesis",
        fill=INK,
        font=_font("serif", 22),
    )
    _hline(draw, 48, W - 48, 68, color=(40, 40, 40), width=1)
    meta = (
        f"Example  {row.get('id')}    ·    topology {topo}    ·    "
        f"{len(graph['states'])} states / {len(graph['retro_edges'])} edge"
        f"    ·    BE units: single bond = 1"
    )
    draw.text((48, 78), meta, fill=MUTED, font=_font("sans", 14))

    # -------- Panel a: input --------
    a = (48, 115, 560, 470)
    _panel(draw, a, "a", "Input — target product")
    canvas.paste(img_prod, (a[0] + 70, a[1] + 50))
    draw.text((a[0] + 16, a[1] + 360), "User prompt exposes only TARGET SMILES (no map labels drawn).", fill=MUTED, font=_font("sans", 13))
    draw.text((a[0] + 16, a[1] + 382), "Soft coral highlight = atoms / bonds in BE_DELTA.", fill=MUTED, font=_font("sans", 13))

    # -------- Panel b: perceive / ET --------
    b = (580, 115, 1632, 470)
    _panel(draw, b, "b", "Perceive electronic context  →  name inverse ET signature")
    # two-column content
    left_x, top_y = b[0] + 28, b[1] + 55
    draw.text((left_x, top_y), "PERCEIVE", fill=TEAL, font=_font("sans_b", 15))
    y = top_y + 28
    if endpoints:
        for ep in endpoints[:4]:
            maps = ",".join(map(str, ep.get("maps") or []))
            draw.text((left_x, y), f"ENDPOINT   {ep.get('label')}", fill=INK, font=_font("mono", 14))
            draw.text((left_x + 260, y), f"maps = {{{maps}}}", fill=MUTED, font=_font("mono", 14))
            y += 24
    else:
        draw.text((left_x, y), "(no endpoints listed)", fill=MUTED, font=_font("sans", 14))
        y += 24
    y += 8
    for c in centers[:4]:
        draw.text((left_x, y), f"CENTER     {c}", fill=INK, font=_font("mono", 14))
        y += 24

    # divider
    draw.line((b[0] + 520, b[1] + 50, b[0] + 520, b[3] - 30), fill=RULE, width=1)

    rx = b[0] + 560
    draw.text((rx, top_y), "INVERSE ELECTRON TRANSFER", fill=TEAL, font=_font("sans_b", 15))
    draw.text((rx, top_y + 40), "ET_SIGNATURE", fill=MUTED, font=_font("sans", 13))
    draw.text((rx, top_y + 62), sig, fill=INK, font=_font("serif_b", 26))
    draw.text((rx, top_y + 120), "ET_DEMAND", fill=MUTED, font=_font("sans", 13))
    draw.text((rx, top_y + 142), demand, fill=INK, font=_font("serif_b", 26))

    draw.rounded_rectangle((rx, top_y + 210, b[2] - 28, top_y + 290), radius=4, outline=RULE, fill=PANEL)
    note = (
        "Signature names the forward class;\n"
        "demand constrains feasible reverse electron flow."
    )
    draw.multiline_text((rx + 16, top_y + 228), note, fill=MUTED, font=_font("serif", 14), spacing=4)

    # -------- Panel c: reverse edge --------
    cbox = (48, 490, 1632, 860)
    _panel(draw, cbox, "c", f"Reverse mechanism graph — RETRO_EDGE  {src} → {dst}")

    # s0
    sx0 = cbox[0] + 40
    sy0 = cbox[1] + 55
    draw.text((sx0, sy0), f"{src}  ·  TARGET_STATE", fill=MUTED, font=_font("sans_b", 12))
    canvas.paste(img_s0, (sx0, sy0 + 22))

    # arrow + BE table in center
    table_x = sx0 + 390
    table_y = sy0 + 30
    draw.text((table_x + 40, table_y), "BE_DELTA", fill=ACCENT_RGB, font=_font("sans_b", 14))
    draw.text((table_x + 130, table_y), "(FlowER bond–electron units)", fill=MUTED, font=_font("sans", 12))
    _arrow(draw, table_x, table_y + 40, table_x + 280, color=ACCENT_RGB, width=2)

    # table header
    th = table_y + 58
    draw.rectangle((table_x, th, table_x + 300, th + 26), fill=PANEL, outline=RULE)
    for col, label in ((8, "type"), (90, "maps"), (210, "Δ")):
        draw.text((table_x + col, th + 5), label, fill=MUTED, font=_font("sans_b", 12))
    rows = _fmt_delta(delta)
    for i, (typ, maps, val) in enumerate(rows):
        yy = th + 28 + i * 24
        if i % 2 == 0:
            draw.rectangle((table_x, yy, table_x + 300, yy + 24), fill=(252, 252, 252))
        draw.line((table_x, yy + 24, table_x + 300, yy + 24), fill=RULE, width=1)
        draw.text((table_x + 8, yy + 4), typ, fill=INK, font=_font("mono", 13))
        draw.text((table_x + 90, yy + 4), maps, fill=INK, font=_font("mono", 13))
        draw.text((table_x + 210, yy + 4), val, fill=ACCENT_RGB if val.startswith("-") or "→" in val else TEAL, font=_font("mono", 13))
    draw.rectangle((table_x, th, table_x + 300, th + 28 + 24 * max(len(rows), 1)), outline=RULE, width=1)

    # s1 side
    sx1 = table_x + 340
    draw.text((sx1, sy0), f"{dst}  ·  PRECURSOR_STATE", fill=MUTED, font=_font("sans_b", 12))
    canvas.paste(img_s1a, (sx1, sy0 + 22))
    if img_s1b is not None:
        canvas.paste(img_s1b, (sx1 + 300, sy0 + 22))
        draw.text((sx1 + 300, sy0), "+ partner", fill=MUTED, font=_font("sans", 11))

    foot = "Highlighted atoms / bonds participate in the annotated BE_DELTA; Δ is locally verifiable from the two STATE SMILES."
    draw.text((cbox[0] + 16, cbox[3] - 28), foot, fill=MUTED, font=_font("sans", 12))

    # -------- Panel d: answer --------
    dbox = (48, 880, 1632, 1125)
    _panel(draw, dbox, "d", "Emit initial reactants  ⟨answer⟩")
    ax = dbox[0] + 30
    ay = dbox[1] + 48
    for i, im in enumerate(img_ans):
        canvas.paste(im, (ax + i * 320, ay))
    if shared:
        draw.text((ax + len(img_ans) * 320 + 10, ay + 20), "SHARED (spectators)", fill=MUTED, font=_font("sans_b", 12))
        # compact shared frags as text, not tiny ions drawings
        shared_txt = shared if len(shared) < 70 else shared[:67] + "…"
        draw.text((ax + len(img_ans) * 320 + 10, ay + 48), shared_txt, fill=INK, font=_font("mono", 11))
        draw.text(
            (ax + len(img_ans) * 320 + 10, ay + 78),
            "Carried as SHARED; appended in final answer.",
            fill=MUTED,
            font=_font("sans", 12),
        )

    draw.text(
        (dbox[0] + 16, dbox[3] - 28),
        "Final answer = PRECURSOR_STATE molecules ∪ SHARED.  Process is checkable without a neural teacher (Self-MechVR).",
        fill=MUTED,
        font=_font("sans", 12),
    )

    # footer
    if dpi_note:
        draw.text(
            (48, H - 36),
            "Generated with RDKit · MechET MECH_ET v3  ·  maps retained for process verification",
            fill=(150, 150, 150),
            font=_font("sans", 11),
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(300, 300))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/aaa/fionafyang/buddy1/whaleywang/reflow/data/orbit_mech_et_sft/valid.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=REPO / "docs/mechet_cot_example.png")
    parser.add_argument("--id", type=str, default="flower_mech_et_val_496")
    args = parser.parse_args()
    if not args.data.exists():
        args.data = REPO / "data/samples/valid_mini.jsonl"
        args.id = ""
    row = _pick_row(args.data, args.id)
    path = render(row, args.out)
    print(f"wrote {path}")
    print(f"id    {row.get('id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
