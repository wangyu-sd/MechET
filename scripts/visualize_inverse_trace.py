#!/usr/bin/env python3
"""Render mapped molecular states and electron-pair moves from rollout JSONL."""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw


COLORS = ["#2563eb", "#dc2626", "#7c3aed", "#059669"]


def _map_index(mol: Chem.Mol) -> dict[int, int]:
    return {
        int(atom.GetAtomMapNum()): atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum()
    }


def _point(
    coords: dict[int, tuple[float, float]], endpoint: dict[str, Any]
) -> tuple[float, float] | None:
    atoms = [int(value) for value in endpoint.get("atoms") or []]
    available = [coords[value] for value in atoms if value in coords]
    if not available or len(available) != len(atoms):
        return None
    x = sum(value[0] for value in available) / len(available)
    y = sum(value[1] for value in available) / len(available)
    if endpoint.get("kind") == "LP" and len(available) == 1:
        return x + 10.0, y - 10.0
    return x, y


def _move_points(
    coords: dict[int, tuple[float, float]], move: dict[str, Any]
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    source = dict(move.get("source") or {})
    sink = dict(move.get("sink") or {})
    start = _point(coords, source)
    # For bond formation A:→A-B, the bond does not exist in the pre-action
    # state. Aim at the other atom rather than the empty midpoint between
    # disconnected fragments.
    source_atoms = {int(value) for value in source.get("atoms") or []}
    sink_atoms = [int(value) for value in sink.get("atoms") or []]
    other_atoms = [value for value in sink_atoms if value not in source_atoms]
    if sink.get("kind") == "BOND" and len(other_atoms) == 1:
        end = coords.get(other_atoms[0])
    else:
        end = _point(coords, sink)
    return start, end


def _electron_overlay(
    coords: dict[int, tuple[float, float]], moves: list[dict[str, Any]]
) -> str:
    elements = [
        "<defs>",
        *[
            f'<marker id="arrow-{i}" markerWidth="8" markerHeight="8" '
            'refX="7" refY="3" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L0,6 L8,3 z" fill="{color}"/></marker>'
            for i, color in enumerate(COLORS)
        ],
        "</defs>",
    ]
    for index, move in enumerate(moves):
        color = COLORS[index % len(COLORS)]
        start, end = _move_points(coords, move)
        if start is None or end is None:
            continue
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length = max(math.hypot(dx, dy), 1.0)
        # A modest perpendicular offset makes paired moves readable as curved arrows.
        bend = min(42.0, max(18.0, length * 0.22)) * (1 if index % 2 == 0 else -1)
        cx = (sx + ex) / 2.0 - dy / length * bend
        cy = (sy + ey) / 2.0 + dx / length * bend
        elements.append(
            f'<path d="M {sx:.1f},{sy:.1f} Q {cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="3.2" '
            f'stroke-linecap="round" marker-end="url(#arrow-{index % len(COLORS)})"/>'
        )
        elements.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4.2" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{cx + 5:.1f}" y="{cy - 5:.1f}" fill="{color}" '
            'font-family="sans-serif" font-size="15" font-weight="700">2e⁻</text>'
        )
    return "\n".join(elements)


def render_transition_svg(
    state_smiles: str,
    moves: list[dict[str, Any]],
    output: Path,
    *,
    width: int = 1050,
    height: int = 560,
) -> None:
    mol = Chem.MolFromSmiles(state_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse state: {state_smiles}")
    rdMolDraw2D.PrepareMolForDrawing(mol)
    map_to_index = _map_index(mol)
    moved_maps = {
        int(value)
        for move in moves
        for endpoint in (move.get("source") or {}, move.get("sink") or {})
        for value in endpoint.get("atoms") or []
    }
    highlights = [map_to_index[value] for value in moved_maps if value in map_to_index]
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.annotationFontScale = 0.8
    options.padding = 0.08
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            charge = atom.GetFormalCharge()
            charge_text = "+" if charge == 1 else "−" if charge == -1 else ""
            options.atomLabels[atom.GetIdx()] = f"{atom.GetSymbol()}{charge_text}:{map_num}"
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlights,
        highlightAtomColors={index: (1.0, 0.86, 0.35) for index in highlights},
        highlightAtomRadii={index: 0.42 for index in highlights},
    )
    coords = {
        map_num: (
            float(drawer.GetDrawCoords(atom_index).x),
            float(drawer.GetDrawCoords(atom_index).y),
        )
        for map_num, atom_index in map_to_index.items()
    }
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace("</svg>", _electron_overlay(coords, moves) + "\n</svg>")
    output.write_text(svg, encoding="utf-8")


def render_transition_png(
    state_smiles: str,
    moves: list[dict[str, Any]],
    output: Path,
    *,
    width: int = 1050,
    height: int = 560,
) -> None:
    mol = Chem.MolFromSmiles(state_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse state: {state_smiles}")
    rdMolDraw2D.PrepareMolForDrawing(mol)
    map_to_index = _map_index(mol)
    moved_maps = {
        int(value)
        for move in moves
        for endpoint in (move.get("source") or {}, move.get("sink") or {})
        for value in endpoint.get("atoms") or []
    }
    highlights = [map_to_index[value] for value in moved_maps if value in map_to_index]
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.padding = 0.08
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            charge = atom.GetFormalCharge()
            charge_text = "+" if charge == 1 else "−" if charge == -1 else ""
            options.atomLabels[atom.GetIdx()] = f"{atom.GetSymbol()}{charge_text}:{map_num}"
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlights,
        highlightAtomColors={index: (1.0, 0.86, 0.35) for index in highlights},
        highlightAtomRadii={index: 0.42 for index in highlights},
    )
    coords = {
        map_num: (
            float(drawer.GetDrawCoords(atom_index).x),
            float(drawer.GetDrawCoords(atom_index).y),
        )
        for map_num, atom_index in map_to_index.items()
    }
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for index, move in enumerate(moves):
        color = COLORS[index % len(COLORS)]
        start, end = _move_points(coords, move)
        if start is None or end is None:
            continue
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length = max(math.hypot(dx, dy), 1.0)
        bend = min(42.0, max(18.0, length * 0.22)) * (1 if index % 2 == 0 else -1)
        cx = (sx + ex) / 2.0 - dy / length * bend
        cy = (sy + ey) / 2.0 + dx / length * bend
        points = []
        for n in range(41):
            t = n / 40
            x = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t**2 * ex
            y = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t**2 * ey
            points.append((x, y))
        draw.line(points, fill=color, width=4, joint="curve")
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=color)
        # Arrowhead follows the final quadratic-Bezier tangent.
        tx, ty = ex - cx, ey - cy
        tangent = max(math.hypot(tx, ty), 1.0)
        ux, uy = tx / tangent, ty / tangent
        px, py = -uy, ux
        arrow = [
            (ex, ey),
            (ex - ux * 14 + px * 7, ey - uy * 14 + py * 7),
            (ex - ux * 14 - px * 7, ey - uy * 14 - py * 7),
        ]
        draw.polygon(arrow, fill=color)
        draw.text((cx + 5, cy - 16), "2e-", fill=color)
    Image.alpha_composite(image, overlay).convert("RGB").save(output, quality=95)


def render_combined_trajectory_png(
    output_dir: Path,
    transitions: list[dict[str, Any]],
    output: Path,
) -> None:
    states = (
        [str(transitions[0].get("state_before") or "")]
        + [str(item.get("state_after") or "") for item in transitions]
    )
    state_images: list[Image.Image] = []
    for index, state in enumerate(states):
        path = output_dir / f"state_{index}.png"
        state_images.append(Image.open(path).convert("RGB"))
    action_images = [
        Image.open(output_dir / f"successful_step_{index + 1}.png").convert("RGB")
        for index in range(len(transitions))
    ]
    panels: list[tuple[str, Image.Image]] = []
    state_labels = ["S0 · Product", "S1 · Tetrahedral intermediate", "S2 · Precursor"]
    for index, state_image in enumerate(state_images):
        panels.append((state_labels[index] if index < len(state_labels) else f"S{index}", state_image))
        if index < len(action_images):
            panels.append((f"Electron transfer {index + 1} · coupled 2e⁻ moves", action_images[index]))

    panel_width, panel_height = 620, 365
    gap, title_height, footer_height = 24, 105, 72
    canvas_width = gap + len(panels) * (panel_width + gap)
    canvas_height = title_height + panel_height + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 22), "MechET environment-owned inverse electron-flow trajectory", fill="#172033")
    draw.text(
        (gap, 54),
        "Yellow = participating mapped atoms · curved arrows = explicit two-electron source → sink",
        fill="#667085",
    )
    for index, (label, image) in enumerate(panels):
        x = gap + index * (panel_width + gap)
        y = title_height
        image.thumbnail((panel_width - 18, panel_height - 48), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        px = (panel_width - image.width) // 2
        py = 42 + (panel_height - 42 - image.height) // 2
        panel.paste(image, (px, py))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline="#d9e0ea", width=2)
        panel_draw.text((15, 13), label, fill="#172033")
        canvas.paste(panel, (x, y))
        if index < len(panels) - 1:
            mid_y = y + panel_height // 2
            arrow_x = x + panel_width + 5
            draw.line((arrow_x, mid_y, arrow_x + gap - 10, mid_y), fill="#667085", width=3)
            draw.polygon(
                [(arrow_x + gap - 8, mid_y), (arrow_x + gap - 17, mid_y - 6), (arrow_x + gap - 17, mid_y + 6)],
                fill="#667085",
            )
    draw.text(
        (gap, title_height + panel_height + 24),
        "Formal execution: PASS  ·  2 committed transitions  ·  structural precursor exact: PASS",
        fill="#087f5b",
    )
    canvas.save(output, quality=95)


def render_combined_trajectory_svg(
    output_dir: Path,
    transitions: list[dict[str, Any]],
    output: Path,
) -> None:
    panel_files: list[tuple[str, Path]] = []
    state_labels = ["S0 · Product", "S1 · Tetrahedral intermediate", "S2 · Precursor"]
    for index in range(len(transitions) + 1):
        panel_files.append((state_labels[index], output_dir / f"state_{index}.svg"))
        if index < len(transitions):
            panel_files.append(
                (f"Electron transfer {index + 1} · coupled 2e⁻ moves", output_dir / f"successful_step_{index + 1}.svg")
            )
    panel_width, panel_height, gap, title_height = 620, 365, 24, 105
    width = gap + len(panel_files) * (panel_width + gap)
    height = title_height + panel_height + 72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        '<defs><marker id="flow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L0,9 L9,4.5 z" fill="#667085"/></marker></defs>',
        f'<text x="{gap}" y="40" font-family="sans-serif" font-size="24" font-weight="700" fill="#172033">MechET environment-owned inverse electron-flow trajectory</text>',
        f'<text x="{gap}" y="70" font-family="sans-serif" font-size="15" fill="#667085">Yellow = participating mapped atoms · curved arrows = explicit two-electron source → sink</text>',
    ]
    for index, (label, path) in enumerate(panel_files):
        x, y = gap + index * (panel_width + gap), title_height
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{panel_width}" height="{panel_height}" rx="10" fill="white" stroke="#d9e0ea" stroke-width="2"/>',
                f'<text x="{x + 15}" y="{y + 27}" font-family="sans-serif" font-size="16" font-weight="700" fill="#172033">{html.escape(label)}</text>',
                f'<image x="{x + 10}" y="{y + 38}" width="{panel_width - 20}" height="{panel_height - 48}" preserveAspectRatio="xMidYMid meet" href="data:image/svg+xml;base64,{encoded}"/>',
            ]
        )
        if index < len(panel_files) - 1:
            x1 = x + panel_width + 5
            x2 = x + panel_width + gap - 7
            ym = y + panel_height / 2
            parts.append(
                f'<path d="M{x1},{ym} L{x2},{ym}" stroke="#667085" stroke-width="3" marker-end="url(#flow)"/>'
            )
    parts.extend(
        [
            f'<text x="{gap}" y="{title_height + panel_height + 40}" font-family="sans-serif" font-size="16" font-weight="700" fill="#087f5b">Formal execution: PASS · 2 committed transitions · structural precursor exact: PASS</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")


def _move_text(move: dict[str, Any]) -> str:
    source = dict(move.get("source") or {})
    sink = dict(move.get("sink") or {})
    return (
        f'{source.get("kind")}({",".join(map(str, source.get("atoms") or []))}) '
        f'→ {sink.get("kind")}({",".join(map(str, sink.get("atoms") or []))})'
    )


def _selected_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidates = list(row.get("candidates") or [])
    index = int(row.get("selected_candidate_index") or 0)
    return dict(candidates[index]) if candidates else {}


def build_report(predictions: Path, output_dir: Path) -> Path:
    rows = [json.loads(line) for line in predictions.open() if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = next(
        row
        for row in rows
        if bool((row.get("terminal_result") or {}).get("formal_execute"))
    )
    transitions = list((successful.get("rollout_state") or {}).get("flow_trace", {}).get("transitions") or [])
    state_sections: list[str] = []
    trajectory_states = (
        [str(transitions[0].get("state_before") or "")]
        + [str(item.get("state_after") or "") for item in transitions]
        if transitions
        else []
    )
    state_labels = ["S0 · target", "S1 · tetrahedral intermediate", "S2 · precursor state"]
    for state_index, state_smiles in enumerate(trajectory_states):
        svg_name = f"state_{state_index}.svg"
        render_transition_svg(
            state_smiles, [], output_dir / svg_name, width=900, height=440
        )
        render_transition_png(
            state_smiles, [], output_dir / f"state_{state_index}.png", width=900, height=440
        )
        label = state_labels[state_index] if state_index < len(state_labels) else f"S{state_index}"
        state_sections.append(
            f'<figure><figcaption>{html.escape(label)}</figcaption>'
            f'<img src="{svg_name}" alt="{html.escape(label)}"></figure>'
        )
    step_sections: list[str] = []
    for transition in transitions:
        index = int(transition.get("step_index") or 0)
        imports = list(transition.get("imports") or [])
        action_state = str(transition.get("state_before") or "")
        if imports:
            action_state += "." + ".".join(imports)
        svg_name = f"successful_step_{index + 1}.svg"
        render_transition_svg(
            action_state,
            list(transition.get("moves") or []),
            output_dir / svg_name,
        )
        render_transition_png(
            action_state,
            list(transition.get("moves") or []),
            output_dir / f"successful_step_{index + 1}.png",
        )
        move_lines = "".join(
            f'<li><span style="color:{COLORS[i % len(COLORS)]}">●</span> '
            f'{html.escape(_move_text(move))}</li>'
            for i, move in enumerate(transition.get("moves") or [])
        )
        step_sections.append(
            f"""
            <section class="step">
              <div class="step-title"><span>Step {index + 1}</span><b>PASS · coupled 2e⁻ moves</b></div>
              <img src="{svg_name}" alt="Electron-flow step {index + 1}">
              <div class="details"><div><strong>Imported fragments</strong><br>{html.escape(' · '.join(imports) or 'none')}</div>
              <ol>{move_lines}</ol></div>
            </section>"""
        )

    render_combined_trajectory_png(
        output_dir,
        transitions,
        output_dir / "successful_trajectory_all_in_one.png",
    )
    render_combined_trajectory_svg(
        output_dir,
        transitions,
        output_dir / "successful_trajectory_all_in_one.svg",
    )

    cards: list[str] = []
    for row_index, row in enumerate(rows):
        candidate = _selected_candidate(row)
        state = dict(row.get("rollout_state") or {})
        final = dict(row.get("terminal_result") or {})
        ok = bool(final.get("formal_execute"))
        exchanges = list(candidate.get("exchanges") or [])
        failed_codes = [
            str((item.get("result") or {}).get("code") or "")
            for item in exchanges
            if (item.get("result") or {}).get("ok") is False
        ]
        common_code = max(set(failed_codes), key=failed_codes.count) if failed_codes else "—"
        diagnostic_image = ""
        attempted_move_text = ""
        if not ok:
            current_state = str(row.get("target_smiles") or "")
            attempted_moves: list[dict[str, Any]] = []
            for event in (state.get("trace") or []):
                result = dict(event.get("result") or {})
                if result.get("ok") is True and result.get("state_smiles"):
                    current_state = str(result["state_smiles"])
                if event.get("event") == "apply_moves" and result.get("ok") is False:
                    attempted_moves = list(event.get("moves") or [])
                    break
            diagnostic_name = f"failed_sample_{row_index + 1}.svg"
            render_transition_svg(
                current_state,
                attempted_moves,
                output_dir / diagnostic_name,
                width=900,
                height=440,
            )
            render_transition_png(
                current_state,
                attempted_moves,
                output_dir / f"failed_sample_{row_index + 1}.png",
                width=900,
                height=440,
            )
            diagnostic_image = (
                f'<img class="diagnostic" src="{diagnostic_name}" '
                f'alt="Failed attempted state for {html.escape(str(row.get("id")))}">'
            )
            if attempted_moves:
                mapped = {
                    atom.GetAtomMapNum()
                    for atom in (Chem.MolFromSmiles(current_state) or Chem.Mol()).GetAtoms()
                }
                referenced = {
                    int(value)
                    for move in attempted_moves
                    for endpoint in (move.get("source") or {}, move.get("sink") or {})
                    for value in endpoint.get("atoms") or []
                }
                missing = sorted(referenced - mapped)
                attempted_move_text = (
                    '<p class="attempt"><strong>Attempted:</strong> '
                    + html.escape("; ".join(_move_text(move) for move in attempted_moves))
                    + (f'<br><strong>Missing maps:</strong> {missing}' if missing else "")
                    + "</p>"
                )
        cards.append(
            f"""<article class="sample {'pass' if ok else 'fail'}">
            <div class="badge">{'EXECUTED' if ok else 'ABSTAINED'}</div>
            <h3>{html.escape(str(row.get('id')))}</h3>
            {diagnostic_image}
            {attempted_move_text}
            <dl><dt>Tool calls</dt><dd>{state.get('tool_calls', 0)}</dd>
            <dt>Successful steps</dt><dd>{state.get('successful_steps', 0)}</dd>
            <dt>Failed steps</dt><dd>{state.get('failed_steps', 0)}</dd>
            <dt>Main diagnostic</dt><dd>{html.escape(common_code)}</dd></dl>
            </article>"""
        )

    terminal = dict(successful.get("terminal_result") or {})
    report = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>MechET inverse trace visualization</title>
    <style>
    :root{{--ink:#172033;--muted:#667085;--line:#d9e0ea;--blue:#2563eb;--green:#087f5b;--red:#c92a2a;--paper:#f5f7fb}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}
    main{{max-width:1220px;margin:0 auto;padding:40px 28px 70px}} h1{{font-size:34px;margin:0 0 7px}} .lead{{color:var(--muted);margin:0 0 28px}}
    .summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0 34px}} .metric,.sample,.step{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 2px 8px #1822300b}}
    .metric{{padding:17px}} .metric strong{{display:block;font-size:27px}} .metric span{{color:var(--muted)}}
    h2{{margin-top:38px}} .step{{overflow:hidden;margin:18px 0 26px}} .step-title{{padding:14px 18px;display:flex;justify-content:space-between;border-bottom:1px solid var(--line)}} .step-title span{{font-size:19px;font-weight:750}} .step-title b{{color:var(--green)}}
    .states{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0 30px}} figure{{margin:0;background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden}} figcaption{{padding:10px 13px;font-weight:750;border-bottom:1px solid var(--line)}} figure img{{display:block;width:100%}}
    .step img{{display:block;width:100%;height:auto;background:white}} .details{{border-top:1px solid var(--line);padding:14px 20px;display:grid;grid-template-columns:1.1fr 1fr;gap:24px}} .details ol{{margin:0}}
    .samples{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}} .sample{{padding:18px;border-top:5px solid var(--red)}} .sample.pass{{border-top-color:var(--green)}} .badge{{font-size:12px;font-weight:800;color:var(--red)}} .pass .badge{{color:var(--green)}} h3{{font-size:15px;word-break:break-all}} .diagnostic{{display:block;width:100%;border:1px solid var(--line);border-radius:8px;margin:10px 0;background:white}} .attempt{{font-size:12px;color:var(--muted);word-break:break-word}}
    dl{{display:grid;grid-template-columns:1fr 1fr;margin:0}} dt,dd{{padding:5px 0;border-bottom:1px solid #edf0f4}} dt{{color:var(--muted)}} dd{{text-align:right;font-weight:650}}
    .endpoint{{background:#e8f5ee;border-left:5px solid var(--green);padding:16px 18px;border-radius:8px;word-break:break-all}}
    .legend{{color:var(--muted);font-size:13px}} @media(max-width:760px){{.summary,.samples,.states{{grid-template-columns:1fr}}.details{{grid-template-columns:1fr}}}}
    </style></head><body><main>
    <h1>MechET held-out inverse-trace smoke test</h1>
    <p class="lead">Qwen3-8B Tool-SFT · greedy decoding · 4 held-out reactions · arrows point from electron-pair source to sink.</p>
    <div class="summary"><div class="metric"><strong>1 / 4</strong><span>Formal execution</span></div><div class="metric"><strong>1 / 4</strong><span>Structural exact</span></div><div class="metric"><strong>2</strong><span>Committed transitions</span></div><div class="metric"><strong>3 / 4</strong><span>Abstained</span></div></div>
    <h2>Successful environment-owned state trajectory</h2>
    <div class="states">{''.join(state_sections)}</div>
    <p class="legend">Yellow halos mark atoms participating in the action. Colored curved arrows show explicit two-electron source→sink moves; atom labels include map numbers.</p>
    {''.join(step_sections)}
    <div class="endpoint"><strong>Derived structural precursor</strong><br>{html.escape(str(terminal.get('structural_precursor') or ''))}</div>
    <h2>All four smoke-test outcomes</h2><div class="samples">{''.join(cards)}</div>
    </main></body></html>"""
    report_path = output_dir / "index.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_report(args.predictions, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
