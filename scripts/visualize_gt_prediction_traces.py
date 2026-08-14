#!/usr/bin/env python3
"""Render publication-ready ground-truth versus predicted inverse traces.

The figure deliberately separates atom-contributing states from auxiliary
imports.  Electron-flow arrows are drawn on the pre-action state and all
reported predictions are rechecked with the strict trace-owned evaluator.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import cairosvg
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from mechet.endpoints import structural_exact
from mechet.strict_prediction_evaluation import endpoint_evaluation


INK = "#171717"
MUTED = "#62676F"
RULE = "#D7D9DD"
PAPER = "#FFFFFF"
GT = "#245C8A"
PRED = "#237A57"
ARROW = "#A83232"
HIGHLIGHT = (1.0, 0.88, 0.46)

DEFAULT_CASES = (
    ("mech-uspto31k-inverse:6820", 0, "SN2-type C–N disconnection"),
    ("mech-uspto31k-inverse:18225", 3, "Sulfonamide formation"),
    ("mech-uspto31k-inverse:22494", 1, "Amide formation"),
)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _load_rows(paths: Iterable[Path], wanted: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A remote shard may be observed while its last line is being written.
                    continue
                identifier = str(row.get("id") or "")
                if identifier in wanted:
                    rows[identifier] = row
                    if len(rows) == len(wanted):
                        return rows
    return rows


def _maps(smiles: str) -> set[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}


def _structural_view(smiles: str, target_maps: set[int]) -> str:
    """Drop non-atom-contributing spectators from a state drawing."""
    kept: list[str] = []
    for fragment in str(smiles).split("."):
        if not fragment:
            continue
        if _maps(fragment) & target_maps:
            kept.append(fragment)
    return ".".join(kept) or smiles


def _active_maps(moves: Iterable[Mapping[str, Any]]) -> set[int]:
    return {
        int(atom)
        for move in moves
        for endpoint in (move.get("source") or {}, move.get("sink") or {})
        for atom in endpoint.get("atoms") or []
    }


def _endpoint_point(
    coords: Mapping[int, tuple[float, float]], endpoint: Mapping[str, Any]
) -> tuple[float, float] | None:
    atoms = [int(value) for value in endpoint.get("atoms") or []]
    points = [coords[value] for value in atoms if value in coords]
    if len(points) != len(atoms) or not points:
        return None
    x = sum(point[0] for point in points) / len(points)
    y = sum(point[1] for point in points) / len(points)
    if endpoint.get("kind") == "LP":
        return x + 9.0, y - 11.0
    return x, y


def _move_points(
    mol: Chem.Mol,
    map_to_idx: Mapping[int, int],
    coords: Mapping[int, tuple[float, float]],
    move: Mapping[str, Any],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    source = dict(move.get("source") or {})
    sink = dict(move.get("sink") or {})
    start = _endpoint_point(coords, source)
    sink_atoms = [int(value) for value in sink.get("atoms") or []]
    if sink.get("kind") == "BOND" and len(sink_atoms) == 2:
        left, right = sink_atoms
        bond_exists = (
            left in map_to_idx
            and right in map_to_idx
            and mol.GetBondBetweenAtoms(map_to_idx[left], map_to_idx[right]) is not None
        )
        source_atoms = {int(value) for value in source.get("atoms") or []}
        novel = [value for value in sink_atoms if value not in source_atoms]
        if not bond_exists and len(novel) == 1:
            end = coords.get(novel[0])
        else:
            end = _endpoint_point(coords, sink)
    else:
        end = _endpoint_point(coords, sink)
    return start, end


def _mol_panel_svg(
    smiles: str,
    moves: list[dict[str, Any]],
    *,
    uid: str,
    width: int = 470,
    height: int = 250,
) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles}")
    rdDepictor.Compute2DCoords(mol)
    map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum()
    }
    active = _active_maps(moves)
    draw_mol = Chem.Mol(mol)
    # Reaction-centre maps remain visible; the rest would overwhelm a paper panel.
    for atom in draw_mol.GetAtoms():
        if atom.GetAtomMapNum() not in active:
            atom.SetAtomMapNum(0)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.padding = 0.08
    opts.bondLineWidth = 2.0
    opts.fixedBondLength = 24
    opts.minFontSize = 12
    opts.maxFontSize = 20
    opts.annotationFontScale = 0.72
    active_indices = [map_to_idx[value] for value in active if value in map_to_idx]
    drawer.DrawMolecule(
        draw_mol,
        highlightAtoms=active_indices,
        highlightAtomColors={index: HIGHLIGHT for index in active_indices},
        highlightAtomRadii={index: 0.36 for index in active_indices},
    )
    coords = {
        atom_map: (
            float(drawer.GetDrawCoords(atom_idx).x),
            float(drawer.GetDrawCoords(atom_idx).y),
        )
        for atom_map, atom_idx in map_to_idx.items()
    }
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg_start = svg.find("<svg")
    inner = svg[svg.find(">", svg_start) + 1 : svg.rfind("</svg>")]
    overlays = [
        f'<defs><marker id="ah-{uid}" markerWidth="7" markerHeight="7" '
        'refX="6.2" refY="3.5" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L0,7 L7,3.5 z" fill="{ARROW}"/></marker></defs>'
    ]
    for index, move in enumerate(moves):
        start, end = _move_points(mol, map_to_idx, coords, move)
        if start is None or end is None:
            continue
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1.0)
        sign = 1 if index % 2 == 0 else -1
        bend = sign * min(34.0, max(16.0, distance * 0.24))
        cx = (sx + ex) / 2 - dy / distance * bend
        cy = (sy + ey) / 2 + dx / distance * bend
        overlays.append(
            f'<path d="M {sx:.1f},{sy:.1f} Q {cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" '
            f'fill="none" stroke="{ARROW}" stroke-width="2.5" stroke-linecap="round" '
            f'marker-end="url(#ah-{uid})"/>'
        )
        overlays.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.8" fill="{ARROW}"/>')
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        + inner
        + "".join(overlays)
        + "</svg>"
    )


def _embedded_svg(value: str, x: float, y: float, width: float, height: float) -> str:
    encoded = base64.b64encode(value.encode()).decode()
    return (
        f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
        f'href="data:image/svg+xml;base64,{encoded}"/>'
    )


def _atom_symbols(smiles: str) -> dict[int, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        atom.GetAtomMapNum(): atom.GetSymbol()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum()
    }


def _container_text(endpoint: Mapping[str, Any], symbols: Mapping[int, str]) -> str:
    kind = str(endpoint.get("kind") or "")
    atoms = [int(value) for value in endpoint.get("atoms") or []]
    labels = [f"{symbols.get(value, '?')}:{value}" for value in atoms]
    if kind == "LP":
        return f"lp({labels[0]})"
    if kind == "ATOM":
        return f"{labels[0]}"
    return "–".join(labels)


def _move_text(state: str, moves: Iterable[Mapping[str, Any]]) -> str:
    symbols = _atom_symbols(state)
    return ";  ".join(
        f"{_container_text(move.get('source') or {}, symbols)} → "
        f"{_container_text(move.get('sink') or {}, symbols)}"
        for move in moves
    )


def _fragment_label(fragment: str) -> str:
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return fragment
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        formula = CalcMolFormula(mol)
    except Exception:
        formula = Chem.MolToSmiles(mol)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if charge and not formula.endswith(("+", "-")):
        formula += f" ({charge:+d})"
    return formula


def _imports_text(transitions: Iterable[Mapping[str, Any]], initial: Iterable[str] = ()) -> str:
    fragments = list(initial)
    for transition in transitions:
        fragments.extend(str(value) for value in transition.get("imports") or [])
    if not fragments:
        return "Auxiliary imports: none"
    return "Auxiliary imports: " + ", ".join(_fragment_label(value) for value in fragments)


def _gt_transitions(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = dict((reference.get("metadata") or {}).get("trace_plan") or {})
    return [dict(value) for value in plan.get("steps") or []]


def _predicted_row(
    reference: Mapping[str, Any], prediction: Mapping[str, Any], candidate_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = list(prediction.get("candidates") or [])
    candidate = next(
        (value for value in candidates if int(value.get("sample_index", -1)) == candidate_index),
        None,
    )
    if candidate is None:
        raise ValueError(f"candidate {candidate_index} missing for {reference['id']}")
    merged = deepcopy(dict(reference))
    merged.update(
        {
            key: value
            for key, value in prediction.items()
            if key not in {"messages", "rollout_state", "terminal_result", "candidates"}
        }
    )
    merged["messages"] = candidate.get("messages") or []
    merged["rollout_state"] = candidate.get("rollout_state") or {}
    merged["terminal_result"] = merged["rollout_state"].get("final_result") or {}
    evaluation = endpoint_evaluation(merged)
    if not evaluation.get("formal_execute") or not evaluation.get("structural_exact"):
        raise ValueError(
            f"selected example is not strict structural-exact: {reference['id']} "
            f"candidate={candidate_index} evaluation={evaluation}"
        )
    return dict(candidate), evaluation


def _lane(
    *,
    y: int,
    label: str,
    color: str,
    transitions: list[dict[str, Any]],
    target_maps: set[int],
    uid: str,
    initial_imports: Iterable[str] = (),
    exact: bool = False,
) -> str:
    states = [str(transitions[0]["state_before"])] + [
        str(value["state_after"]) for value in transitions
    ]
    count = len(states)
    panel_w, panel_h = 470, 250
    available_x0, available_x1 = 245, 1745
    if count == 2:
        xs = [350, 1040]
    else:
        spacing = (available_x1 - available_x0 - panel_w) / max(count - 1, 1)
        xs = [available_x0 + index * spacing for index in range(count)]
    parts = [
        f'<rect x="36" y="{y}" width="1708" height="346" rx="8" fill="#FFFFFF" stroke="{RULE}"/>',
        f'<rect x="36" y="{y}" width="9" height="346" rx="4" fill="{color}"/>',
        f'<text x="62" y="{y + 37}" class="lane" fill="{color}">{_escape(label)}</text>',
        f'<text x="62" y="{y + 66}" class="small">{_escape(_imports_text(transitions, initial_imports))}</text>',
    ]
    if exact:
        parts.extend(
            [
                f'<rect x="1553" y="{y + 18}" width="164" height="34" rx="17" fill="#E8F4EE"/>',
                f'<text x="1635" y="{y + 41}" text-anchor="middle" class="badge" fill="{PRED}">STRUCTURAL EXACT</text>',
            ]
        )
    for index, state in enumerate(states):
        moves = list(transitions[index].get("moves") or []) if index < len(transitions) else []
        structural = _structural_view(state, target_maps)
        panel = _mol_panel_svg(structural, moves, uid=f"{uid}-{index}", width=panel_w, height=panel_h)
        parts.append(_embedded_svg(panel, xs[index], y + 58, panel_w, panel_h))
        state_name = "Product" if index == 0 else ("Precursor" if index == count - 1 else "Intermediate")
        parts.append(
            f'<text x="{xs[index] + panel_w / 2:.1f}" y="{y + 323}" text-anchor="middle" class="state">'
            f'S{index} · {state_name}</text>'
        )
        if index < len(transitions):
            move_text = _move_text(state, moves)
            parts.append(
                f'<text x="{xs[index] + panel_w / 2:.1f}" y="{y + 342}" text-anchor="middle" class="moves">'
                f'{_escape(move_text)}</text>'
            )
        if index < count - 1:
            arrow_x0 = xs[index] + panel_w + 8
            arrow_x1 = xs[index + 1] - 12
            arrow_y = y + 184
            parts.append(
                f'<path d="M {arrow_x0:.1f},{arrow_y} L {arrow_x1:.1f},{arrow_y}" '
                f'stroke="{INK}" stroke-width="1.8" marker-end="url(#reaction-arrow)"/>'
            )
            parts.append(
                f'<text x="{(arrow_x0 + arrow_x1) / 2:.1f}" y="{arrow_y - 12}" '
                f'text-anchor="middle" class="step">inverse step {index + 1}</text>'
            )
    return "".join(parts)


def build_figure(
    references: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    cases: Iterable[tuple[str, int, str]],
) -> tuple[str, list[dict[str, Any]]]:
    cases = list(cases)
    width = 1780
    header_h, case_h, footer_h = 154, 820, 100
    height = header_h + case_h * len(cases) + footer_h
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """<defs><marker id="reaction-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#171717"/></marker></defs>""",
        f'<rect width="100%" height="100%" fill="{PAPER}"/>',
        """<style>
        text{font-family:Arial,Helvetica,sans-serif;fill:#171717}
        .title{font-family:Georgia,'Times New Roman',serif;font-size:34px;font-weight:700}
        .subtitle{font-size:18px;fill:#62676F}.case{font-family:Georgia,'Times New Roman',serif;font-size:25px;font-weight:700}
        .lane{font-size:21px;font-weight:700}.small{font-size:14px;fill:#62676F}.badge{font-size:12px;font-weight:700;letter-spacing:.6px}
        .state{font-size:15px;font-weight:700}.moves{font-size:11.5px;fill:#62676F}.step{font-size:12px;font-style:italic;fill:#62676F}
        .foot{font-size:14px;fill:#62676F}.paneltag{font-size:22px;font-weight:700}
        </style>""",
        '<text x="36" y="52" class="title">Executable inverse electron-flow reasoning: ground truth vs MechET</text>',
        '<text x="36" y="85" class="subtitle">Held-out mech-USPTO-31k examples · Qwen3-8B Tool-SFT · full-headed red arrows denote two-electron flow</text>',
        f'<line x1="36" y1="112" x2="1744" y2="112" stroke="{INK}" stroke-width="1.4"/>',
    ]
    manifest_cases: list[dict[str, Any]] = []
    for case_index, (identifier, candidate_index, title) in enumerate(cases):
        reference = references[identifier]
        prediction = predictions[identifier]
        candidate, evaluation = _predicted_row(reference, prediction, candidate_index)
        gt_transitions = _gt_transitions(reference)
        predicted_transitions = list(
            ((candidate.get("rollout_state") or {}).get("flow_trace") or {}).get("transitions") or []
        )
        if not predicted_transitions:
            raise ValueError(f"predicted transitions missing for {identifier}")
        base_y = header_h + case_index * case_h
        tag = chr(ord("A") + case_index)
        content.extend(
            [
                f'<text x="36" y="{base_y + 31}" class="paneltag">({tag})</text>',
                f'<text x="86" y="{base_y + 31}" class="case">{_escape(title)}</text>',
                f'<text x="1718" y="{base_y + 30}" text-anchor="end" class="small">{_escape(identifier)} · sample {candidate_index}</text>',
                _lane(
                    y=base_y + 50,
                    label="Ground-truth inverse replay",
                    color=GT,
                    transitions=gt_transitions,
                    target_maps=_maps(str(reference["target_smiles"])),
                    uid=f"gt-{case_index}",
                    initial_imports=((reference.get("metadata") or {}).get("trace_plan") or {}).get("initial_imports") or [],
                ),
                _lane(
                    y=base_y + 414,
                    label="MechET prediction",
                    color=PRED,
                    transitions=[dict(value) for value in predicted_transitions],
                    target_maps=_maps(str(reference["target_smiles"])),
                    uid=f"pred-{case_index}",
                    exact=True,
                ),
            ]
        )
        gt_moves = [move for transition in gt_transitions for move in transition.get("moves") or []]
        pred_moves = [move for transition in predicted_transitions for move in transition.get("moves") or []]
        terminal = dict((candidate.get("rollout_state") or {}).get("final_result") or {})
        manifest_cases.append(
            {
                "id": identifier,
                "candidate_index": candidate_index,
                "title": title,
                "strict_evaluation": evaluation,
                "n_gt_steps": len(gt_transitions),
                "n_predicted_steps": len(predicted_transitions),
                "gt_moves": gt_moves,
                "predicted_moves": pred_moves,
                "move_sequence_identical": gt_moves == pred_moves,
                "gt_imports": ((reference.get("metadata") or {}).get("trace_plan") or {}).get("initial_imports") or [],
                "predicted_imports": [
                    fragment
                    for transition in predicted_transitions
                    for fragment in transition.get("imports") or []
                ],
                "predicted_structural_precursor": terminal.get("structural_precursor"),
                "reference_structural_precursor": reference.get("structural_precursor"),
                "structural_exact_direct_check": structural_exact(
                    str(terminal.get("structural_precursor") or ""),
                    str(reference.get("structural_precursor") or ""),
                ),
            }
        )
    footer_y = header_h + case_h * len(cases) + 30
    content.extend(
        [
            f'<line x1="36" y1="{footer_y - 15}" x2="1744" y2="{footer_y - 15}" stroke="{RULE}"/>',
            f'<circle cx="49" cy="{footer_y + 8}" r="8" fill="#F9D878"/><text x="66" y="{footer_y + 13}" class="foot">highlighted reaction-centre atom</text>',
            f'<path d="M 330,{footer_y + 10} Q 360,{footer_y - 13} 393,{footer_y + 8}" fill="none" stroke="{ARROW}" stroke-width="2.5" marker-end="url(#reaction-arrow)"/><text x="410" y="{footer_y + 13}" class="foot">electron-pair source → sink</text>',
            f'<text x="1744" y="{footer_y + 13}" text-anchor="end" class="foot">Only reaction-centre atom maps are shown; spectator imports are listed but omitted from the main structures.</text>',
            "</svg>",
        ]
    )
    return "".join(content), manifest_cases


def _parse_case(value: str) -> tuple[str, int, str]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case must be ID|CANDIDATE_INDEX|TITLE")
    return parts[0], int(parts[1]), parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=sorted(Path("outputs/eval/mech_uspto31k_qwen3_8b_k10").glob("predictions.shard-*.jsonl")),
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/mech_uspto_31k_inverse_tool_sft/test.jsonl"),
    )
    parser.add_argument("--case", action="append", type=_parse_case)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations/paper_gt_vs_prediction"),
    )
    args = parser.parse_args()
    cases = tuple(args.case or DEFAULT_CASES)
    wanted = {value[0] for value in cases}
    references = _load_rows([args.references], wanted)
    predictions = _load_rows(args.predictions, wanted)
    missing = sorted(wanted - references.keys())
    if missing:
        raise SystemExit(f"references missing: {missing}")
    missing = sorted(wanted - predictions.keys())
    if missing:
        raise SystemExit(f"predictions missing: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg, manifest_cases = build_figure(references, predictions, cases)
    svg_path = args.output_dir / "mechet_gt_vs_prediction.svg"
    pdf_path = args.output_dir / "mechet_gt_vs_prediction.pdf"
    png_path = args.output_dir / "mechet_gt_vs_prediction.png"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), output_width=2670)
    individual: list[dict[str, str]] = []
    for case_number, case in enumerate(cases, start=1):
        case_svg, _ = build_figure(references, predictions, [case])
        stem = f"example_{case_number}_{case[0].rsplit(':', 1)[-1]}"
        case_svg_path = args.output_dir / f"{stem}.svg"
        case_pdf_path = args.output_dir / f"{stem}.pdf"
        case_png_path = args.output_dir / f"{stem}.png"
        case_svg_path.write_text(case_svg, encoding="utf-8")
        cairosvg.svg2pdf(bytestring=case_svg.encode(), write_to=str(case_pdf_path))
        cairosvg.svg2png(
            bytestring=case_svg.encode(), write_to=str(case_png_path), output_width=2670
        )
        individual.append(
            {
                "svg": str(case_svg_path.resolve()),
                "pdf": str(case_pdf_path.resolve()),
                "png": str(case_png_path.resolve()),
            }
        )
    manifest = {
        "artifact_type": "paper_gt_vs_prediction_figure",
        "references": str(args.references.resolve()),
        "prediction_files": [str(path.resolve()) for path in args.predictions],
        "svg": str(svg_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "png": str(png_path.resolve()),
        "individual_figures": individual,
        "cases": manifest_cases,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    caption = """# Suggested figure caption

**Executable inverse electron-flow trajectories predicted by MechET.** Ground-truth
inverse replays (blue) and independently sampled Qwen3-8B Tool-SFT predictions
(green) are shown for three held-out mech-USPTO-31k reactions: (A) an SN2-type
C–N disconnection, (B) sulfonamide formation through a tetrahedral sulfur
intermediate, and (C) amide formation through a tetrahedral carbonyl intermediate.
Full-headed red curved arrows denote explicit two-electron source-to-sink moves;
yellow halos and atom-map labels identify the reacting atoms. In all examples the
predicted environment-owned trace formally executes and gives the exact structural
precursor. Auxiliary fragments generated by the model are listed separately and
are excluded from structural-precursor accuracy because they contribute no atoms
to the mapped target.

The displayed candidates were selected only after generation for qualitative
visualization. They must not be described as the online top-1 selector output.
Case IDs and sample indices are printed in the figure and frozen in `manifest.json`.
"""
    (args.output_dir / "figure_caption.md").write_text(caption, encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("svg", "pdf", "png")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
