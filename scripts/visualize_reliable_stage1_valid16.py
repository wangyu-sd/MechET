#!/usr/bin/env python3
"""Render the frozen 16-case Stage-I smoke as GT/predicted RDKit trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import textwrap
from typing import Any

import cairosvg
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mechet.endpoints import reference_structural_precursor


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def unmapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles), params)
    if mol is None:
        raise ValueError(f"RDKit could not depict SMILES: {smiles}")
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    rdDepictor.Compute2DCoords(mol)
    return mol


def molecule_svg(smiles: str, *, width: int = 610, height: int = 275) -> str:
    mol = unmapped_mol(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.padding = 0.08
    options.bondLineWidth = 2.0
    options.minFontSize = 12
    options.maxFontSize = 21
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg") :]


def current_state(message: str) -> str:
    match = re.search(r"^CURRENT STATE SMILES: (.+)$", message, re.MULTILINE)
    if match is None:
        raise ValueError("frozen decision has no current-state SMILES")
    return match.group(1).strip()


def action_text(name: str, arguments: dict[str, Any]) -> str:
    if name == "import_fragments":
        parts = [
            f"{item.get('count', 1)} × {item.get('smiles', '')}"
            f" ({item.get('purpose', 'unspecified')})"
            for item in arguments.get("fragments") or []
        ]
        return "Import " + "; ".join(parts)
    if name == "apply_electron_flow":
        arrows = [
            f"{item.get('source', '?')} → {item.get('destination', '?')}"
            for item in arguments.get("electron_flow") or []
        ]
        return "Electron flow: " + "; ".join(arrows)
    if name == "finish_trace":
        return "Finish; derive precursor from executor state"
    return f"Unparsed proposal ({name or 'no tool'})"


def gt_steps(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(decisions, key=lambda row: int(row["metadata"]["decision_index"]))
    result = []
    previous = None
    for index, row in enumerate(ordered):
        before = current_state(str(row["messages"][1]["content"]))
        if previous is not None and before != previous:
            raise ValueError(f"{row['id']}: GT state chain disagrees at step {index}")
        call = row["messages"][2]["tool_calls"][0]["function"]
        feedback = json.loads(row["messages"][3]["content"])
        after = str(feedback.get("current_state") or feedback.get("derived_precursor") or "")
        if not after:
            raise ValueError(f"{row['id']}: GT tool result has no state")
        result.append({
            "index": index,
            "name": str(call["name"]),
            "arguments": dict(call["arguments"]),
            "before": before,
            "after": after,
            "accepted": True,
            "error": "",
        })
        previous = after
    return result


def predicted_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = list(episode.get("attempts") or [])
    result = []
    previous = None
    for index, attempt in enumerate(attempts):
        before = str(attempt["state_before"])
        if previous is not None and before != previous:
            raise ValueError(f"predicted state chain disagrees at attempt {index}")
        accepted = bool(attempt["accepted"])
        after = str(attempt.get("state_after") or "") if accepted else before
        if accepted and not after:
            raise ValueError(f"accepted prediction has no state at attempt {index}")
        result.append({
            "index": index,
            "name": str(attempt.get("name") or ""),
            "arguments": dict(attempt.get("arguments") or {}),
            "before": before,
            "after": after,
            "accepted": accepted,
            "error": str(attempt.get("error") or ""),
        })
        previous = after
    return result


def timeline(steps: list[dict[str, Any]], label: str, color: str) -> str:
    cards = []
    for step in steps:
        action = escape(action_text(step["name"], step["arguments"]))
        state = escape(step["after"])
        status = "accepted" if step["accepted"] else "rejected; state unchanged"
        error = f"<div class='error'>{escape(step['error'])}</div>" if step["error"] else ""
        cards.append(
            f"<div class='step {'rejected' if not step['accepted'] else ''}'>"
            f"<div class='step-head'><span class='step-num'>{step['index'] + 1:02d}</span>"
            f"<span>{action}</span></div>"
            f"<div class='step-status'>{status}</div>{error}"
            + f"<details><summary>State SMILES</summary><code>{state}</code></details>"
            "</div>"
        )
    return (
        f"<section class='timeline' style='--lane:{color}'><h3>{escape(label)}"
        f" <span class='count'>{len(steps)} decisions</span></h3>"
        + "".join(cards) + "</section>"
    )


def summary_card(smiles: str, title: str) -> str:
    return (
        f"<div class='summary-mol'><h3>{escape(title)}</h3>"
        f"<code>{escape(smiles)}</code></div>"
    )


def overview_png(cases: list[dict[str, Any]], output: Path) -> None:
    cell_width, cell_height = 530, 315
    canvas = Image.new("RGB", (cell_width * 3, cell_height * len(cases)), "white")
    pen = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for row_index, item in enumerate(cases):
        short = item["source_id"].replace("flower_mech_proof_val_", "val_")
        for column, (label, smiles) in enumerate((
            ("PRODUCT", item["product"]),
            ("RECORDED PRECURSOR", item["reference"]),
            (("PREDICTED" if item["terminal"] else "LAST STATE / NONTERMINAL"), item["display"]),
        )):
            print(f"[RDKit overview] {short} {label}", flush=True)
            x, y = column * cell_width, row_index * cell_height
            svg = molecule_svg(smiles, width=cell_width - 18, height=cell_height - 48)
            raster = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
            with Image.open(BytesIO(raster)) as rendered:
                canvas.paste(rendered.convert("RGB"), (x + 9, y + 36))
            pen.text((x + 12, y + 9), f"{short}  |  {label}", fill="#1d252b", font=font)
            pen.line((x, y + cell_height - 1, x + cell_width, y + cell_height - 1), fill="#d4d9db", width=1)
            pen.line((x + cell_width - 1, y, x + cell_width - 1, y + cell_height), fill="#d4d9db", width=1)
    canvas.save(output)


def case_png(item: dict[str, Any], output: Path) -> None:
    """One readable, directly previewable GT/model decision-lane image."""
    width, lane_width, top, step_height = 1560, 770, 370, 315
    height = top + step_height * max(len(item["gt"]), len(item["pred"])) + 24
    canvas = Image.new("RGB", (width, height), "white")
    pen = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        heading = ImageFont.truetype(font_path, 24)
        body = ImageFont.truetype(font_path, 16)
        small = ImageFont.truetype(font_path, 13)
    except OSError:
        heading = body = small = ImageFont.load_default()
    status = "RECORDED ENDPOINT MATCH" if item["hit"] else (
        "TERMINAL; DIFFERENT ENDPOINT" if item["terminal"] else "NONTERMINAL"
    )
    pen.text((18, 14), item["source_id"], fill="#1d252b", font=heading)
    pen.text((18, 49), f"Product-only K=1  |  {status}", fill="#4d5960", font=body)
    for column, (label, smiles) in enumerate((
        ("PRODUCT", item["product"]),
        ("RECORDED STRUCTURAL PRECURSOR", item["reference"]),
        ("PREDICTED PRECURSOR" if item["terminal"] else "LAST STATE (NOT A PRECURSOR)", item["display"]),
    )):
        x = column * 520
        pen.text((x + 14, 84), label, fill="#263c4a", font=small)
        svg = molecule_svg(smiles, width=500, height=210)
        raster = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        with Image.open(BytesIO(raster)) as rendered:
            canvas.paste(rendered.convert("RGB"), (x + 10, 108))
        pen.rectangle((x + 7, 79, x + 513, 327), outline="#d8dee0", width=1)
    pen.text((16, 339), f"GROUND TRUTH  |  {len(item['gt'])} decisions", fill="#355f86", font=heading)
    pen.text((lane_width + 16, 339), f"MODEL ROLLOUT  |  {len(item['pred'])} attempts", fill="#7a5139", font=heading)
    for column, (steps, lane_color) in enumerate((
        (item["gt"], "#355f86"), (item["pred"], "#7a5139")
    )):
        x = column * lane_width
        for index, step in enumerate(steps):
            y = top + index * step_height
            border = lane_color if step["accepted"] else "#a84d4d"
            pen.rectangle((x + 8, y + 5, x + lane_width - 10, y + step_height - 8), outline="#d7dde0", width=1)
            pen.rectangle((x + 8, y + 5, x + 13, y + step_height - 8), fill=border)
            pen.text((x + 22, y + 15), f"{index + 1:02d}", fill=border, font=heading)
            action_lines = textwrap.wrap(action_text(step["name"], step["arguments"]), width=72)
            for line_number, line in enumerate(action_lines[:3]):
                pen.text((x + 65, y + 13 + 21 * line_number), line, fill="#24313a", font=body)
            if not step["accepted"]:
                pen.text((x + 23, y + 91), "REJECTED; state unchanged", fill="#a84d4d", font=body)
                for line_number, line in enumerate(textwrap.wrap(step["error"], width=88)[:3]):
                    pen.text((x + 23, y + 120 + line_number * 19), line, fill="#7a3a3a", font=small)
                continue
            svg = molecule_svg(step["after"], width=lane_width - 35, height=216)
            raster = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
            with Image.open(BytesIO(raster)) as rendered:
                canvas.paste(rendered.convert("RGB"), (x + 18, y + 85))
    canvas.save(output)


def build(args: argparse.Namespace) -> dict[str, int]:
    episodes = list(rows(args.episodes))
    wanted_ids = {str(row["source_id"]) for row in episodes}
    sources = {
        str(row["source_id"]): row for row in rows(args.source)
        if str(row["source_id"]) in wanted_ids
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows(args.decisions):
        source_id = str(row["source_id"])
        if source_id in wanted_ids:
            grouped[source_id].append(row)
    if set(sources) != wanted_ids or set(grouped) != wanted_ids:
        raise ValueError("source and GT decision IDs do not cover the 16 episodes")
    if len(episodes) != 16 or len(wanted_ids) != 16:
        raise ValueError("this visualization requires the complete frozen 16-case smoke")
    cases = []
    hit_count = 0
    for row in episodes:
        source_id = str(row["source_id"])
        episode = row["episodes"][0]
        gold = gt_steps(grouped[source_id])
        pred = predicted_steps(episode)
        if pred and gold and pred[0]["before"] != gold[0]["before"]:
            raise ValueError(f"{source_id}: initial visible state differs")
        reference = reference_structural_precursor(sources[source_id])
        if not reference:
            raise ValueError(f"{source_id}: missing recorded structural precursor")
        terminal = bool(episode["terminal"])
        display = str(episode.get("precursor") or "") if terminal else pred[-1]["after"]
        if not display:
            raise ValueError(f"{source_id}: no predicted state to depict")
        hit = terminal and bool(episode["has_electron_event"]) and bool(episode.get("precursor"))
        if hit:
            from mechet.endpoints import structural_exact
            hit = structural_exact(str(episode["precursor"]), reference)
        hit_count += int(hit)
        cases.append({
            "source_id": source_id,
            "product": gold[0]["before"],
            "reference": reference,
            "display": display,
            "terminal": terminal,
            "hit": hit,
            "gt": gold,
            "pred": pred,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overview_png(cases, args.output.with_name("stage1_valid16_overview.png"))
    sections = []
    for index, item in enumerate(cases, 1):
        print(f"[RDKit trace] {index}/16 {item['source_id']}", flush=True)
        short = item["source_id"].replace("flower_mech_proof_val_", "val_")
        case_image = f"stage1_valid16_case_{index:02d}_{short}.png"
        case_png(item, args.output.with_name(case_image))
        status = "recorded endpoint matched" if item["hit"] else (
            "terminal; different recorded endpoint" if item["terminal"] else "nonterminal"
        )
        sections.append(
            f"<article id='case-{index}'><div class='case-title'><h2>"
            f"{index:02d} · {escape(item['source_id'])}</h2>"
            f"<span class='badge'>{escape(status)} · <a href='{case_image}'>PNG</a></span></div>"
            f"<img class='case-image' src='{case_image}' alt='GT and model decision trajectory for {escape(item['source_id'])}'>"
            f"<details class='text-detail'><summary>Full action text and state SMILES</summary>"
            f"<div class='summary-grid'>"
            + summary_card(item["product"], "Input product")
            + summary_card(item["reference"], "Recorded structural precursor")
            + summary_card(item["display"], "Predicted precursor" if item["terminal"] else "Last predicted state (not a precursor)")
            + "</div><div class='lanes'>"
            + timeline(item["gt"], "Ground truth · frozen executor replay", "#355f86")
            + timeline(item["pred"], "Model · product-only rollout", "#7a5139")
            + "</div></details></article>"
        )
    links = " ".join(
        f"<a href='#case-{i}'>{i:02d}</a>" for i in range(1, 17)
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MechET Stage-I validation smoke · GT versus predicted traces</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; background:#f5f5f2; color:#1d252b; font:15px/1.5 Arial,Helvetica,sans-serif; }}
main {{ max-width:1580px; margin:auto; padding:30px 22px 70px; }}
h1,h2,h3 {{ font-family:Georgia,'Times New Roman',serif; font-weight:600; }}
h1 {{ font-size:31px; margin:0 0 8px; }} h2 {{ font-size:23px; margin:0; }} h3 {{ font-size:17px; margin:0; }}
.intro {{ max-width:1100px; color:#4c565e; margin:0 0 16px; }}
nav {{ display:flex; flex-wrap:wrap; gap:10px; margin:17px 0 22px; }}
nav a {{ border:1px solid #bdc6cb; padding:5px 9px; color:#23455c; text-decoration:none; background:white; }}
article {{ background:white; border:1px solid #d7dde0; margin:25px 0; padding:18px; break-inside:avoid; }}
.case-title {{ display:flex; align-items:center; justify-content:space-between; gap:12px; border-bottom:1px solid #d7dde0; padding-bottom:10px; margin-bottom:13px; }}
.badge {{ border:1px solid #aebdc6; padding:4px 9px; font-size:12px; white-space:nowrap; }}
.summary-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:16px; }}
.summary-mol {{ border:1px solid #d8dfe2; padding:10px; min-width:0; }}
.summary-mol h3 {{ margin-bottom:6px; }}
.case-image {{ display:block; width:100%; height:auto; border:1px solid #d8dfe2; }}
.text-detail {{ margin-top:14px; }}
.lanes {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:17px; align-items:start; }}
.timeline {{ border-top:3px solid var(--lane); min-width:0; }}
.timeline>h3 {{ padding:9px 2px 11px; }} .count {{ font:12px Arial,sans-serif; color:#66737b; margin-left:6px; }}
.step {{ border:1px solid #d7dde0; margin:0 0 10px; padding:10px; border-left:3px solid var(--lane); }}
.step.rejected {{ border-left-color:#a84d4d; background:#fffaf9; }}
.step-head {{ display:flex; gap:10px; align-items:flex-start; font-weight:600; overflow-wrap:anywhere; }}
.step-num {{ color:var(--lane); font:600 13px Arial,sans-serif; min-width:24px; }}
.step-status {{ color:#5c6770; font-size:12px; margin:4px 0 6px 34px; }}
.error {{ color:#9a2d2d; margin:4px 0 8px 34px; overflow-wrap:anywhere; }}
details {{ margin-top:5px; }} summary {{ color:#52636b; cursor:pointer; font-size:12px; }}
code {{ display:block; white-space:pre-wrap; overflow-wrap:anywhere; font:11px/1.4 ui-monospace,monospace; padding:6px; background:#f6f7f7; }}
@media(max-width:1000px) {{ .summary-grid,.lanes {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>MechET Stage-I · 16-case validation smoke</h1>
<p class="intro">RDKit depictions of the frozen ground-truth tool trajectory and the actual product-only K=1 rollout from provisional checkpoint-14000. {hit_count}/16 predicted terminal trajectories match the recorded structural precursor. Columns are independent after divergence; rows do not imply state equivalence. Imported context remains visible in trajectory states; the endpoint comparison uses the structural precursor view.</p>
<p class="intro">Blue = ground truth; brown = model. A rejected proposal produces no new state. Molecular drawings hide private atom-map numbers. Electron-flow arrows are written from the saved source/destination arguments; no geometric curved arrow is inferred from an unmapped drawing.</p>
<p class="intro">Source: frozen strict-executable FlowER validation decisions and saved checkpoint-14000 episode shard. This is a 16-case provisional diagnostic, not full validation/test accuracy or a claim of laboratory feasibility.</p>
<nav>{links}</nav>
{''.join(sections)}
</main></body></html>"""
    args.output.write_text(page, encoding="utf-8")
    return {"cases": len(cases), "endpoint_hits": hit_count,
            "gt_decisions": sum(len(c["gt"]) for c in cases),
            "predicted_attempts": sum(len(c["pred"]) for c in cases)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
