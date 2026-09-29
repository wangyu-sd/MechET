#!/usr/bin/env python3
"""Render an unlabeled, hash-bound chemistry review packet for R2 candidates."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest


def _svg(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f'<p class="smiles">Cannot depict: {html.escape(smiles)}</p>'
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 230)
    options = drawer.drawOptions()
    options.bondLineWidth = 2.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def build(source: Path, output: Path, *, expected_candidates: int = 119) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R2 review packet already exists: {output}")
    manifest_path = source.parent / "manifest.json"
    status_path = source.parent / "ARTIFACT_STATUS.json"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    source_sha = digest(source)
    if (manifest.get("cohort_sha256") != source_sha
            or manifest.get("candidate_count") != expected_candidates
            or status.get("evaluation_allowed") is not False
            or status.get("training_allowed") is not False):
        raise ValueError("R2 review source is not the frozen unlabeled queue")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) != expected_candidates:
        raise ValueError("R2 review denominator drifted")
    ids: set[str] = set()
    for row in rows:
        cid = str(row.get("candidate_id") or "")
        if (not cid or cid in ids or row.get("chemical_negative_label") is not None
                or row.get("review_status") != "pending_independent_chemistry_review"):
            raise ValueError("R2 review row is duplicate or already labeled")
        ids.add(cid)

    output.mkdir(parents=True)
    worksheet_path = output / "review_template.csv"
    fields = ["candidate_id", "product_smiles", "proposed_precursors",
              "known_recorded_precursors", "source_trace_reaction_id",
              "event_mutation", "review_finding", "reviewer_id",
              "evidence_kind", "evidence_locator", "rationale"]
    with worksheet_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            context = row["review_context"]
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "product_smiles": row["model_input"]["product_smiles"],
                "proposed_precursors": row["model_input"]["proposed_precursors"],
                "known_recorded_precursors": " || ".join(context["known_recorded_precursors"]),
                "source_trace_reaction_id": row["source_trace_reaction_id"],
                "event_mutation": json.dumps(context["event_mutation"], sort_keys=True),
                "review_finding": "", "reviewer_id": "", "evidence_kind": "",
                "evidence_locator": "", "rationale": "",
            })

    cards = []
    for index, row in enumerate(rows, 1):
        context = row["review_context"]
        product = str(row["model_input"]["product_smiles"])
        proposal = str(row["model_input"]["proposed_precursors"])
        references = [str(item) for item in context["known_recorded_precursors"]]
        figures = [f'<figure><figcaption>Product</figcaption>{_svg(product)}</figure>',
                   f'<figure><figcaption>Candidate precursor</figcaption>{_svg(proposal)}</figure>']
        figures.extend(f'<figure><figcaption>Recorded precursor {j}</figcaption>{_svg(ref)}</figure>'
                       for j, ref in enumerate(references, 1))
        cards.append(
            f'<section><h2>{index}. {html.escape(row["candidate_id"][:16])}</h2>'
            f'<p>Source trace: {html.escape(str(row["source_trace_reaction_id"]))}; '
            f'event mutation: {html.escape(json.dumps(context["event_mutation"], sort_keys=True))}</p>'
            f'<div class="figures">{"".join(figures)}</div>'
            f'<p><strong>Review status:</strong> unlabelled; independent evidence required.</p></section>')
    packet_path = output / "review_packet.html"
    packet_path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>R2 executor-valid counterfactual review</title>'
        '<style>body{font:15px Arial,sans-serif;max-width:1220px;margin:28px auto;color:#111}'
        'section{border-top:1px solid #bbb;padding:18px 0 26px}h1,h2{font-weight:600}'
        'h2{font-size:18px}.figures{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}'
        'figure{margin:0;border:1px solid #ddd;padding:8px;overflow:auto}'
        'figcaption{font-weight:600;margin:0 0 6px}svg{max-width:100%;height:auto}'
        '.smiles{overflow-wrap:anywhere}</style>'
        f'<h1>R2 review candidates ({len(rows)})</h1>'
        '<p>Executor acceptance and absence from the recorded reference set do not '
        'establish chemical invalidity. No labels are supplied. Use independent '
        'evidence before filling a copy of the CSV worksheet.</p>'
        + "".join(cards) + '</html>', encoding="utf-8")
    packet_manifest = {
        "artifact_type": "r2_unlabeled_chemistry_review_packet_v1",
        "source_sha256": source_sha,
        "source_manifest_sha256": digest(manifest_path),
        "source_status_sha256": digest(status_path),
        "cases": len(rows),
        "review_packet_sha256": digest(packet_path),
        "review_template_sha256": digest(worksheet_path),
        "chemical_labels_provided": False,
        "evaluation_allowed": False,
        "claim_boundary": "Visualization and blank review worksheet only; neither source nor packet proves a chemical negative.",
    }
    (output / "manifest.json").write_text(json.dumps(packet_manifest, indent=2, sort_keys=True) + "\n")
    return packet_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
