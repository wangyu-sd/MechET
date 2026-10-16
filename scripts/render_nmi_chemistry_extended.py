#!/usr/bin/env python3
"""Vector Extended Data from recorded chemical deltas, exposure and MCTS visits."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.analysis.enrichment_summary import summarize_enrichment
from mechet.analysis.tree_charts import chart_svg
from mechet.analysis.tree_records import file_digest
from mechet.analysis.figure_qa import inspect_figure
from scripts.render_nmi_tree_figures import svg_document, label, embed, export
from scripts.analyze_nmi_tree_evidence import _write_json


def render_summary(summary, style):
    def ecdf(histograms, xlabel, width=510):
        def draw(ax):
            for histogram, name, color in histograms:
                total = sum(histogram.values())
                if not total:
                    continue
                cumulative = 0
                xs, ys = [], []
                for key in sorted(histogram, key=float):
                    cumulative += histogram[key]
                    xs.append(float(key)); ys.append(cumulative / total)
                ax.step(xs, ys, where="post", label=f"{name} (n={total:,})", color=style["palette"][color], linewidth=1.35)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Recorded fraction")
            ax.set_ylim(0, 1.03)
            ax.set_xscale("symlog", linthresh=1)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=6, loc="lower right")
            else:
                ax.text(.5, .5, "Visits not recorded", ha="center", transform=ax.transAxes)
        return chart_svg(style, width, 190, draw)
    local = ecdf([(summary["changed_atoms"], "Changed atoms", "search_blue_dark"),
        (summary["changed_bonds"], "Changed bonds", "search_blue")], "Graph change per distinct accepted FLOW transition")
    local_doc = svg_document(label(6, 12, "Descriptive symbolic reaction-centre changes") + embed(local, 5, 20, 510, 190)
        + label(6, 222, "Bond, charge and component deltas describe executor states; they do not certify laboratory feasibility."), style, height_pt=238)
    classes = summary["program_classes"]
    names = [("seen_ordered_signature", "Seen ordered signature"), ("unseen_ordered_signature", "Unseen ordered signature"),
        ("unavailable_or_unauthenticated", "Not establishable")]
    def draw_programs(ax):
        ax.set_position([.34, .27, .57, .65])
        ax.barh(range(3), [classes.get(k, 0) for k, _ in names], height=.55,
            color=[style["palette"][k] for k in ["neutral_dark", "search_blue", "unknown_gray"]])
        ax.set_yticks(range(3), [v for _, v in names], fontsize=6)
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Terminal contexts under authenticated actual-actor exposure")
    program_doc = svg_document(label(6, 12, "Generated complete-program signature support")
        + embed(chart_svg(style, 510, 190, draw_programs), 5, 20, 510, 190)
        + label(6, 222, "An unseen generated hash is neither route-level C1 nor independent chemical precedent."), style, height_pt=238)
    visits = ecdf([(summary["planning_visits"], "Actual planning visits", "search_blue_dark")], "Recorded MCTS visits per planning-context snapshot")
    visits_doc = svg_document(label(6, 12, "Native MCTS planning statistics") + embed(visits, 5, 20, 510, 190)
        + label(6, 222, f'Missing visit fields: {summary["planning_visits_missing"]:,}; shared snapshots are counted per target, not globally distinct states.'), style, height_pt=238)
    return {"extended_data_local_chemistry": local_doc, "extended_data_program_support": program_doc, "extended_data_mcts_visits": visits_doc}


def render(root, style_path, *, run_id):
    import yaml
    root = Path(root)
    manifest_path = root / "enrichment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["input_inventory_sha256"] != file_digest(root / "input_inventory.json"):
        raise ValueError("enrichment inventory no longer matches")
    paths = {}
    for name, record in manifest["tables"].items():
        path = root / record["path"]
        if file_digest(path) != record["sha256"]:
            raise ValueError("enrichment table changed")
        paths[name] = path
    def rows(name):
        with gzip.open(paths[name], "rt") as f:
            for line in f:
                yield json.loads(line)
    summary = summarize_enrichment(rows("edge_chemistry"), rows("programs"), rows("planning_states"), run_id=run_id)
    style = yaml.safe_load(Path(style_path).read_text())
    figures = {name: export(svg, root / "figures" / name, style) for name, svg in render_summary(summary, style).items()}
    qa = {name: inspect_figure(files, style) for name, files in figures.items()}
    delivery = {"schema": "nmi_chemistry_figure_manifest_v1", "run_id": run_id, "summary": summary,
        "enrichment_manifest_sha256": file_digest(manifest_path), "figures": figures, "qa": qa,
        "source_tables": manifest["tables"], "style_sha256": file_digest(style_path),
        "producer_files": {str(path.relative_to(ROOT)): file_digest(path) for path in [Path(__file__), ROOT / "src/mechet/analysis/enrichment_summary.py"]},
        "new_sampling": False, "route_C1_claim": False}
    _write_json(root / "chemistry_figure_manifest.json", delivery)
    if not all(v["machine_checks_passed"] for v in qa.values()):
        raise ValueError("chemistry figure QA failed")
    return delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    delivery = render(args.analysis_root, args.style, run_id=args.run_id)
    print(json.dumps({"phase": "chemistry-vectors", "figures": list(delivery["figures"])}), flush=True)


if __name__ == "__main__":
    main()
