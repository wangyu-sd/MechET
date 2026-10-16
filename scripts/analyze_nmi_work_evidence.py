#!/usr/bin/env python3
"""Render source-recorded per-target work without inferring global GPU costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.analysis.tree_records import canonical_entries, file_digest
from mechet.analysis.tree_costs import observed_work_metrics
from mechet.analysis.tree_charts import chart_svg
from mechet.analysis.figure_qa import inspect_figure
from scripts.analyze_nmi_tree_evidence import _write_json
from scripts.render_nmi_tree_figures import svg_document, label, embed, export


def work_svg(costs, style):
    def ecdf(fields, xlabel):
        def draw(ax):
            for key, name, color in fields:
                values = sorted(row["values"][key] for row in costs["per_target"] if row["values"][key] is not None)
                if values:
                    ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post",
                        label=f"{name} (n={len(values)})", color=style["palette"][color], linewidth=1.35)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Observed-target fraction")
            ax.set_ylim(0, 1.03)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(left=0)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=5.5, loc="lower right")
            else:
                ax.text(.5, .5, "Timing not recorded", ha="center", transform=ax.transAxes)
        return chart_svg(style, 250, 190, draw)
    tokens = ecdf([("sampled_tokens", "Sampled tokens", "search_blue_dark"),
        ("prompt_tokens", "Prompt tokens", "search_blue"),
        ("reused_output_tokens", "Reused output tokens", "neutral_dark")], "Recorded tokens per target")
    times = ecdf([("wall_seconds", "Target elapsed", "search_blue_dark"),
        ("cpu_executor_wall_seconds", "Executor wall", "search_blue"),
        ("cpu_executor_thread_seconds", "Executor thread", "neutral_dark")], "Recorded seconds per target")
    return svg_document(label(6, 12, "a", size=8, weight=600) + label(26, 12, "Recorded token charges")
        + label(268, 12, "b", size=8, weight=600) + label(288, 12, "Recorded elapsed and executor time")
        + embed(tokens, 2, 22, 250, 190) + embed(times, 264, 22, 250, 190)
        + label(6, 226, "Shared reuse is retained per target; these records do not provide additive global GPU work.")
        + label(6, 238, "Missing values are excluded, not zero-filled. Elapsed and executor fields are not an additive decomposition."),
        style, height_pt=252)


def build_work_report(analysis_root, style_path, *, run_id):
    import yaml
    root = Path(analysis_root)
    inventory_path = root / "input_inventory.json"
    source_path = root / "source_manifest.json"
    source = json.loads(source_path.read_text())
    if source["input_inventory_sha256"] != file_digest(inventory_path):
        raise ValueError("work inventory no longer matches source manifest")
    inventory = json.loads(inventory_path.read_text())
    all_entries = canonical_entries(inventory["inputs"])
    entries = [entry for entry in all_entries if entry["run_id"] == run_id]
    if not entries:
        raise ValueError("run has no canonical source reports: " + run_id)
    # Recheck every source report used for charged work, not merely the inventory.
    for entry in entries:
        if file_digest(entry["report_path"]) != entry["report_sha256"]:
            raise ValueError("work source report changed: " + entry["target_id"])
    costs = observed_work_metrics(entries)
    costs_by_run = {key: observed_work_metrics([entry for entry in all_entries if entry["run_id"] == key])
        for key in sorted({entry["run_id"] for entry in all_entries})}
    style = yaml.safe_load(Path(style_path).read_text())
    figures = {"extended_data_observed_work": export(work_svg(costs, style), root / "figures/extended_data_observed_work", style)}
    qa = {name: inspect_figure(files, style) for name, files in figures.items()}
    result = {"schema": "nmi_observed_work_manifest_v1", "run_id": run_id, "costs": costs,
        "costs_by_run": costs_by_run, "cross_run_cost_pooling": False,
        "secondary_cost_source_status": "frozen report contents in hash-bound input inventory; primary plotted reports rechecked against original files",
        "input_inventory_sha256": file_digest(inventory_path), "source_manifest_sha256": file_digest(source_path),
        "figures": figures, "qa": qa, "style_sha256": file_digest(style_path),
        "producer_files": {str(path.relative_to(ROOT)): file_digest(path) for path in [Path(__file__), ROOT / "src/mechet/analysis/tree_costs.py"]},
        "new_sampling": False, "global_GPU_work_inferred": False, "matched_compute_gain_claim": False}
    _write_json(root / "work_manifest.json", result)
    if not all(check["machine_checks_passed"] for check in qa.values()):
        raise ValueError("observed work figure QA failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = build_work_report(args.analysis_root, args.style, run_id=args.run_id)
    print(json.dumps({"phase": "observed-work", "targets": result["costs"]["observed_targets"], "fields": result["costs"]["fields"]}), flush=True)


if __name__ == "__main__":
    main()
