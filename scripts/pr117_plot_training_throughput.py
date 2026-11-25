"""Draw measured stage times and historical credit from a verified training audit."""
import argparse
import json
from pathlib import Path


def render(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    rows = summary["rows"]
    step = [r["update"] for r in rows]
    totals = summary["totals"]
    fig, (runtime, history) = plt.subplots(2, 1, figsize=(8.1, 5.25), sharex=True,
                                          gridspec_kw={"height_ratios": [2.1, 1.25]})
    bottom = [0.0] * len(rows)
    for field, label, color in (
        ("collection_seconds", "Collect rollouts", "#3C658D"),
        ("receipt_read_seconds", "Load receipts", "#ACC2D2"),
        ("learner_compute_seconds", "Update weights", "#BD7548"),
    ):
        visible = [r[field]/60 if r["update"] > 1 else 0 for r in rows]
        runtime.bar(step, visible, bottom=bottom, width=0.72, color=color, label=label, zorder=3)
        bottom = [prior + current for prior, current in zip(bottom, visible, strict=True)]
    runtime.set_title("a   Recorded time by training phase", loc="left", weight="bold", pad=9)
    runtime.set_ylabel("Minutes per update")
    runtime.set_ylim(0, max(bottom)*1.18)
    runtime.legend(loc="upper center", ncol=3, frameon=False, handlelength=1.3, columnspacing=1.5)

    observed = [r["historical_comparison_groups"] for r in rows]
    color = "#176652"
    history.hlines(0, 0.45, step[-1] + 0.55, linewidth=0.75, color="#98A4AA", zorder=2)
    history.vlines(step, 0, observed, color=color, linewidth=1.35, zorder=3)
    history.scatter(step, observed, color=color, s=20, edgecolors="white", linewidths=0.4, zorder=4)
    history.set_title("b   Groups with an observed same-state history contrast", loc="left", weight="bold", pad=9)
    history.set_ylabel("Groups per update")
    history.set_xlabel("Optimizer update")
    history.set_ylim(-0.18, max(observed, default=0)+0.92)
    history.yaxis.set_major_locator(MaxNLocator(integer=True))
    history.set_xlim(0.45, step[-1] + 0.55)
    ticks = sorted({step[0], step[-1], *range(5, step[-1], 5)})
    history.set_xticks(ticks)
    for ax in (runtime, history):
        ax.grid(axis="y", linewidth=0.55, color="#DFE5E8", zorder=0)
        ax.spines["left"].set_color("#9EA8AE")
        ax.spines["bottom"].set_color("#9EA8AE")
        ax.tick_params(colors="#303C45", length=3)
    fig.text(0.115, 0.051, "Training observations. The timing panel begins at update 2 after collection resumed.",
             fontsize=8, color="#52616A")
    fig.text(0.115, 0.026,
             f"{totals['physical_calls']:,} total model calls, including {totals['recovered_original_calls']:,} reused at update 1. "
             f"History contrast in {totals['historical_comparison_groups']:,} of {totals['policy_candidate_groups']:,} candidate groups.",
             fontsize=8, color="#52616A")
    fig.subplots_adjust(left=0.115, right=0.97, top=0.93, bottom=0.175, hspace=0.48)
    fig.savefig(output / "training_phases_and_credit.pdf", metadata={"Title": "MechET training phases and observed history credit"})
    fig.savefig(output / "training_phases_and_credit.png", dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    source = args.summary.resolve()
    if not source.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Use the audit and output on shared storage")
    for suffix in ("pdf", "png"):
        if (source.parent / f"training_phases_and_credit.{suffix}").exists():
            parser.error("Existing plot needs a new analysis directory")
    render(json.loads(source.read_bytes()), source.parent)


if __name__ == "__main__":
    main()
