"""Publication-scale vector panels; data preparation stays source-addressable."""
from __future__ import annotations

import io


def chart_svg(style, width, height, draw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in [style["fonts"]["family"], *style["fonts"]["fallbacks"]] if f in installed), "DejaVu Sans")
    with plt.rc_context({"font.family": family, "font.size": style["fonts"]["figure_text_pt"],
            "axes.labelsize": style["fonts"]["figure_text_pt"], "xtick.labelsize": style["fonts"]["tick_label_pt"],
            "ytick.labelsize": style["fonts"]["tick_label_pt"], "text.color": style["palette"]["ink"],
            "axes.labelcolor": style["palette"]["ink"], "axes.edgecolor": style["palette"]["ink"],
            "axes.linewidth": style["charts"]["axis_line_pt"], "axes.spines.top": False,
            "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "svg.hashsalt": "mechet-nmi-tree-evidence-v1"}):
        figure = plt.figure(figsize=(width / 72, height / 72))
        axis = figure.add_axes([.15, .28, .81, .64])
        draw(axis)
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", transparent=True, metadata={"Date": None})
        plt.close(figure)
        return buffer.getvalue()


def route_chart(panel, style, width=510, height=125):
    rows = list(panel["cohorts"].items())
    labels = {"ursa_expert100": "URSA Expert", "ursa_drugs100": "URSA Drugs", "retrocast_mkt_cnv160": "RetroCast", "paroutes_derived_n5_100": "PaRoutes"}
    def draw(ax):
        ax.bar(range(len(rows)), [100 * r["lower_bound_rate"] for _, r in rows],
            color=style["palette"]["search_blue"], width=.6, hatch="//", edgecolor=style["palette"]["search_blue_dark"], linewidth=.45)
        ax.set_xticks(range(len(rows)), [labels.get(k, k) for k, _ in rows])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Stock closure lower bound (%)")
        for i, (_, r) in enumerate(rows):
            ax.text(i, 100 * r["lower_bound_rate"] + 4, f'{r["observed_nonzero_stock_connected"]}/{r["denominator"]}', ha="center", fontsize=6)
        ax.set_xlabel("All frozen targets retained; unfinished searches are pending")
    return chart_svg(style, width, height, draw)
def empirical_chart(rows, style, width=250, height=140):
    def draw(ax):
        for field, label, color in [("native_contexts", "Contexts", "search_blue_dark"), ("terminal_contexts", "Terminals", "terminal_teal_dark")]:
            values = sorted(r["summary"]["morphology"][field] for r in rows)
            ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", label=label, color=style["palette"][color], linewidth=1.35)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Recorded count per target")
        ax.set_ylabel("Observed-target fraction")
        ax.set_ylim(0, 1.03)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    return chart_svg(style, width, height, draw)


def depth_chart(panel, style, width=260, height=155):
    rows = list(panel["depth"].items())
    def draw(ax):
        ax.bar(range(len(rows)), [100 * r["lower_bound_rate"] for _, r in rows], color=style["palette"]["search_blue"], width=.55)
        ax.set_xticks(range(len(rows)), [k for k, _ in rows])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Stock closure lower bound (%)")
        ax.set_xlabel("Reference-route depth")
        for i, (_, r) in enumerate(rows):
            ax.text(i, 100 * r["lower_bound_rate"] + 3, f'{r["observed_nonzero_stock_connected"]}/{r["denominator"]}', ha="center", fontsize=6)
    return chart_svg(style, width, height, draw)


def growth_chart(report, style, width=240, height=155):
    points = report["vocabulary_growth"]
    if not points or points[-1]["unique_primitives"] != 604 or points[-1]["unique_compositions"] != 6553:
        raise ValueError("not the authenticated historical v1 census")
    def draw(ax):
        for key, label, color, marker in [("unique_compositions", "Ordered programs", "search_blue_dark", "o"), ("unique_primitives", "Local signatures", "neutral_dark", "s")]:
            ax.plot([100 * p["hash_sample_fraction"] for p in points], [p[key] for p in points], marker=marker, markersize=2.5, linewidth=1.35, color=style["palette"][color], label=label)
        ax.set_yscale("log")
        ax.set_ylim(200, 10000)
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Historical training-pool coverage (%)")
        ax.set_ylabel("Distinct signatures (log)")
        ax.legend(frameon=False, fontsize=5.5, loc="upper left")
        ax.annotate("6,553", (100, 6553), xytext=(-3, 5), textcoords="offset points", ha="right", fontsize=6)
        ax.annotate("604", (100, 604), xytext=(-3, 5), textcoords="offset points", ha="right", fontsize=6)
    return chart_svg(style, width, height, draw)


def ranking_chart(rows, style, width=510, height=180):
    labeled = [r for r in rows if r["summary"]["ranking"]["pool_recall"] is not None]
    if not labeled:
        raise ValueError("no labeled candidate pools")
    n = len(labeled)
    keys = [("top1", "NLL Top-1"), ("top5", "NLL Top-5"), ("top10", "NLL Top-10"), ("pool_recall", "Recorded pool recall")]
    counts = [sum(r["summary"]["ranking"][k] for r in labeled) for k, _ in keys]
    def draw(ax):
        ax.bar(range(len(keys)), [100 * c / n for c in counts], width=.6, color=[style["palette"]["search_blue_dark"], style["palette"]["search_blue"], style["palette"]["neutral_dark"], style["palette"]["terminal_teal_dark"]])
        ax.set_xticks(range(len(keys)), [label for _, label in keys])
        ax.set_ylabel("Observed-target endpoint hit (%)")
        ax.set_ylim(0, max(100 * max(counts) / n * 1.4, 1))
        ax.set_xlabel(f"Same recorded pools; observed n = {n:,}; rxn_prod_min proxy diagnostic")
        for i, c in enumerate(counts):
            ax.annotate(f"{c}/{n:,}", (i, 100 * c / n), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6)
    return chart_svg(style, width, height, draw)


def morphology_chart(rows, style, width=510, height=200):
    def draw(ax):
        for field, name, color in [("native_contexts", "Contexts", "search_blue_dark"),
                ("unique_context_transitions", "Distinct transitions", "search_blue"),
                ("terminal_contexts", "Terminals", "terminal_teal_dark"),
                ("flow_fork_contexts", "FLOW fork states", "neutral_dark")]:
            values = sorted(r["summary"]["morphology"][field] for r in rows)
            ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", label=name,
                color=style["palette"][color], linewidth=style["charts"]["data_line_pt"])
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel(f"Native count per target (observed n = {len(rows):,})")
        ax.set_ylabel("Observed-target fraction")
        ax.set_ylim(0, 1.03)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    return chart_svg(style, width, height, draw)


def error_chart(rows, style, width=510, height=200):
    from collections import Counter
    counts = Counter()
    for row in rows:
        for message, count in row["summary"]["morphology"]["error_counts"].items():
            counts[str(message).split(":", 1)[0] or "Unspecified"] += count
    ranked = counts.most_common(7)
    if len(counts) > 7:
        ranked.append(("Other recorded errors", sum(counts.values()) - sum(n for _, n in ranked)))
    def draw(ax):
        ax.set_position([.38, .25, .57, .68])
        ax.barh(range(len(ranked)), [n for _, n in ranked], color=style["palette"]["rejected_clay"], height=.55)
        ax.set_yticks(range(len(ranked)), [name[:45] for name, _ in ranked], fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel("Rejected attempts (not successor states)")
        ax.set_xscale("symlog", linthresh=1)
    return chart_svg(style, width, height, draw)


def quality_chart(rows, style, width=510, height=190):
    fields = [("attempted_actions", "Attempted actions"), ("unique_context_transitions", "Distinct accepted transitions"),
        ("terminal_contexts", "Complete terminal contexts")]
    counts = [sum(r["summary"]["morphology"][field] for r in rows) for field, _ in fields]
    def draw(ax):
        ax.set_position([.36, .26, .56, .66])
        ax.barh(range(3), counts, height=.55, color=[style["palette"][k] for k in ["neutral_dark", "search_blue", "terminal_teal_dark"]])
        ax.set_yticks(range(3), [name for _, name in fields], fontsize=6)
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Counts at different counting units; not one success rate", fontsize=6)
        for i, count in enumerate(counts):
            ax.annotate(f"{count:,}", (count, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6)
        ax.set_xlim(0, max(counts, default=1) * 4 + 1)
    return chart_svg(style, width, height, draw)
