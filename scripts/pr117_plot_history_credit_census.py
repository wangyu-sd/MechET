"""Export the observed candidate funnel and paired local teacher scores."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        table = csv.DictWriter(f, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def plot(pairs, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.2, 5.7))
    scores = [p[k] for p in pairs for k in ("closed_teacher_score", "failed_teacher_score")]
    low, high = min(scores), max(scores)
    padding = max((high-low)*.085, 0.01)
    limits = (low-padding, high+padding)
    ax.plot(limits, limits, color="#87929A", linewidth=.9, linestyle="--", zorder=1)
    for category, color, label in (
        ("failed", "#B65032", "Teacher favored failed branch"),
        ("closed", "#226B68", "Teacher favored closed branch"),
        ("tie", "#76808A", "Tied teacher scores"),
    ):
        items = [p for p in pairs if p["teacher_preference"] == category]
        if items:
            ax.scatter([p["failed_teacher_score"] for p in items], [p["closed_teacher_score"] for p in items],
                       color=color, s=44, edgecolors="white", linewidths=.6,
                       label=f"{label} ({len(items)})", zorder=3)
    ax.set(xlabel="Teacher score of failed branch", ylabel="Teacher score of stock-closed branch",
           xlim=limits, ylim=limits, aspect="equal")
    ax.grid(linewidth=.48, color="#E3E7EB", zorder=0)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .074),
               frameon=False, fontsize=9)
    ax.set_title("Local teacher scores at one electronic state", loc="left", fontsize=12, weight="bold", pad=13)
    fig.text(.5, .051, f"{len(pairs)} observed FLOW pairs with nonzero history contrast in training.",
             fontsize=9, color="#53616A", ha="center")
    fig.subplots_adjust(left=.17, right=.96, top=.89, bottom=.27)
    fig.savefig(output / "same_state_flow_teacher_scores.pdf")
    fig.savefig(output / "same_state_flow_teacher_scores.png", dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("census", "widths", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Use a shared evidence directory")
    census, widths = (json.loads(path.read_bytes()) for path in (args.census, args.widths))
    if ([r["complete_sha256"] for r in census["rounds"]] != [r["complete_sha256"] for r in widths["rows"]]
            or census["historical_comparison_groups"] != widths["history_contrasts"]):
        raise ValueError("Expected the same completed learner population")
    by_width = Counter(row["candidate_count"] for row in census["rows"])
    counts = [dict(model_responses_per_group=int(width), all_candidate_groups=n,
                   nonzero_observed_history_groups=by_width[int(width)])
              for width, n in sorted(widths["candidate_groups_by_width"].items(), key=lambda v: int(v[0]))]
    pairs = []
    for row in census["rows"]:
        closed = [a for a in row["actions"] if a["outcome"] == "closed" and a["tool"] == "apply_electron_flow"]
        failed = [a for a in row["actions"] if a["outcome"] == "failed" and a["tool"] == "apply_electron_flow"]
        for a in closed:
            for b in failed:
                x, y = b["teacher_score"], a["teacher_score"]
                if x is None or y is None:
                    continue
                preference = "failed" if x > y else "closed" if y > x else "tie"
                pairs.append(dict(update=row["update"], target_id=row["target_id"], lane=row["train_lane"],
                                  reference_verified=row["reference_verified"], closed_teacher_score=y,
                                  failed_teacher_score=x, teacher_preference=preference))
    if len(pairs) != census["pairwise"]["flow_pairs_with_two_teacher_scores"] or not pairs:
        raise ValueError("Paired score census changed")
    write_csv(output / "candidate_width.csv", counts)
    write_csv(output / "same_state_flow_teacher_pairs.csv", pairs)
    plot(pairs, output)
    print(json.dumps({"candidate_groups": counts,
                      "preferences": dict(Counter(p["teacher_preference"] for p in pairs))}))


if __name__ == "__main__":
    main()
