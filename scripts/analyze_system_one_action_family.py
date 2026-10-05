#!/usr/bin/env python3
"""Inspect Phase-1a action routing without rerunning the frozen Qwen encoder."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.system_one_action_family import ACTION_NAMES, ActionFamilyHead
from scripts.train_system_one_action_family import file_sha256, load_split, metrics


def calibration(probabilities: list[list[float]], labels: list[int]) -> dict:
    """Report top-class calibration and exact-route selective accuracy."""
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probability and label rows must match and be nonempty")
    bins = []
    for lower in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        upper = 0.5 if lower == 0.0 else (1.0 if lower == 0.9 else lower + 0.1)
        selected = [i for i, p in enumerate(probabilities)
                    if lower <= max(p) < upper or (upper == 1.0 and max(p) == 1.0)]
        bins.append({
            "confidence_range": [lower, upper],
            "n": len(selected),
            "mean_confidence": (sum(max(probabilities[i]) for i in selected) / len(selected)
                                if selected else None),
            "accuracy": (sum(probabilities[i].index(max(probabilities[i])) == labels[i]
                             for i in selected) / len(selected) if selected else None),
        })
    selective = []
    for threshold in (0.5, 0.7, 0.9):
        selected = [i for i, p in enumerate(probabilities) if max(p) >= threshold]
        selective.append({
            "minimum_confidence": threshold,
            "n": len(selected),
            "coverage": len(selected) / len(labels),
            "accuracy": (sum(probabilities[i].index(max(probabilities[i])) == labels[i]
                             for i in selected) / len(selected) if selected else None),
        })
    return {"bins": bins, "selective_accuracy": selective}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-confusion", type=int, default=12)
    args = parser.parse_args()
    if args.cases_per_confusion < 1:
        raise ValueError("cases-per-confusion must be positive")

    import torch
    from safetensors.torch import load_file

    report = json.loads((args.run_dir / "report.json").read_text())
    if report.get("artifact_type") != "system_one_phase1a_action_family_result":
        raise ValueError("input is not a completed Phase-1a report")
    source = args.data_dir / f"{args.split}.jsonl"
    if file_sha256(source) != report["source_sha256"][args.split]:
        raise ValueError("source hash differs from the completed run")
    feature_path = args.run_dir / f"{args.split}_features.safetensors"
    if file_sha256(feature_path) != report["feature_sha256"][args.split]:
        raise ValueError("frozen features differ from the completed run")
    head_path = args.run_dir / "action_family_head.pt"
    if file_sha256(head_path) != report["head_sha256"]:
        raise ValueError("action head differs from the completed run")

    examples, counts = load_split(source)
    features = load_file(str(feature_path))["features"]
    if features.ndim != 2 or features.shape[0] != len(examples):
        raise ValueError("feature matrix does not align with decision rows")
    head = ActionFamilyHead(features.shape[1])
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    head.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(head(features.float()), -1).tolist()
    predictions = [max(range(len(ACTION_NAMES)), key=lambda j: row[j])
                   for row in probabilities]
    labels = [example.label for example in examples]
    summary = metrics(labels, predictions)
    if (summary["n"] != report[args.split]["n"]
            or summary["confusion_gold_rows_predicted_columns"] !=
            report[args.split]["confusion_gold_rows_predicted_columns"]):
        raise ValueError("post-hoc predictions do not reproduce the run report")

    by_history = defaultdict(lambda: {"n": 0, "correct": 0, "import_n": 0,
                                      "import_correct": 0, "premature_finish": 0})
    confusion_cases = defaultdict(list)
    for example, prediction, probability in zip(examples, predictions, probabilities,
                                                 strict=True):
        bucket = str(min(example.history_accepted_actions, 4))
        group = by_history[bucket]
        group["n"] += 1
        group["correct"] += prediction == example.label
        group["import_n"] += example.label == 1
        group["import_correct"] += example.label == prediction == 1
        group["premature_finish"] += example.label != 2 and prediction == 2
        if prediction != example.label:
            confusion_cases[(example.label, prediction)].append({
                "row_id": example.row_id,
                "reaction_id": example.reaction_id,
                "gold": ACTION_NAMES[example.label],
                "predicted": ACTION_NAMES[prediction],
                "confidence": probability[prediction],
                "probabilities": dict(zip(ACTION_NAMES, probability, strict=True)),
                "history_accepted_actions": example.history_accepted_actions,
                "atom_count": len(example.atom_names),
                "current_observation": example.messages[-1]["content"],
            })
    for value in by_history.values():
        value["accuracy"] = value["correct"] / value["n"]
        value["import_recall"] = (value["import_correct"] / value["import_n"]
                                  if value["import_n"] else None)
    cases = {f"{ACTION_NAMES[gold]}_to_{ACTION_NAMES[pred]}":
             sorted(rows, key=lambda row: row["confidence"], reverse=True)
             [:args.cases_per_confusion]
             for (gold, pred), rows in confusion_cases.items()}
    output = {
        "artifact_type": "system_one_phase1a_action_family_error_analysis",
        "split": args.split,
        "source_sha256": report["source_sha256"][args.split],
        "head_sha256": report["head_sha256"],
        "reaction_count": counts["reactions"],
        "summary": summary,
        "calibration": calibration(probabilities, labels),
        "by_accepted_action_count_capped_at_4": dict(sorted(by_history.items())),
        "high_confidence_errors": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "split": args.split,
                      "n": summary["n"], "accuracy": summary["accuracy"],
                      "macro_f1": summary["macro_f1"],
                      "import_recall": summary["classes"]["import_fragments"]["recall"],
                      "premature_finish": summary["premature_finish"],
                      "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
