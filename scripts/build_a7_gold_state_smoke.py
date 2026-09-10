#!/usr/bin/env python3
"""Freeze and validate the model-free portion of the rapid A7 smoke."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.a7_rescue import audit_gold_row, stratified_sample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-rows", type=int, default=2890)
    parser.add_argument(
        "--expected-observation-mode", default="compact_full_state_v1"
    )
    parser.add_argument("--minimum-gold-legal-event-rate", type=float, default=0.99)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    split = (manifest.get("splits") or {}).get("valid") or {}
    observed_sha = sha256(args.data)
    if int(split.get("rows") or -1) != args.expected_rows:
        raise ValueError("validation row count disagrees with frozen manifest")
    if str(split.get("sha256") or "") != observed_sha:
        raise ValueError("validation SHA-256 disagrees with frozen manifest")
    if manifest.get("observation_mode") != args.expected_observation_mode:
        raise ValueError("unexpected observation mode")
    rows = read_jsonl(args.data)
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, observed {len(rows)}")

    selected = stratified_sample(rows, size=args.size, seed=args.seed)
    audited = [audit_gold_row(row) for row in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "selection.jsonl"
    selection_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": item["id"],
                    "stratum": item["stratum"],
                    "n_events": item["n_events"],
                    "n_moves": item["n_moves"],
                },
                sort_keys=True,
            )
            + "\n"
            for item in audited
        )
    )
    selected_rows_path = args.output / "selected_rows.jsonl"
    selected_rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    )
    n_events = sum(item["n_events"] for item in audited)
    report = {
        "artifact_type": "a7_gold_state_model_free_gate_v1",
        "source": {
            "data": str(args.data),
            "sha256": observed_sha,
            "rows": len(rows),
            "manifest": str(args.manifest),
            "observation_mode": manifest["observation_mode"],
        },
        "selection": {
            "seed": args.seed,
            "size": len(audited),
            "sha256": sha256(selection_path),
            "selected_rows_sha256": sha256(selected_rows_path),
            "strata": dict(Counter(item["stratum"] for item in audited)),
        },
        "counts": {
            "reactions": len(audited),
            "events": n_events,
            "moves": sum(item["n_moves"] for item in audited),
            "be_delta_moves": sum(
                event["n_be_delta_moves"]
                for item in audited
                for event in item["events"]
            ),
            "supervision_aligned_reactions": sum(
                item["supervision_aligned"] for item in audited
            ),
            "gold_legal_reactions": sum(item["gold_legal"] for item in audited),
            "gold_legal_events": sum(
                event["gold_legal"] for item in audited for event in item["events"]
            ),
            "visible_inventory_covered_events": sum(
                event["visible_inventory_covered"]
                for item in audited
                for event in item["events"]
            ),
            "candidate_inventory_covered_events": sum(
                event["candidate_inventory_covered"]
                for item in audited
                for event in item["events"]
            ),
        },
        "means": {
            "events_per_reaction": n_events / len(audited),
            "moves_per_reaction": sum(item["n_moves"] for item in audited)
            / len(audited),
        },
        "gate_threshold": args.minimum_gold_legal_event_rate,
        "gate_pass": (
            sum(event["gold_legal"] for item in audited for event in item["events"])
            / n_events
            >= args.minimum_gold_legal_event_rate
        ),
        "rows": audited,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("selection", "counts", "means", "gate_pass")}), flush=True)
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
