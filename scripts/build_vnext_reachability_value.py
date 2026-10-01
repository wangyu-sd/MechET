#!/usr/bin/env python3
"""Mine bounded on-policy successor reachability labels without fake negatives.

An off-reference but executable successor is *not* automatically negative.
Terminal wrong endpoints are negative; repeated failed continuations are
labelled only as non-reachability under the declared search budget.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.successor_value import reachability_value_row
from mechet.vnext_tree_credit import chemical_state_key


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_rows(
    sources: list[dict[str, Any]], rollouts: list[dict[str, Any]],
    *, decision_budget: int, min_failed_rollouts: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    if decision_budget < 2 or min_failed_rollouts < 2:
        raise ValueError("invalid budget or minimum failed rollout count")
    source_by_id = {str(row["source_id"]): row for row in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("source file has duplicate reaction identifiers")
    groups: dict[tuple[str, str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    skipped_invalid = skipped_ambiguous = 0
    for record in rollouts:
        if record.get("kind") != "rl":
            continue
        rid = str(record["id"])
        if rid not in source_by_id:
            raise ValueError(f"rollout {rid} absent from training-only source")
        score = dict(record.get("score") or {})
        successor = str(score.get("first_successor_state") or "")
        trajectory = list(score.get("trajectory") or [])
        if not successor or not trajectory:
            skipped_invalid += 1
            continue
        current = str(trajectory[0].get("state_before") or "")
        anchor_hash = str((record.get("anchor") or {}).get("state_hash") or "")
        if not current or hashlib.sha256(current.encode()).hexdigest() != anchor_hash:
            raise ValueError(f"{rid}: first executed state does not match rollout anchor")
        terminal = bool(score.get("first_successor_terminal"))
        key = chemical_state_key(successor, terminal=terminal)
        groups[(rid, anchor_hash, key, terminal)].append(record)

    output: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    stats = defaultdict(int)
    for (rid, anchor_hash, _, terminal), records in sorted(groups.items()):
        scores = [dict(record["score"]) for record in records]
        successes = sum(bool(score.get("correct")) for score in scores)
        if successes:
            label, provenance = "P", "on_policy_exact_endpoint_reached"
        elif terminal:
            label, provenance = "N", "actor_terminal_wrong_endpoint"
        elif len(records) >= min_failed_rollouts and all(
            str(score.get("failure") or "") in {"DECISION_BUDGET", "NO_EXECUTABLE_CONTINUATION"}
            for score in scores
        ):
            label, provenance = "N", "repeated_bounded_policy_nonreachability"
        else:
            skipped_ambiguous += 1
            continue
        source = source_by_id[rid]
        trajectory = scores[0]["trajectory"]
        current = str(trajectory[0]["state_before"])
        successor = str(scores[0]["first_successor_state"])
        # Keep every reaction in exactly one critic split. The original
        # benchmark validation/test rows are never mined as critic training.
        digest = hashlib.sha256(f"17:{rid}".encode()).digest()
        split = "valid" if int.from_bytes(digest[:4], "big") % 10 == 0 else "train"
        item = reachability_value_row(
            reaction_id=rid, state_hash=anchor_hash,
            target=str(source["target_smiles"]), current_state=current,
            successor_state=successor, terminal=terminal,
            remaining_decisions=decision_budget - 1,
            label=label, provenance=provenance, observations=len(records),
        )
        output[split].append(item)
        stats[f"{split}_{label}"] += 1
        stats[provenance] += 1
    stats["valid_successors_seen"] = len(groups)
    stats["invalid_or_unexecuted_skipped"] = skipped_invalid
    stats["ambiguous_nonterminal_skipped"] = skipped_ambiguous
    if not output["train"] or not output["valid"]:
        raise ValueError("critic requires nonempty train and internal validation")
    if not any(item["metadata"]["label"] == "N" for item in output["train"]):
        raise ValueError("critic training has no verified/bounded negative examples")
    return output, dict(stats)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-source", type=Path, required=True)
    p.add_argument("--rollout", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--decision-budget", type=int, default=40)
    p.add_argument("--min-failed-rollouts", type=int, default=2)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    staging = args.output_dir.with_name(args.output_dir.name + ".building")
    if staging.exists():
        raise FileExistsError(staging)
    source = load_jsonl(args.train_source)
    rollouts = [r for path in args.rollout for r in load_jsonl(path)]
    output, stats = build_rows(
        source, rollouts, decision_budget=args.decision_budget,
        min_failed_rollouts=args.min_failed_rollouts,
    )
    staging.mkdir(parents=True)
    manifest = {
        "artifact_type": "bounded_on_policy_successor_reachability_v1",
        "source_sha256": hashlib.sha256(args.train_source.read_bytes()).hexdigest(),
        "rollout_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.rollout},
        "decision_budget": args.decision_budget,
        "min_failed_rollouts": args.min_failed_rollouts,
        "reference_endpoint_model_visible": False,
        "negative_means_chemically_impossible": False,
        "statistics": stats, "splits": {},
    }
    for split, rows in output.items():
        path = staging / f"{split}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "rows": len(rows),
            "reactions": len({r["source_id"] for r in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if {r["source_id"] for r in output["train"]} & {r["source_id"] for r in output["valid"]}:
        raise ValueError("reaction leakage across critic splits")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (staging / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "status": "validated_train_only_on_policy_value",
        "training_allowed": True,
        "negative_interpretation": "bounded-policy observation, not absolute chemistry",
    }, indent=2))
    staging.rename(args.output_dir)
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
