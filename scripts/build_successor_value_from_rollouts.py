#!/usr/bin/env python3
"""Mine successor-value positives and actor-induced hard negatives.

This consumes training rollouts only.  It reconstructs the private reference
first successor from the frozen trace, pairs it with distinct executable states
actually proposed by the actor, and writes ordinary assistant-only SFT rows.
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.natural_language_anchor_branch_rl import task_from_episode
from mechet.successor_value import VERSION, successor_value_row
from scripts.eval_natural_language_event_suffix import reference_episode
from scripts.run_natural_language_value_search import visible


def _read_rollouts(paths: list[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") != "rl":
                    continue
                key = (str(row["id"]), str(row["anchor"]["state_hash"]))
                groups[key].append(row)
    return groups


def _source_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifiers = {
                str(value)
                for value in (row.get("id"), row.get("source_id"))
                if value
            }
            for reaction_id in identifiers & wanted:
                found[reaction_id] = row
                if len(found) == len(wanted):
                    break
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"source reactions missing for {len(missing)} rollout groups")
    return found


def build_rows(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    sources: dict[str, dict[str, Any]],
    *,
    max_negatives: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    positives = negatives = alternate_positives = groups_without_negative = 0
    for (reaction_id, state_hash), records in sorted(groups.items()):
        horizon = int(records[0]["anchor"]["horizon"])
        episode = reference_episode(sources[reaction_id], horizon)
        task = task_from_episode(episode)
        if task.state_hash != state_hash:
            raise ValueError(f"{reaction_id}: rollout/source anchor hash mismatch")
        event = dict(episode["events"][0])
        reference_successor = str(
            event["event_state"] if event.get("imports") else event["reference_successor"]
        )
        output.append(
            successor_value_row(
                reaction_id=reaction_id,
                state_hash=state_hash,
                target=task.target,
                current_state=task.anchor_state,
                successor_state=reference_successor,
                terminal=False,
                label="P",
                provenance="executor_reference_successor",
            )
        )
        positives += 1

        seen = {visible(reference_successor)}
        candidates = []
        for record in records:
            score = dict(record.get("score") or {})
            successor = str(score.get("first_successor_state") or "")
            if not successor or score.get("reference_first_successor_exact"):
                continue
            public = visible(successor)
            if public in seen:
                continue
            seen.add(public)
            label = "P" if score.get("correct") else "N"
            candidates.append(
                (
                    float(record.get("anchor_action_q") or record.get("reward") or 0.0),
                    bool(score.get("first_successor_terminal")),
                    successor,
                    label,
                )
            )
        # The actor's most plausible mistakes are the useful hard negatives.
        candidates.sort(key=lambda value: value[0], reverse=True)
        alternative_successes = [value for value in candidates if value[3] == "P"]
        hard_negatives = [value for value in candidates if value[3] == "N"]
        if not hard_negatives:
            groups_without_negative += 1
        selected = alternative_successes + hard_negatives[: max(int(max_negatives), 0)]
        for _, terminal, successor, label in selected:
            output.append(
                successor_value_row(
                    reaction_id=reaction_id,
                    state_hash=state_hash,
                    target=task.target,
                    current_state=task.anchor_state,
                    successor_state=successor,
                    terminal=terminal,
                    label=label,
                    provenance=(
                        "actor_alternative_exact_endpoint_successor"
                        if label == "P"
                        else "actor_executable_off_reference_hard_negative"
                    ),
                )
            )
            if label == "P":
                positives += 1
                alternate_positives += 1
            else:
                negatives += 1
    return output, {
        "groups": len(groups),
        "positives": positives,
        "alternate_endpoint_positive_successors": alternate_positives,
        "hard_negatives": negatives,
        "groups_without_hard_negative": groups_without_negative,
        "rows": len(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-negatives", type=int, default=4)
    args = parser.parse_args()
    groups = _read_rollouts(args.rollout)
    if not groups:
        raise ValueError("no RL rollout groups")
    sources = _source_rows(args.source, {key[0] for key in groups})
    rows, report = build_rows(groups, sources, max_negatives=args.max_negatives)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    report.update(
        {
            "artifact_type": VERSION,
            "source_rollout_sha256": hashlib.sha256(
                "\n".join(str(path) for path in args.rollout).encode("utf-8")
            ).hexdigest(),
            "reference_endpoint_model_visible": False,
        }
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
