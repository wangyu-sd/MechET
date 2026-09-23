#!/usr/bin/env python3
"""Build train-only successor critic labels from EARHO v2 actor rollouts.

The reference endpoint and successor are private labels.  Critic prompts
contain only the public product, current state, candidate successor and a
terminal flag.  Distinct executed successors, not action strings, are units.
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

from mechet.successor_value import successor_value_row, visible_successor_state
from scripts.earho_v2_protocol import replay_reference


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_rows(
    sources: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    *,
    max_hard_negatives: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    by_source = {str(row["source_id"]): row for row in sources}
    if len(by_source) != len(sources):
        raise ValueError("duplicate source reaction ID")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        if row.get("kind") != "rl":
            continue
        anchor = dict(row.get("anchor") or {})
        if anchor.get("version") != "earho_first_divergence_v2":
            raise ValueError("rollout is not from EARHO v2")
        if anchor.get("divergence_reason") in {"EXACT_ENDPOINT", "REFERENCE_ALIGNED"}:
            continue
        groups[(str(row["id"]), str(anchor["state_hash"]))].append(row)
    if not groups:
        raise ValueError("no EARHO v2 actor rollout groups")
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    statistics = defaultdict(int)
    references = {}
    for (reaction_id, state_hash), candidates in sorted(groups.items()):
        source = by_source.get(reaction_id)
        if source is None:
            raise ValueError(f"missing training source: {reaction_id}")
        if reaction_id not in references:
            references[reaction_id] = replay_reference(
                source, source["earho_v2_reference_decisions"]
            )
        reference = references[reaction_id]
        indices = {int(item["anchor"]["decision_index"]) for item in candidates}
        if len(indices) != 1:
            raise ValueError(f"mixed anchor indices: {reaction_id}")
        index = indices.pop()
        if not 0 <= index < len(reference.decisions):
            raise ValueError(f"invalid anchor index: {reaction_id}:{index}")
        current = reference.nodes[index]
        if hashlib.sha256(current.state.encode()).hexdigest() != state_hash:
            raise ValueError(f"anchor state hash mismatch: {reaction_id}")
        expected = reference.nodes[index + 1]
        split = (
            "valid"
            if int.from_bytes(hashlib.sha256(f"17:{reaction_id}".encode()).digest()[:4], "big") % 10 == 0
            else "train"
        )
        outputs[split].append(
            successor_value_row(
                reaction_id=reaction_id,
                state_hash=state_hash,
                target=reference.target,
                current_state=current.state,
                successor_state=expected.state,
                terminal=expected.terminal,
                label="P",
                provenance="verified_reference_successor",
            )
        )
        statistics["reference_positives"] += 1
        seen = {visible_successor_state(expected.state)}
        alternatives = []
        for candidate in candidates:
            score = dict(candidate.get("score") or {})
            successor = str(score.get("first_successor_state") or "")
            if not successor:
                continue
            public = visible_successor_state(successor)
            if public in seen:
                continue
            seen.add(public)
            productive = bool(score.get("correct"))
            alternatives.append((
                float(candidate.get("anchor_action_q") or candidate.get("reward") or 0.0),
                successor,
                bool(score.get("first_successor_terminal")),
                "P" if productive else "N",
            ))
        alternatives.sort(key=lambda item: item[0], reverse=True)
        positives = [item for item in alternatives if item[3] == "P"]
        negatives = [item for item in alternatives if item[3] == "N"][:max_hard_negatives]
        for _, successor, terminal, label in positives + negatives:
            outputs[split].append(
                successor_value_row(
                    reaction_id=reaction_id,
                    state_hash=state_hash,
                    target=reference.target,
                    current_state=current.state,
                    successor_state=successor,
                    terminal=terminal,
                    label=label,
                    provenance=(
                        "actor_exact_endpoint_alternative"
                        if label == "P"
                        else "actor_executable_unverified_hard_negative"
                    ),
                )
            )
            statistics["alternative_positives" if label == "P" else "hard_negatives"] += 1
        statistics["groups"] += 1
    if not outputs["train"] or not outputs["valid"]:
        raise ValueError("successor critic requires nonempty disjoint train and valid")
    train_ids = {item["source_id"] for item in outputs["train"]}
    valid_ids = {item["source_id"] for item in outputs["valid"]}
    if train_ids & valid_ids:
        raise ValueError("critic train/valid reaction overlap")
    return outputs, dict(statistics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hard-negatives", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    building = args.output_dir.with_name(args.output_dir.name + ".building")
    if building.exists():
        raise FileExistsError(f"incomplete earlier critic build: {building}")
    sources = read_jsonl(args.source)
    rollouts = [row for path in args.rollout for row in read_jsonl(path)]
    rows, stats = build_rows(
        sources, rollouts, max_hard_negatives=args.max_hard_negatives
    )
    building.mkdir(parents=True)
    manifest = {
        "artifact_type": "earho_v2_successor_value_supervision",
        "source_file": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "reference_endpoint_model_visible": False,
        "split_reaction_overlap": 0,
        "statistics": stats,
        "splits": {},
    }
    for split, items in rows.items():
        path = building / f"{split}.jsonl"
        with path.open("w") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["splits"][split] = {
            "rows": len(items),
            "reactions": len({item["source_id"] for item in items}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (building / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (building / "ARTIFACT_STATUS.json").write_text(
        json.dumps({"training_allowed": True, "reason": "verified train-only EARHO v2 successors"}, indent=2) + "\n"
    )
    building.rename(args.output_dir)
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
