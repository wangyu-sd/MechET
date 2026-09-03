#!/usr/bin/env python3
"""Build a small A7-continuation corpus that emphasizes late trace decisions.

The builder never invents chemistry.  It converts replay-verified A7 expert
conversations into two kinds of rows:

* anchor rows retain the complete original expert conversation;
* state-reset rows replace an accepted prefix by the target and the latest
  authoritative executor observation, then retain the exact expert suffix.

State-reset rows train the policy to condition on the current chemical state
instead of relying on a pristine, fully visible action history.  They are not
off-policy recovery examples and the manifest labels that distinction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "a8_lite_state_reset_continuation_sft_v1"


@dataclass(frozen=True)
class Candidate:
    line_index: int
    row_id: str
    score: int
    tool_calls: int


def _stable_score(seed: int, namespace: str, row_id: str) -> int:
    payload = f"{seed}\0{namespace}\0{row_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _offer_smallest(heap: list[tuple[int, int, Candidate]], item: Candidate, limit: int) -> None:
    """Keep the ``limit`` candidates with the smallest deterministic scores."""

    entry = (-item.score, -item.line_index, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    if entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _assistant_tool_calls(messages: Iterable[dict[str, Any]]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in messages
        if message.get("role") == "assistant"
    )


def _accepted_state_points(messages: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    points: list[tuple[int, dict[str, Any]]] = []
    for index, message in enumerate(messages[:-1]):
        if message.get("role") != "tool" or message.get("name") == "finish_trace":
            continue
        try:
            observation = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        if observation.get("ok") is not True:
            continue
        state = str(observation.get("current_state_smiles") or "").strip()
        if not state:
            continue
        if messages[index + 1].get("role") != "assistant":
            continue
        points.append((index, observation))
    return points


def _choose_reset_point(messages: list[dict[str, Any]]) -> tuple[int, dict[str, Any], int]:
    points = _accepted_state_points(messages)
    if not points:
        raise ValueError("conversation has no accepted authoritative state checkpoint")

    # Prefer a checkpoint near two thirds of the trace.  The suffix remains long
    # enough to supervise recovery of the terminal program while emphasizing the
    # decisions most exposed to accumulated history errors.
    target = (2 * (len(points) - 1)) / 3
    position = min(range(len(points)), key=lambda value: (abs(value - target), value))
    message_index, observation = points[position]
    return message_index, observation, position + 1


def _target(row: dict[str, Any]) -> str:
    target = str(row.get("target_smiles") or "").strip()
    if target:
        return target
    metadata = dict(row.get("metadata") or {})
    plan = dict(metadata.get("trace_plan") or {})
    target = str(plan.get("target_smiles") or "").strip()
    if not target:
        raise ValueError(f"row has no target_smiles: {row.get('id')}")
    return target


def make_state_reset_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = [copy.deepcopy(message) for message in row.get("messages") or []]
    if len(messages) < 4 or messages[0].get("role") != "system":
        raise ValueError(f"unexpected conversation prefix: {row.get('id')}")
    reset_index, observation, committed_calls = _choose_reset_point(messages)
    suffix = messages[reset_index + 1 :]
    if not suffix or suffix[0].get("role") != "assistant":
        raise ValueError(f"invalid state-reset suffix: {row.get('id')}")

    remaining_calls = _assistant_tool_calls(suffix)
    if remaining_calls < 1:
        raise ValueError(f"state-reset suffix has no remaining calls: {row.get('id')}")
    if sum(
        call.get("function", {}).get("name") == "finish_trace"
        for message in suffix
        for call in (message.get("tool_calls") or [])
    ) != 1:
        raise ValueError(f"state-reset suffix must contain finish_trace: {row.get('id')}")

    restored = json.dumps(observation, ensure_ascii=False, sort_keys=True)
    user_message = {
        "role": "user",
        "content": (
            f"TARGET: {_target(row)}\n"
            "Continue the same executor-owned inverse trace from the restored "
            "authoritative state below. Earlier accepted calls are already "
            "committed: do not replay them. Use only the current state and the "
            "target to choose the remaining executable actions, then call "
            "finish_trace.\n\nRESTORED EXECUTOR OBSERVATION:\n"
            f"{restored}"
        ),
    }

    result = copy.deepcopy(row)
    source_id = str(row.get("id") or "")
    result["id"] = f"a8-lite-reset:{source_id}:after-{committed_calls}"
    result["source_id"] = source_id
    result["artifact_type"] = ARTIFACT_TYPE
    result["messages"] = [messages[0], user_message, *suffix]
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "a8_lite_role": "state_reset",
            "a8_lite_source_id": source_id,
            "a8_lite_reset_after_committed_calls": committed_calls,
            "a8_lite_remaining_tool_calls": remaining_calls,
            "a8_lite_off_policy_recovery": False,
            "a8_lite_history_policy": "target_plus_latest_authoritative_state_v1",
        }
    )
    result["metadata"] = metadata
    return result


def make_anchor_row(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    source_id = str(row.get("id") or "")
    result["id"] = f"a8-lite-anchor:{source_id}"
    result["source_id"] = source_id
    result["artifact_type"] = ARTIFACT_TYPE
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "a8_lite_role": "expert_anchor",
            "a8_lite_source_id": source_id,
            "a8_lite_off_policy_recovery": False,
        }
    )
    result["metadata"] = metadata
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_candidates(
    source: Path,
    *,
    restart_rows: int,
    anchor_rows: int,
    min_tool_calls: int,
    seed: int,
) -> tuple[list[Candidate], list[Candidate], int]:
    restart_heap: list[tuple[int, int, Candidate]] = []
    # Keep enough anchor candidates to remove any overlap with reset sources.
    anchor_heap: list[tuple[int, int, Candidate]] = []
    anchor_pool = anchor_rows + restart_rows
    total = 0
    with source.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = dict(json.loads(line))
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"source row {line_index} has no ID")
            tool_calls = _assistant_tool_calls(row.get("messages") or [])
            anchor = Candidate(
                line_index=line_index,
                row_id=row_id,
                score=_stable_score(seed, "anchor", row_id),
                tool_calls=tool_calls,
            )
            _offer_smallest(anchor_heap, anchor, anchor_pool)
            if tool_calls >= min_tool_calls and _accepted_state_points(
                list(row.get("messages") or [])
            ):
                restart = Candidate(
                    line_index=line_index,
                    row_id=row_id,
                    score=_stable_score(seed, "state-reset", row_id),
                    tool_calls=tool_calls,
                )
                _offer_smallest(restart_heap, restart, restart_rows)
            total += 1
            if total % 20_000 == 0:
                print(
                    f"[a8-lite][scan] rows={total} restart_pool={len(restart_heap)} "
                    f"anchor_pool={len(anchor_heap)}",
                    flush=True,
                )
    restarts = sorted((entry[2] for entry in restart_heap), key=lambda item: item.score)
    reset_ids = {item.row_id for item in restarts}
    anchors = sorted(
        (entry[2] for entry in anchor_heap if entry[2].row_id not in reset_ids),
        key=lambda item: item.score,
    )[:anchor_rows]
    if len(restarts) != restart_rows:
        raise ValueError(f"eligible state-reset rows {len(restarts)} != requested {restart_rows}")
    if len(anchors) != anchor_rows:
        raise ValueError(f"disjoint anchor rows {len(anchors)} != requested {anchor_rows}")
    return restarts, anchors, total


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_train.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "train.jsonl"

    restarts, anchors, source_rows = _scan_candidates(
        source,
        restart_rows=args.restart_rows,
        anchor_rows=args.anchor_rows,
        min_tool_calls=args.min_tool_calls,
        seed=args.seed,
    )
    restart_by_line = {item.line_index: item for item in restarts}
    anchor_by_line = {item.line_index: item for item in anchors}
    written_restart = written_anchor = 0
    reset_committed: list[int] = []
    reset_remaining: list[int] = []
    output_digest = hashlib.sha256()
    with source.open(encoding="utf-8") as source_handle, output.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_index, line in enumerate(source_handle):
            if line_index not in restart_by_line and line_index not in anchor_by_line:
                continue
            row = dict(json.loads(line))
            rows: list[dict[str, Any]] = []
            if line_index in anchor_by_line:
                rows.append(make_anchor_row(row))
                written_anchor += 1
            if line_index in restart_by_line:
                reset = make_state_reset_row(row)
                meta = dict(reset["metadata"])
                reset_committed.append(int(meta["a8_lite_reset_after_committed_calls"]))
                reset_remaining.append(int(meta["a8_lite_remaining_tool_calls"]))
                rows.append(reset)
                written_restart += 1
            for converted in rows:
                encoded = (
                    json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                output_handle.write(encoded.decode("utf-8"))
                output_digest.update(encoded)
            total_written = written_anchor + written_restart
            if total_written and total_written % 10_000 == 0:
                print(f"[a8-lite][write] rows={total_written}", flush=True)

    expected = args.restart_rows + args.anchor_rows
    if written_restart != args.restart_rows or written_anchor != args.anchor_rows:
        raise ValueError(
            "written row mismatch: "
            f"restart={written_restart}/{args.restart_rows} "
            f"anchor={written_anchor}/{args.anchor_rows}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "paper_condition": "A8-Lite",
        "scientific_hypothesis": "state_reset_continuation_reduces_long_horizon_history_drift",
        "source_train": str(source),
        "source_train_rows": source_rows,
        "source_train_sha256": _sha256(source),
        "selection": {
            "seed": args.seed,
            "restart_rows": args.restart_rows,
            "anchor_rows": args.anchor_rows,
            "min_tool_calls": args.min_tool_calls,
            "source_disjoint_between_roles": True,
        },
        "semantics": {
            "authoritative_state_visible": True,
            "expert_suffix_messages_preserved": True,
            "model_generated_states": False,
            "off_policy_recovery": False,
            "claim_boundary": "history-robust continuation, not recovery from arbitrary erroneous chemistry",
        },
        "splits": {
            "train": {
                "file": str(output),
                "rows": expected,
                "unique_ids": expected,
                "sha256": output_digest.hexdigest(),
                "restart_rows": written_restart,
                "anchor_rows": written_anchor,
                "mean_reset_after_committed_calls": sum(reset_committed)
                / max(len(reset_committed), 1),
                "mean_remaining_tool_calls": sum(reset_remaining)
                / max(len(reset_remaining), 1),
            },
            "valid": {
                "file": str(args.validation_file.resolve()),
                "rows": args.expected_validation_rows,
                "sha256": _sha256(args.validation_file.resolve()),
            },
            "test": {
                "file": str(args.test_file.resolve()),
                "rows": args.expected_test_rows,
                "sha256": _sha256(args.test_file.resolve()),
            },
        },
        "observation_mode": "compact_full_state_v1",
        "intermediate_state_model_visible": True,
        "strict_trace_universe_complete": False,
        "targeted_continuation_subset": True,
        "heldout_rows_used_for_training": False,
        "official_reaction_denominators": {
            "train": 257171,
            "valid": 2890,
            "test": 28971,
        },
        "strict_test_denominator": 28967,
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    status = {
        "artifact_id": "flower_a8_lite_state_reset_v1",
        "status": "validated_builder_output",
        "training_allowed": True,
        "view": "targeted_A7_continuation_state_reset",
        "rows": {
            "train": expected,
            "valid": args.expected_validation_rows,
            "test": args.expected_test_rows,
        },
        "full_test_required": True,
        "off_policy_recovery": False,
    }
    (output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-train",
        type=Path,
        default=Path("data/flower_inverse_tool_sft_compact_full_state_v1/train.jsonl"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/flower_inverse_tool_sft_compact_full_state_v1/valid.jsonl"),
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/flower_inverse_tool_sft_compact_full_state_v1/test.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/flower_a8_lite_state_reset_v1")
    )
    parser.add_argument("--restart-rows", type=int, default=30_000)
    parser.add_argument("--anchor-rows", type=int, default=10_000)
    parser.add_argument("--min-tool-calls", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-validation-rows", type=int, default=2_890)
    parser.add_argument("--expected-test-rows", type=int, default=28_967)
    args = parser.parse_args()
    if args.restart_rows <= 0 or args.anchor_rows <= 0:
        raise ValueError("restart-rows and anchor-rows must be positive")
    build(args)


if __name__ == "__main__":
    main()
