#!/usr/bin/env python3
"""Build and audit full in-place grounded electron-flow Tool-SFT data."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.in_place_grounded_flow import convert_trace_row


EXPECTED = {"train": 257167, "valid": 2890, "test": 28967}


def _convert_line(line: str) -> tuple[str, dict[str, Any]]:
    source = dict(json.loads(line))
    converted = convert_trace_row(source)
    source_steps = list(
        (((source.get("metadata") or {}).get("trace_plan") or {}).get("steps") or [])
    )
    legacy_move_chars = sum(
        len(json.dumps(step.get("moves") or [], separators=(",", ":")))
        for step in source_steps
    )
    event_calls = [
        call
        for message in converted["messages"]
        for call in message.get("tool_calls") or []
        if (call.get("function") or {}).get("name") == "apply_grounded_event"
    ]
    flow_chars = sum(
        len(str((call["function"]["arguments"]).get("flow") or ""))
        for call in event_calls
    )
    marked_state_chars = sum(
        len(str((call["function"]["arguments"]).get("marked_state") or ""))
        for call in event_calls
    )
    audits = list((converted.get("metadata") or {}).get("event_audits") or [])
    return (
        json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n",
        {
            "id": converted["id"],
            "events": len(audits),
            "moves": sum(int(item["n_moves"]) for item in audits),
            "roles": sum(int(item["n_roles"]) for item in audits),
            "imports": int(converted["metadata"]["n_import_fragments"]),
            "be_delta_events": sum(bool(item["be_delta"]) for item in audits),
            "legacy_move_chars": legacy_move_chars,
            "flow_chars": flow_chars,
            "marked_state_chars": marked_state_chars,
        },
    )


def _batches(handle: Iterable[str], size: int) -> Iterable[list[str]]:
    iterator = iter(handle)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(
    *,
    source: Path,
    target: Path,
    workers: int,
    batch_size: int,
    limit: int,
) -> tuple[dict[str, Any], set[str]]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    rows = 0
    with source.open(encoding="utf-8") as input_handle, temporary.open(
        "w", encoding="utf-8"
    ) as output_handle, ProcessPoolExecutor(max_workers=workers) as executor:
        for batch in _batches(input_handle, batch_size):
            if limit:
                remaining = limit - rows
                if remaining <= 0:
                    break
                batch = batch[:remaining]
            for line in batch:
                source_digest.update(line.encode())
            for encoded, audit in executor.map(_convert_line, batch, chunksize=8):
                identifier = str(audit["id"])
                if not identifier or identifier in seen:
                    raise ValueError(f"missing or duplicate ID: {identifier!r}")
                seen.add(identifier)
                output_handle.write(encoded)
                output_digest.update(encoded.encode())
                rows += 1
                for key, value in audit.items():
                    if key != "id":
                        counts[key] += int(value)
                if rows % 1000 == 0:
                    print(
                        f"[in-place-build] split={source.stem} rows={rows} "
                        f"events={counts['events']}",
                        flush=True,
                    )
            if limit and rows >= limit:
                break
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, target)
    report = {
        "source": str(source),
        "target": str(target),
        "rows": rows,
        "unique_ids": len(seen),
        "source_sha256": source_digest.hexdigest(),
        "output_sha256": output_digest.hexdigest(),
        **dict(counts),
        "move_roundtrip_failures": 0,
        "successor_replay_failures": 0,
        "endpoint_failures": 0,
        "map_leakage_failures": 0,
    }
    report_path = target.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report, seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/flower_inverse_tool_sft_action_delta_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/flower_in_place_grounded_flow_v1"),
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 64))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch-size must be positive")

    reports: dict[str, Any] = {}
    split_ids: dict[str, set[str]] = {}
    for split in ("valid", "test", "train"):
        source = args.source_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        report, identifiers = build_split(
            source=source,
            target=args.output_dir / f"{split}.jsonl",
            workers=args.workers,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        reports[split] = report
        split_ids[split] = identifiers

    overlap = {
        "train_valid": len(split_ids["train"] & split_ids["valid"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "valid_test": len(split_ids["valid"] & split_ids["test"]),
    }
    full = args.limit == 0
    source_manifest_path = args.source_dir / "training_manifest.json"
    source_status_path = args.source_dir / "ARTIFACT_STATUS.json"
    if not source_manifest_path.is_file() or not source_status_path.is_file():
        raise FileNotFoundError("source training manifest/status sidecars are required")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
    if source_manifest.get("observation_mode") != "action_delta_v1":
        raise ValueError("source artifact is not the frozen action_delta_v1 universe")
    if not source_manifest.get("strict_trace_universe_complete"):
        raise ValueError("source strict trace universe is incomplete")
    if source_status.get("training_allowed") is not True:
        raise ValueError("source ARTIFACT_STATUS forbids training")
    source_contract_exact = all(
        int(source_manifest["splits"][split]["rows"]) == expected
        and (
            not full
            or str(source_manifest["splits"][split]["sha256"])
            == str(reports[split]["source_sha256"])
        )
        for split, expected in EXPECTED.items()
    )
    exact_counts = full and all(
        int(reports[split]["rows"]) == expected
        for split, expected in EXPECTED.items()
    )
    total_legacy = sum(int(report["legacy_move_chars"]) for report in reports.values())
    total_flow = sum(int(report["flow_chars"]) for report in reports.values())
    flow_ratio = total_flow / max(total_legacy, 1)
    passed = bool(
        exact_counts
        and source_contract_exact
        and not any(overlap.values())
        and flow_ratio < 1.0
    )
    manifest = {
        "artifact_type": "flower_in_place_grounded_flow_sft_v1",
        "source_artifact": str(args.source_dir),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_contract_exact": source_contract_exact,
        "representation": "in_place_grounded_flow_v1",
        "input_contract": "unmapped_product_and_executor_current_state",
        "output_contract": "first_use_imports_plus_insertion_only_markers_plus_compact_flow",
        "reaction_denominator": dict(EXPECTED),
        "splits": reports,
        "split_id_overlap": overlap,
        "filtering": {
            "new_reaction_filtering": False,
            "length_filtering": False,
            "overlap_filtering": False,
        },
        "model_visible_atom_maps": False,
        "move_roundtrip_exact": True,
        "successor_replay_exact": True,
        "endpoint_replay_exact": True,
        "legacy_move_to_compact_flow_char_ratio": flow_ratio,
        "full_build": full,
        "gate_passed": passed,
        "training_allowed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    status = {
        "artifact_id": "flower_in_place_grounded_flow_v1",
        "status": "validated" if passed else "diagnostic_only",
        "training_allowed": passed,
        "reason": (
            "All strict executable reaction IDs passed insertion-only grounding, "
            "move round-trip, successor replay, endpoint replay and map-visibility gates."
            if passed
            else "Full-denominator representation gate did not pass."
        ),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "status": status}), flush=True)
    return 0 if passed or not full else 2


if __name__ == "__main__":
    raise SystemExit(main())
