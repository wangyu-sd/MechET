#!/usr/bin/env python3
"""Build the frozen 512-train / 128-monitor endpoint-process RLVR pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.endpoint_process_rl_env import EndpointProcessRLEnv
from mechet.chemical_runtime import require_endpoint_process_rdkit
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    convert_trace_row,
    deterministic_unmapped_state,
    encode_grounded_event,
    extract_import_fragments,
    mapped_atom_numbers,
    mapped_state_signature,
    retain_mapped_components,
    schedule_imports,
)


ID_RE = re.compile(rb'"id"\s*:\s*"([^"\\]+)"')
STEP_RE = re.compile(rb'"n_trace_steps"\s*:\s*(\d+)')
EXPECTED_COUNTS = {"train": 257_167, "valid": 2_890}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["id"]) for row in rows) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def event_stratum(n_events: int) -> str:
    if n_events <= 2:
        return "short_1_2"
    if n_events <= 5:
        return "medium_3_5"
    return "long_6_plus"


def _quotas(count: int) -> dict[str, int]:
    names = ("short_1_2", "medium_3_5", "long_6_plus")
    base, remainder = divmod(count, len(names))
    return {
        name: base + int(index < remainder) for index, name in enumerate(names)
    }


def _selection_key(seed: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{identifier}".encode()).digest()
    return int.from_bytes(digest, "big")


def select_ids_streaming(
    path: Path,
    *,
    count: int,
    seed: int,
    expected_source_rows: int | None = None,
) -> list[tuple[str, int]]:
    """Select equal length strata without deserializing a multi-GB source."""

    quotas = _quotas(count)
    heaps: dict[str, list[tuple[int, str, int]]] = {name: [] for name in quotas}
    source_rows = 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            source_rows += 1
            id_match = ID_RE.search(raw)
            step_match = STEP_RE.search(raw)
            if id_match is None or step_match is None:
                raise ValueError(f"source row {source_rows} lacks id/n_trace_steps")
            identifier = id_match.group(1).decode()
            n_events = int(step_match.group(1))
            stratum = event_stratum(n_events)
            key = _selection_key(seed, identifier)
            candidate = (-key, identifier, n_events)
            heap = heaps[stratum]
            if len(heap) < quotas[stratum]:
                heapq.heappush(heap, candidate)
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
            if source_rows % 25_000 == 0:
                print(
                    f"[endpoint-pilot] scan path={path.name} rows={source_rows}",
                    flush=True,
                )
    if expected_source_rows is not None and source_rows != expected_source_rows:
        raise ValueError(
            f"source row count mismatch for {path}: {source_rows} != {expected_source_rows}"
        )
    missing = {
        name: quotas[name] - len(values)
        for name, values in heaps.items()
        if len(values) != quotas[name]
    }
    if missing:
        raise ValueError(f"insufficient rows for length strata: {missing}")
    selected = [
        (identifier, n_events)
        for values in heaps.values()
        for _, identifier, n_events in values
    ]
    return sorted(selected, key=lambda item: (_selection_key(seed, item[0]), item[0]))


def load_selected_rows(path: Path, identifiers: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(identifiers)
    found: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for row_index, raw in enumerate(handle, start=1):
            id_match = ID_RE.search(raw)
            if id_match is None:
                continue
            identifier = id_match.group(1).decode()
            if identifier in wanted:
                found[identifier] = json.loads(raw)
                if len(found) == len(wanted):
                    break
            if row_index % 25_000 == 0:
                print(
                    f"[endpoint-pilot] materialize path={path.name} "
                    f"rows={row_index} selected={len(found)}/{len(wanted)}",
                    flush=True,
                )
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"selected IDs missing from source: {sorted(missing)[:8]}")
    return [found[identifier] for identifier in sorted(wanted)]


def _maps_in_moves(moves: Sequence[Mapping[str, Any]]) -> set[int]:
    output: set[int] = set()
    for move in moves:
        if move.get("mode") == "BE_DELTA":
            for item in move.get("bond_deltas") or []:
                output.update(int(value) for value in item.get("atoms") or [])
            for item in move.get("charge_actions") or []:
                output.add(int(item["atom_map"]))
        else:
            output.update(int(value) for value in (move.get("source") or {}).get("atoms") or [])
            output.update(int(value) for value in (move.get("sink") or {}).get("atoms") or [])
    return output


def build_pilot_record(source: Mapping[str, Any]) -> dict[str, Any]:
    """Create one public/private pilot row and replay every gold transition."""

    converted = convert_trace_row(source)
    metadata = dict(source.get("metadata") or {})
    plan = dict(metadata.get("trace_plan") or {})
    steps = [dict(value) for value in plan.get("steps") or []]
    expected = str(source.get("full_precursor_state") or source.get("expected_precursor") or "")
    fragments = extract_import_fragments(source)
    scheduled = schedule_imports(fragments, steps)
    target_maps = mapped_atom_numbers(str(source.get("target_smiles") or ""))
    target = retain_mapped_components(
        str(steps[0].get("state_before") or source.get("target_smiles") or ""),
        target_maps,
    )
    if mapped_atom_numbers(target) != target_maps:
        raise ValueError(f"{source.get('id')}: product prefix exposes future maps")
    contributing_maps: set[int] = set()
    prefix_states = [target]
    transitions: list[dict[str, Any]] = []
    gold_actions: list[dict[str, Any]] = []
    current = target
    present_maps = set(target_maps)
    for index, (step, event_imports) in enumerate(zip(steps, scheduled)):
        before = current
        authoritative_prefix = retain_mapped_components(
            str(step.get("state_before") or ""), present_maps
        )
        if mapped_state_signature(before) != mapped_state_signature(
            authoritative_prefix
        ):
            raise ValueError(
                f"{source.get('id')}: private prefix mismatch at event {index}"
            )
        event_state = append_mapped_fragments_verbatim(before, event_imports)
        for fragment in event_imports:
            present_maps.update(mapped_atom_numbers(fragment))
        if mapped_atom_numbers(event_state) != present_maps:
            raise ValueError(
                f"{source.get('id')}: event {index} exposes unscheduled maps"
            )
        moves = [dict(value) for value in step.get("moves") or []]
        grounded = encode_grounded_event(event_state, moves)
        replay = verify_electron_step(event_state, list(grounded.compiled_moves))
        if not replay.get("ok"):
            raise ValueError(f"{source.get('id')}: private replay failed at event {index}")
        current = retain_mapped_components(
            str(step.get("state_after") or replay["state_smiles"]), present_maps
        )
        if mapped_state_signature(str(replay["state_smiles"])) != mapped_state_signature(
            current
        ):
            raise ValueError(
                f"{source.get('id')}: private successor mismatch at event {index}"
            )
        if mapped_atom_numbers(current) != present_maps:
            raise ValueError(
                f"{source.get('id')}: successor {index} exposes unscheduled maps"
            )
        transitions.append(
            {
                "step_index": index,
                "state_before": before,
                "state_after": current,
                "moves": list(grounded.compiled_moves),
                "imports": list(event_imports),
            }
        )
        gold_actions.append(
            {
                "name": "apply_grounded_event",
                "arguments": {
                    "imports": [
                        deterministic_unmapped_state(fragment).text
                        for fragment in event_imports
                    ],
                    "marked_state": grounded.marked_state,
                    "flow": grounded.flow,
                },
            }
        )
        contributing_maps.update(_maps_in_moves(moves) - target_maps)
        prefix_states.append(current)
    if mapped_state_signature(current) != mapped_state_signature(expected):
        raise ValueError(f"{source.get('id')}: pilot replay endpoint mismatch")
    row = {
        "id": str(source.get("id") or ""),
        "artifact_type": "endpoint_process_rlvr_pilot_v1",
        "public": {
            "system_prompt": str(converted["messages"][0]["content"]),
            "product": str(converted["target_smiles"]),
            "tools": converted["tools"],
        },
        "private_reward": {
            "target_mapped_state": target,
            "expected_precursor_mapped": expected,
            "target_atom_maps": sorted(target_maps),
            "contributing_atom_maps": sorted(contributing_maps),
            "gold_import_fragments": list(fragments),
            "prefix_states": prefix_states,
            "transitions": transitions,
            "gold_actions": gold_actions,
        },
        "metadata": {
            "n_events": len(steps),
            "event_length_stratum": event_stratum(len(steps)),
            "source_trace_digest": metadata.get("trace_digest"),
            "source_move_sequence_digest": metadata.get("move_sequence_digest"),
            "reward_contract": "endpoint_grounded_process_rlvr_v1",
            "gold_model_visible": False,
        },
    }
    EndpointProcessRLEnv(row).assert_no_reward_leakage()
    return row


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _build_split(
    source_path: Path,
    *,
    count: int,
    seed: int,
    expected_source_rows: int | None,
) -> list[dict[str, Any]]:
    selected = select_ids_streaming(
        source_path,
        count=count,
        seed=seed,
        expected_source_rows=expected_source_rows,
    )
    rows = load_selected_rows(source_path, (identifier for identifier, _ in selected))
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        identifier = str(row["id"])
        try:
            by_id[identifier] = build_pilot_record(row)
        except Exception as exc:
            raise ValueError(f"{identifier}: pilot record build failed") from exc
        if index % 64 == 0 or index == len(rows):
            print(
                f"[endpoint-pilot] replay path={source_path.name} "
                f"selected={index}/{len(rows)}",
                flush=True,
            )
    return [by_id[identifier] for identifier, _ in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-source",
        type=Path,
        default=REPO / "data/flower_inverse_tool_sft_action_delta_v1/train.jsonl",
    )
    parser.add_argument(
        "--monitor-source",
        type=Path,
        default=REPO / "data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl",
    )
    parser.add_argument(
        "--parent-adapter",
        type=Path,
        default=REPO
        / "outputs/agent/in_place_grounded_flow_qwen3_8b_a100_seed17_20260911/checkpoint-8037",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/endpoint_process_rlvr_pilot_v1",
    )
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--monitor-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-source-count-check", action="store_true")
    args = parser.parse_args()
    runtime_version = require_endpoint_process_rdkit()
    print(f"[endpoint-pilot] rdkit={runtime_version}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = _build_split(
        args.train_source,
        count=args.train_count,
        seed=args.seed,
        expected_source_rows=None
        if args.skip_source_count_check
        else EXPECTED_COUNTS["train"],
    )
    monitor = _build_split(
        args.monitor_source,
        count=args.monitor_count,
        seed=args.seed,
        expected_source_rows=None
        if args.skip_source_count_check
        else EXPECTED_COUNTS["valid"],
    )
    overlap = {row["id"] for row in train} & {row["id"] for row in monitor}
    if overlap:
        raise ValueError(f"train/monitor ID overlap: {sorted(overlap)[:8]}")
    train_path = args.output_dir / "train.jsonl"
    monitor_path = args.output_dir / "monitor.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(monitor_path, monitor)
    adapter_weights = args.parent_adapter / "adapter_model.safetensors"
    manifest_parent = next(
        (
            parent / "adapter_manifest.json"
            for parent in (args.parent_adapter, *args.parent_adapter.parents)
            if (parent / "adapter_manifest.json").is_file()
        ),
        None,
    )
    if not adapter_weights.is_file() or manifest_parent is None:
        raise FileNotFoundError("parent adapter weights/manifest are missing")
    parent_manifest = json.loads(manifest_parent.read_text())
    manifest = {
        "artifact_type": "endpoint_process_rlvr_pilot_manifest_v1",
        "selection_seed": args.seed,
        "train": {
            "rows": len(train),
            "ids_sha256": ids_sha256(train),
            "file_sha256": file_sha256(train_path),
            "source": str(args.train_source),
            "source_sha256": file_sha256(args.train_source),
            "strata": dict(Counter(row["metadata"]["event_length_stratum"] for row in train)),
        },
        "monitor": {
            "rows": len(monitor),
            "ids_sha256": ids_sha256(monitor),
            "file_sha256": file_sha256(monitor_path),
            "source": str(args.monitor_source),
            "source_sha256": file_sha256(args.monitor_source),
            "strata": dict(Counter(row["metadata"]["event_length_stratum"] for row in monitor)),
        },
        "parent": {
            "adapter": str(args.parent_adapter),
            "adapter_model_sha256": file_sha256(adapter_weights),
            "base_model_revision": parent_manifest.get("base_model_revision"),
        },
        "reward_contract": "endpoint_grounded_process_rlvr_v1",
        "chemical_runtime": {"rdkit": runtime_version, "minimum": "2026.03.4"},
        "product_only_monitor": True,
        "gold_model_visible": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
