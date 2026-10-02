#!/usr/bin/env python3
"""Stream the frozen training pool to measure execution-basis composition.

The hash probe is a feasibility estimate only, never the frozen H2 split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.proof_splits import extract_split_features


def _unit_hash(seed: int, identifier: str) -> float:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _frequency_bins(counts: Counter[str]) -> dict[str, int]:
    return {
        "one": sum(value == 1 for value in counts.values()),
        "two_to_four": sum(2 <= value <= 4 for value in counts.values()),
        "five_to_nine": sum(5 <= value <= 9 for value in counts.values()),
        "ten_or_more": sum(value >= 10 for value in counts.values()),
    }


def audit_basis(
    source: Path, *, expected_sha256: str, expected_rows: int,
    seed: int = 42, probe_fraction: float = 0.1, verify_first: int = 32,
) -> dict[str, Any]:
    if not 0 < probe_fraction < 1:
        raise ValueError("probe_fraction must be in (0,1)")
    if verify_first < 0:
        raise ValueError("verify_first must be nonnegative")
    digest = hashlib.sha256()
    ids: set[str] = set()
    primitive_counts: Counter[str] = Counter()
    composition_counts: Counter[str] = Counter()
    train_primitives: set[str] = set()
    train_compositions: set[str] = set()
    first_primitive_hash: dict[str, float] = {}
    first_composition_hash: dict[str, float] = {}
    probe_rows: list[tuple[str, tuple[str, ...]]] = []
    n_rows = 0
    n_moves = 0
    n_steps = 0
    with source.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                raise ValueError(f"blank source row after {n_rows} records")
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            identifier = str(row.get("source_id") or "")
            composition = metadata.get("execution_composition_signature")
            primitives = metadata.get("execution_primitive_signatures")
            if not identifier or identifier in ids:
                raise ValueError(f"missing or duplicate source_id: {identifier}")
            if metadata.get("executor_replayed") is not True or metadata.get("endpoint_source") != "environment_owned_trace":
                raise ValueError(f"non-replayed or non-environment endpoint: {identifier}")
            if not isinstance(composition, str) or len(composition) != 64:
                raise ValueError(f"invalid composition signature: {identifier}")
            if not isinstance(primitives, list) or not primitives or any(not isinstance(item, str) for item in primitives):
                raise ValueError(f"invalid primitive signatures: {identifier}")
            unique_primitives = tuple(sorted(set(primitives)))
            if len(unique_primitives) != len(primitives):
                raise ValueError(f"duplicated primitive signature: {identifier}")
            if n_rows < verify_first:
                recomputed = extract_split_features(row)
                if recomputed.composition != composition or recomputed.primitives != unique_primitives:
                    raise ValueError(f"stored signature/replay mismatch: {identifier}")
            ids.add(identifier)
            n_rows += 1
            n_steps += int(metadata.get("n_trace_steps") or 0)
            n_moves += int(metadata.get("n_trace_moves") or 0)
            composition_counts[composition] += 1
            primitive_counts.update(unique_primitives)
            rank = _unit_hash(seed, identifier)
            first_composition_hash[composition] = min(first_composition_hash.get(composition, 1.0), rank)
            for primitive in unique_primitives:
                first_primitive_hash[primitive] = min(first_primitive_hash.get(primitive, 1.0), rank)
            if rank < probe_fraction:
                probe_rows.append((composition, unique_primitives))
            else:
                train_compositions.add(composition)
                train_primitives.update(unique_primitives)
    actual_sha256 = digest.hexdigest()
    if n_rows != expected_rows or actual_sha256 != expected_sha256:
        raise ValueError(
            f"source contract mismatch: rows={n_rows}/{expected_rows}, "
            f"sha256={actual_sha256}/{expected_sha256}"
        )
    program_unseen = 0
    program_unseen_primitive_seen = 0
    primitive_unseen = 0
    for composition, primitives in probe_rows:
        all_seen = all(primitive in train_primitives for primitive in primitives)
        unseen_program = composition not in train_compositions
        program_unseen += unseen_program
        program_unseen_primitive_seen += unseen_program and all_seen
        primitive_unseen += not all_seen
    growth = []
    for fraction in (0.01, 0.05, 0.10, 0.25, 0.50, 1.0):
        growth.append({
            "hash_sample_fraction": fraction,
            "unique_primitives": sum(rank < fraction for rank in first_primitive_hash.values()),
            "unique_compositions": sum(rank < fraction for rank in first_composition_hash.values()),
        })
    return {
        "artifact_type": "nmi_execution_operator_basis_audit_v1",
        "scope": "strict_executable_training_pool_only_no_final_heldout_test",
        "source": str(source.resolve()),
        "source_sha256": actual_sha256,
        "primitive_basis": "source_to_sink_execution_moves_v1",
        "composition_basis": "execution_composition_signature_ordered_moves_v1",
        "rows": n_rows,
        "unique_source_ids": len(ids),
        "verified_recomputed_signatures": min(verify_first, n_rows),
        "total_execution_steps": n_steps,
        "total_electron_moves": n_moves,
        "unique_local_primitives": len(primitive_counts),
        "unique_complete_move_compositions": len(composition_counts),
        "primitive_reaction_frequency_bins": _frequency_bins(primitive_counts),
        "composition_reaction_frequency_bins": _frequency_bins(composition_counts),
        "vocabulary_growth": growth,
        "exploratory_hash_probe_not_frozen_split": {
            "seed": seed,
            "fraction": probe_fraction,
            "heldout_reactions": len(probe_rows),
            "program_unseen_reactions": program_unseen,
            "program_unseen_primitive_seen_reactions": program_unseen_primitive_seen,
            "primitive_unseen_reactions": primitive_unseen,
            "program_unseen_primitive_seen_fraction_of_heldout": (
                program_unseen_primitive_seen / len(probe_rows) if probe_rows else 0.0
            ),
            "program_unseen_primitive_seen_fraction_of_program_unseen": (
                program_unseen_primitive_seen / program_unseen if program_unseen else 0.0
            ),
        },
        "interpretation_guard": (
            "The v1 composition signature hashes ordered execution moves, not a "
            "standalone normalized import schedule; do not equate this count "
            "with unique physical mechanisms or unique reagent programs."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-fraction", type=float, default=0.1)
    parser.add_argument("--verify-first", type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    contract = manifest["splits"]["train"]
    if Path(contract["file"]).name != args.source.name:
        raise ValueError("manifest source basename mismatch")
    report = audit_basis(
        args.source, expected_sha256=str(contract["sha256"]),
        expected_rows=int(contract["rows"]), seed=args.seed,
        probe_fraction=args.probe_fraction, verify_first=args.verify_first,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen basis audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
