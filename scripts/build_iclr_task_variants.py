#!/usr/bin/env python3
"""Build matched outcome/state/edit/proof task variants from frozen row ids."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.iclr_tasks import (
    build_net_edit_row,
    build_outcome_only_row,
    build_proof_row,
    build_state_cot_row,
)


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row.get("id")): row for row in rows}


def canonical_proof_id(row_id: str) -> str:
    """Map trace-owned wrapper IDs back to the underlying FlowER proof ID."""
    prefix = "textbook-tool-sft:"
    return row_id[len(prefix) :] if row_id.startswith(prefix) else row_id


def load_id_filter(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(canonical_proof_id(str(json.loads(line).get("id") or "")))
    ids.discard("")
    return ids


def write_task(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-input", type=Path, required=True)
    parser.add_argument("--state-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--id-filter-from",
        type=Path,
        default=None,
        help="Optional JSONL whose IDs freeze the replayable common subset.",
    )
    parser.add_argument("--tasks", nargs="+", choices=["outcome_only", "state_cot", "net_edit", "proof"], default=["outcome_only", "state_cot", "net_edit", "proof"])
    args = parser.parse_args()

    proof_rows = load(args.proof_input)
    state_rows = load(args.state_input) if args.state_input else {}
    common_ids = set(proof_rows)
    allowed_ids = load_id_filter(args.id_filter_from) if args.id_filter_from else None
    if allowed_ids is not None:
        common_ids &= allowed_ids
    if "state_cot" in args.tasks:
        if not state_rows:
            raise ValueError("--state-input is required for the state_cot baseline")
        common_ids &= set(state_rows)
    ordered_ids = sorted(common_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    prior_manifest = {}
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "proof_input": str(args.proof_input),
        "state_input": str(args.state_input) if args.state_input else None,
        "n_proof": len(proof_rows),
        "n_state": len(state_rows),
        "id_filter_from": str(args.id_filter_from) if args.id_filter_from else None,
        "n_filter_ids": len(allowed_ids) if allowed_ids is not None else None,
        "n_matched": len(ordered_ids),
        "tasks": dict(prior_manifest.get("tasks") or {}),
    }

    builders = {
        "outcome_only": (build_outcome_only_row, proof_rows),
        "net_edit": (build_net_edit_row, proof_rows),
        "proof": (build_proof_row, proof_rows),
        "state_cot": (build_state_cot_row, state_rows),
    }
    for task in args.tasks:
        builder, source = builders[task]
        accepted: list[dict] = []
        skipped: Counter[str] = Counter()
        for row_id in ordered_ids:
            try:
                accepted.append(builder(source[row_id]))
            except Exception as exc:
                skipped[type(exc).__name__] += 1
        path = args.output_dir / f"{task}.jsonl"
        write_task(path, accepted)
        manifest["tasks"][task] = {
            "path": str(path),
            "n_accepted": len(accepted),
            "n_skipped": len(ordered_ids) - len(accepted),
            "skip_reasons": dict(skipped),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
