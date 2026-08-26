#!/usr/bin/env python3
"""Build full-coverage FlowER representation baselines without ID intersection."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.iclr_tasks import (  # noqa: E402
    build_net_edit_row,
    build_outcome_only_row,
    build_proof_row,
    build_state_cot_row,
)

SPLITS = ("train", "valid", "test")
EXPECTED = {
    "outcome_only": {"train": 257171, "valid": 2890, "test": 28971},
    "state_cot": {"train": 257171, "valid": 2890, "test": 28971},
    "net_edit": {"train": 257167, "valid": 2890, "test": 28967},
    "proof": {"train": 257167, "valid": 2890, "test": 28967},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file(
    source: Path,
    target: Path,
    builder: Callable[[dict], dict],
    *,
    limit: int = 0,
) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    accepted = 0
    skipped: Counter[str] = Counter()
    with source.open(encoding="utf-8") as source_handle, temporary.open(
        "w", encoding="utf-8"
    ) as target_handle:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            if limit and accepted >= limit:
                break
            try:
                output = builder(json.loads(line))
                if not output.get("metadata", {}).get("core_precursor"):
                    raise ValueError("empty structural precursor")
                target_handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                accepted += 1
            except Exception as exc:  # preserve a reason audit before failing contract
                skipped[f"{type(exc).__name__}: {exc}"] += 1
                if sum(skipped.values()) <= 10:
                    print(f"skip {source}:{line_number}: {type(exc).__name__}: {exc}")
    temporary.replace(target)
    return {
        "source": str(source),
        "path": str(target),
        "rows": accepted,
        "skipped": sum(skipped.values()),
        "skip_reasons": dict(skipped),
        "bytes": target.stat().st_size,
        "sha256": file_sha256(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-dir", type=Path, default=Path("data/flower_full_endpoint_sft")
    )
    parser.add_argument(
        "--state-dir", type=Path, default=Path("data/mechet_sft_flower_full_v4")
    )
    parser.add_argument(
        "--proof-dir", type=Path, default=Path("data/mechet_proof_sft_flower_full_v4")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/iclr_full_v4"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["outcome_only", "state_cot", "net_edit", "proof"],
        default=["outcome_only", "state_cot", "net_edit", "proof"],
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    definitions = {
        "outcome_only": (args.endpoint_dir, build_outcome_only_row),
        "state_cot": (args.state_dir, build_state_cot_row),
        "net_edit": (args.proof_dir, build_net_edit_row),
        "proof": (args.proof_dir, build_proof_row),
    }
    manifest_path = args.output_dir / "manifest.json"
    prior_tasks = {}
    if manifest_path.is_file() and not args.limit:
        prior_tasks = dict(json.loads(manifest_path.read_text()).get("tasks") or {})
    manifest = {
        "artifact_type": "flower_full_representation_baselines_v4",
        "protocol": "product_only_task_specific_prompts_no_id_intersection",
        "full_reaction_split_rows": {"train": 257171, "valid": 2890, "test": 28971},
        "strict_proof_unavailable_upstream_corrupt_rows": {
            "train": 4,
            "valid": 0,
            "test": 4,
        },
        "tasks": prior_tasks,
    }
    for task in args.tasks:
        source_dir, builder = definitions[task]
        task_report = {}
        for split in SPLITS:
            report = build_file(
                source_dir / f"{split}.jsonl",
                args.output_dir / task / f"{split}.jsonl",
                builder,
                limit=args.limit,
            )
            expected = args.limit or EXPECTED[task][split]
            if report["rows"] != min(expected, EXPECTED[task][split]) or report["skipped"]:
                raise RuntimeError(
                    f"{task}/{split} contract failed: expected "
                    f"{min(expected, EXPECTED[task][split])}, got {report}"
                )
            task_report[split] = report
            print(f"built {task}/{split}: {report['rows']} rows", flush=True)
        manifest["tasks"][task] = task_report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
