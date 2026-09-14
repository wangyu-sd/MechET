#!/usr/bin/env python3
"""Run strict RetroSynFlow preprocessing on a frozen MechET dataset export."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from retflow.datasets.data.uspto import USPTO
from retflow.datasets.data.uspto_synthon import SynthonUSPTO
from retflow.datasets.data.preprocess_utils import stable_ids_sha256


BUILDERS = {
    "direct": USPTO,
    "synthon": SynthonUSPTO,
}

RAW_FILES = {
    "train": "uspto50k_train.csv",
    "val": "uspto50k_val.csv",
    "test": "uspto50k_test.csv",
}
ATOM_MAP_PATTERN = re.compile(r":(\d+)\]")


def maximum_required_dummy_nodes(root: Path) -> tuple[int, dict[str, int]]:
    """Find one fixed capacity without using each row's target atom count."""
    maxima = {}
    for split, filename in RAW_FILES.items():
        split_maximum = 0
        with (root / "raw" / filename).open(
            newline="", encoding="utf-8"
        ) as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                parts = str(row["reactants>reagents>production"]).split(">")
                if len(parts) != 3:
                    raise ValueError(
                        f"invalid reaction separators in {filename} row {row_index}"
                    )
                reactant_maps = set(ATOM_MAP_PATTERN.findall(parts[0]))
                product_maps = set(ATOM_MAP_PATTERN.findall(parts[2]))
                if not reactant_maps or not product_maps:
                    raise ValueError(
                        f"missing atom maps in {filename} row {row_index}"
                    )
                split_maximum = max(
                    split_maximum, len(reactant_maps - product_maps)
                )
        maxima[split] = split_maximum
    return max(maxima.values()), maxima


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--builders",
        nargs="+",
        choices=tuple(BUILDERS),
        default=list(BUILDERS),
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help=(
            "save supported rows and record unsupported rows; evaluation must "
            "count every recorded test failure as an incorrect prediction"
        ),
    )
    parser.add_argument(
        "--generalize-unsupported-synthons",
        action="store_true",
        help="use the non-published multi-center synthon fallback",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument(
        "--max-dummy-nodes",
        type=int,
        help=(
            "fixed product/synthon dummy capacity; defaults to the maximum "
            "reactant-only atom-map count across all raw splits"
        ),
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not (root / "raw").is_dir():
        raise FileNotFoundError(f"missing raw dataset directory: {root / 'raw'}")
    if args.max_dummy_nodes is None:
        max_dummy_nodes, required_by_split = maximum_required_dummy_nodes(root)
    else:
        if args.max_dummy_nodes < 0:
            raise ValueError("--max-dummy-nodes must be non-negative")
        max_dummy_nodes = args.max_dummy_nodes
        _, required_by_split = maximum_required_dummy_nodes(root)

    summaries: list[dict[str, Any]] = []
    failed = False
    for builder_name in args.builders:
        dataset_class = BUILDERS[builder_name]
        for split in args.splits:
            report_path = root / "processed" / f"{builder_name}_{split}.preprocessing.json"
            try:
                builder_kwargs = dict(
                    split=split,
                    root=str(root),
                    download_and_process=True,
                    max_dummy_nodes=max_dummy_nodes,
                    allow_failed_rows=args.allow_failures,
                )
                if builder_name == "synthon":
                    builder_kwargs["generalize_unsupported_synthons"] = (
                        args.generalize_unsupported_synthons
                    )
                dataset = dataset_class(**builder_kwargs)
                stable_ids = [str(dataset[i].stable_id) for i in range(len(dataset))]
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["loaded_rows"] = len(dataset)
                report["loaded_stable_ids_unique"] = len(set(stable_ids))
                report["loaded_order_sha256_matches"] = (
                    stable_ids_sha256(stable_ids)
                    == report["output_stable_ids_sha256"]
                )
                expected_rows = (
                    report["output_rows"]
                    if args.allow_failures
                    else report["input_rows"]
                )
                if len(dataset) != expected_rows:
                    raise RuntimeError("saved graph cache row count disagrees with audit")
                if not args.allow_failures and not report["coverage_ok"]:
                    raise RuntimeError("saved graph cache does not satisfy strict coverage")
                if not report["loaded_order_sha256_matches"]:
                    raise RuntimeError("saved graph cache order disagrees with audit")
                configured_capacity = report["statistics"].get(
                    "configured_max_dummy_nodes"
                )
                if configured_capacity != max_dummy_nodes:
                    raise RuntimeError(
                        "saved graph cache has dummy capacity "
                        f"{configured_capacity}, expected {max_dummy_nodes}"
                    )
                if builder_name == "synthon":
                    cached_generalization = report["statistics"].get(
                        "generalization_enabled"
                    )
                    if cached_generalization != args.generalize_unsupported_synthons:
                        raise RuntimeError(
                            "saved synthon cache generalization setting is "
                            f"{cached_generalization}, expected "
                            f"{args.generalize_unsupported_synthons}"
                        )
                summaries.append(report)
            except Exception as error:
                failed = True
                summary: dict[str, Any] = {
                    "builder": builder_name,
                    "split": split,
                    "coverage_ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                if report_path.is_file():
                    summary["audit"] = json.loads(report_path.read_text(encoding="utf-8"))
                summaries.append(summary)

    print(
        json.dumps(
            {
                "dataset_root": str(root),
                "max_dummy_nodes": max_dummy_nodes,
                "required_dummy_nodes_by_split": required_by_split,
                "results": summaries,
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
