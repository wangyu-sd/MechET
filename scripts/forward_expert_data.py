#!/usr/bin/env python3
"""Download, inspect, standardize and split data for the forward expert.

Restricted datasets are never downloaded silently. The caller must explicitly
acknowledge the upstream license, and every download writes a provenance
manifest with the repository id, revision and local file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.forward_data import (
    flatten_step_examples,
    iter_source_rows,
    iter_standardized,
    standardize_path,
)

DATASETS = {
    "mech_uspto_31k": {
        "repo_id": "SchwallerGroup/mech_uspto_31k",
        "license": "CC-BY-4.0",
        "restricted": False,
        "description": "Mechanistic USPTO polar reactions with source-sink arrows",
    },
    "flower_dataset": {
        "repo_id": "SchwallerGroup/flower_dataset",
        "license": "see upstream dataset card",
        "restricted": False,
        "description": "FlowER reaction/mechanism dataset mirror",
    },
    "pmechdb_elem": {
        "repo_id": "SchwallerGroup/pmechdb_elem",
        "license": "CC-BY-NC-ND; explicit upstream acceptance required",
        "restricted": True,
        "description": "Curated elementary polar mechanism steps",
    },
    "ord_data": {
        "repo_id": "open-reaction-database/ord-data",
        "license": "CC-BY-SA-4.0",
        "restricted": False,
        "description": "Open Reaction Database mirror; condition/outcome supervision",
    },
}

MODELS = {
    "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
    "molformer": "ibm-research/MoLFormer-XL-both-10pct",
    "qwen_small": "Qwen/Qwen3-0.6B",
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path, payload: dict) -> None:
    files = []
    for path in sorted(
        file
        for file in root.rglob("*")
        if file.is_file() and file.name != "manifest.json"
    ):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    payload["files"] = files
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def snapshot(
    repo_id: str,
    output: Path,
    *,
    repo_type: str,
    revision: str,
    allow_patterns: list[str] | None,
    dry_run: bool,
) -> None:
    if dry_run:
        print(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "output": str(output),
                    "allow_patterns": allow_patterns,
                },
                indent=2,
            )
        )
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install mechet[data] for download support") from exc
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(output),
        allow_patterns=allow_patterns or None,
    )


def command_download(args) -> None:
    spec = DATASETS[args.dataset]
    if spec["restricted"] and not args.accept_restricted_license:
        raise SystemExit(
            f"{args.dataset} is restricted ({spec['license']}). "
            "Review the upstream license and rerun with "
            "--accept-restricted-license."
        )
    output = args.output / args.dataset
    snapshot(
        spec["repo_id"],
        output,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=args.allow_pattern,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        _manifest(
            output,
            {
                "kind": "dataset",
                "name": args.dataset,
                **spec,
                "revision": args.revision,
            },
        )
        print(output)


def command_predownload(args) -> None:
    names = args.model or list(MODELS)
    for name in names:
        repo_id = MODELS.get(name, name)
        output = args.output / name.replace("/", "__")
        snapshot(
            repo_id,
            output,
            repo_type="model",
            revision=args.revision,
            allow_patterns=args.allow_pattern,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            _manifest(
                output,
                {
                    "kind": "model",
                    "name": name,
                    "repo_id": repo_id,
                    "revision": args.revision,
                },
            )
            print(output)


def command_inspect(args) -> None:
    rows = []
    for index, row in enumerate(iter_source_rows(args.input)):
        rows.append({"index": index, "keys": sorted(row), "sample": row})
        if len(rows) >= args.rows:
            break
    print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))


def command_standardize(args) -> None:
    report = standardize_path(
        args.input,
        args.output,
        source=args.source,
        quarantine_path=args.quarantine,
        require_maps=not args.allow_unmapped,
        limit=args.limit,
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), indent=2))


def command_build(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "valid", "test")
    }
    counts = {split: 0 for split in handles}
    try:
        for example in flatten_step_examples(iter_standardized(args.input)):
            split = example.get("split", "train")
            handles[split].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"source": str(args.input), "step_counts": counts}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))


def command_map(args) -> None:
    try:
        from rxnmapper import RXNMapper
    except ImportError as exc:
        raise RuntimeError(
            "install mechet[mapping] to atom-map outcome-only reactions"
        ) from exc
    rows = list(iter_standardized(args.input))
    mapper = RXNMapper()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            reactions = [f"{row['reactants']}>>{row['products']}" for row in chunk]
            mapped = mapper.get_attention_guided_atom_maps(reactions=reactions)
            for row, result in zip(chunk, mapped):
                mapped_rxn = str(
                    result.get("mapped_rxn")
                    or result.get("mapped_reaction")
                    or ""
                )
                if ">>" not in mapped_rxn:
                    continue
                reactants, products = mapped_rxn.split(">>", 1)
                row = dict(row)
                row["reactants"] = reactants
                row["products"] = products
                row["reaction_smiles"] = (
                    f"{reactants}>{row.get('reagents', '')}>{products}"
                )
                row.setdefault("metadata", {})[
                    "atom_mapping_confidence"
                ] = result.get("confidence")
                row["metadata"]["atom_mapper"] = "rxnmapper"
                row["steps"] = []
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(
        json.dumps(
            {"read": len(rows), "written": written, "output": str(args.output)},
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download a registered dataset snapshot")
    download.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    download.add_argument("--output", type=Path, default=Path("data/raw"))
    download.add_argument("--revision", default="main")
    download.add_argument("--allow-pattern", action="append", default=[])
    download.add_argument("--accept-restricted-license", action="store_true")
    download.add_argument("--dry-run", action="store_true")
    download.set_defaults(func=command_download)

    models = sub.add_parser(
        "predownload",
        help="pre-download optional baseline/backbone models",
    )
    models.add_argument("--model", action="append", default=[])
    models.add_argument("--output", type=Path, default=Path("models/baselines"))
    models.add_argument("--revision", default="main")
    models.add_argument("--allow-pattern", action="append", default=[])
    models.add_argument("--dry-run", action="store_true")
    models.set_defaults(func=command_predownload)

    inspect = sub.add_parser("inspect", help="show source columns and a few rows")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--rows", type=int, default=3)
    inspect.set_defaults(func=command_inspect)

    standardize = sub.add_parser(
        "standardize",
        help="write the conservative canonical reaction schema",
    )
    standardize.add_argument("--input", type=Path, required=True)
    standardize.add_argument("--output", type=Path, required=True)
    standardize.add_argument("--source", required=True)
    standardize.add_argument("--quarantine", type=Path)
    standardize.add_argument("--allow-unmapped", action="store_true")
    standardize.add_argument("--limit", type=int, default=0)
    standardize.set_defaults(func=command_standardize)

    mapping = sub.add_parser(
        "map",
        help="atom-map standardized outcome-only reactions with RXNMapper",
    )
    mapping.add_argument("--input", type=Path, required=True)
    mapping.add_argument("--output", type=Path, required=True)
    mapping.add_argument("--batch-size", type=int, default=32)
    mapping.set_defaults(func=command_map)

    build = sub.add_parser(
        "build",
        help="flatten standardized trajectories and outcome-only rows into train/valid/test data",
    )
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(func=command_build)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
