#!/usr/bin/env python3
"""Convert frozen MechET JSONL splits to Retro-MTGR's native five columns.

The native text files contain only reactions representable by Retro-MTGR's
one-bond/two-reactant target.  A parallel index and target JSONL retain every
source row so unsupported validation/test cases can be scored as failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Iterable, Iterator

from rdkit import Chem, RDLogger

from mechet_retro_mtgr import analyze_row, sha256_file, stable_id


SPLITS = ("train", "valid", "test")
SCHEMA_VERSION = 1


@dataclass
class ProcessedRow:
    target: dict[str, Any]
    label_specs: list[dict[str, Any]]
    native_line: str | None


def _mapped_canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _atom_map_indices(mol: Chem.Mol) -> dict[int, int]:
    result: dict[int, int] = {}
    for atom in mol.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map:
            if atom_map in result:
                raise ValueError(f"duplicate atom map {atom_map}")
            result[atom_map] = atom.GetIdx()
    return result


def _native_record(row: dict[str, Any], target: dict[str, Any]) -> str:
    """Build a five-column row whose bond indices match serialized product."""
    product = Chem.MolFromSmiles(row["product_mapped"])
    precursor = Chem.MolFromSmiles(row["precursor_mapped"])
    if product is None or precursor is None:
        raise ValueError("cannot parse mapped product or precursor")

    product_smiles = _mapped_canonical_smiles(product)
    serialized_product = Chem.MolFromSmiles(product_smiles)
    if serialized_product is None:
        raise ValueError("cannot reparse serialized product")
    product_map_indices = _atom_map_indices(serialized_product)
    center_maps = [int(value) for value in target["center_atom_maps"]]
    try:
        center = [(product_map_indices[atom_map], atom_map) for atom_map in center_maps]
    except KeyError as error:
        raise ValueError(f"center atom map missing after serialization: {error}") from error
    center.sort()
    (left_index, left_map), (right_index, right_map) = center
    if serialized_product.GetBondBetweenAtoms(left_index, right_index) is None:
        raise ValueError("serialized product center indices are not a bond")

    fragments = list(Chem.GetMolFrags(precursor, asMols=True, sanitizeFrags=False))
    if len(fragments) != 2:
        raise ValueError(f"expected two precursor fragments, found {len(fragments)}")
    fragment_maps = [set(_atom_map_indices(fragment)) for fragment in fragments]
    left_matches = [index for index, maps in enumerate(fragment_maps) if left_map in maps]
    right_matches = [index for index, maps in enumerate(fragment_maps) if right_map in maps]
    if len(left_matches) != 1 or len(right_matches) != 1:
        raise ValueError("center atom maps do not uniquely identify precursor fragments")
    if left_matches[0] == right_matches[0]:
        raise ValueError("both center endpoints occur in the same precursor fragment")
    left_fragment = _mapped_canonical_smiles(fragments[left_matches[0]])
    right_fragment = _mapped_canonical_smiles(fragments[right_matches[0]])

    target.update({
        "native_product_mapped": product_smiles,
        "native_center_atom_indices": [left_index, right_index],
        "native_center_atom_maps": [left_map, right_map],
        "native_precursor_fragments_mapped": [left_fragment, right_fragment],
    })
    return "\t".join((
        "0",
        product_smiles,
        f"{left_index},{right_index}",
        left_fragment,
        right_fragment,
    ))


def _process_item(item: tuple[int, str]) -> ProcessedRow:
    source_index, line = item
    row = json.loads(line)
    analysis = analyze_row(row)
    target = analysis.target
    target["source_index"] = source_index
    if target["status"] != "supported":
        return ProcessedRow(target, [], None)
    try:
        native_line = _native_record(row, target)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        target["status"] = "unsupported"
        target["reason"] = "native_serialization_failure"
        target["detail"] = str(error)
        return ProcessedRow(target, [], None)
    return ProcessedRow(target, analysis.label_specs, native_line)


def _nonempty_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def _ordered_process(
    source: Path,
    workers: int,
    chunksize: int,
) -> Iterable[ProcessedRow]:
    items = enumerate(_nonempty_lines(source))
    if workers == 1:
        return map(_process_item, items)
    context = mp.get_context("spawn")
    pool = context.Pool(workers)

    def iterator() -> Iterator[ProcessedRow]:
        try:
            yield from pool.imap(_process_item, items, chunksize=chunksize)
        finally:
            pool.close()
            pool.join()

    return iterator()


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n"


def _atomic_paths(output_dir: Path, split: str) -> dict[str, tuple[Path, Path]]:
    final = {
        "native": output_dir / f"{split}.txt",
        "index": output_dir / f"{split}_index.jsonl",
        "targets": output_dir / f"{split}_targets.jsonl",
    }
    return {key: (path.with_name(path.name + ".tmp"), path) for key, path in final.items()}


def _process_split(
    source: Path,
    output_dir: Path,
    workers: int,
    chunksize: int,
    progress_every: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    split = source.stem
    paths = _atomic_paths(output_dir, split)
    counts: Counter[str] = Counter()
    train_specs: dict[str, dict[str, Any]] = {}
    source_rows = 0
    native_rows = 0
    started = time.monotonic()
    with ExitStack() as stack:
        native_handle = stack.enter_context(paths["native"][0].open("w", encoding="utf-8"))
        index_handle = stack.enter_context(paths["index"][0].open("w", encoding="utf-8"))
        targets_handle = stack.enter_context(paths["targets"][0].open("w", encoding="utf-8"))
        for source_rows, processed in enumerate(
            _ordered_process(source, workers, chunksize), start=1
        ):
            target = processed.target
            status_key = "supported" if target["status"] == "supported" else target["reason"]
            counts[status_key] += 1
            native_index: int | None = None
            if processed.native_line is not None:
                native_index = native_rows
                native_handle.write(processed.native_line + "\n")
                native_rows += 1
                if split == "train":
                    for spec in processed.label_specs:
                        train_specs[spec["key"]] = spec
            index_handle.write(_json_line({
                "source_index": target["source_index"],
                "stable_id": target["stable_id"],
                "status": target["status"],
                "reason": target["reason"],
                "native_output_index": native_index,
            }))
            targets_handle.write(_json_line(target))
            if progress_every and source_rows % progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"[{split}] {source_rows} rows, {native_rows} supported, "
                    f"{source_rows / elapsed:.1f} rows/s",
                    flush=True,
                )
    for temporary, final in paths.values():
        temporary.replace(final)
    elapsed = time.monotonic() - started
    print(
        f"[{split}] complete: {source_rows} input, {native_rows} native, "
        f"{elapsed:.1f}s",
        flush=True,
    )
    report = {
        "source_file": str(source.resolve()),
        "input_rows": source_rows,
        "target_rows": source_rows,
        "native_rows": native_rows,
        "native_supported_rate": native_rows / max(source_rows, 1),
        "status_counts": dict(sorted(counts.items())),
        "elapsed_seconds": round(elapsed, 3),
        "source_sha256": sha256_file(source),
        "native_sha256": sha256_file(paths["native"][1]),
        "index_sha256": sha256_file(paths["index"][1]),
        "targets_sha256": sha256_file(paths["targets"][1]),
    }
    return report, train_specs


def _label_catalog(specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": index, **specs[key]}
        for index, key in enumerate(sorted(specs))
    ]


def _stable_id_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_native_item(item: tuple[int, str]) -> tuple[int, str]:
    line_number, line = item
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 5:
        return line_number, f"expected 5 tab-separated fields, found {len(fields)}"
    reaction_class, product_smiles, center_text, left_smiles, right_smiles = fields
    if reaction_class != "0":
        return line_number, f"reaction class is {reaction_class!r}, expected '0'"
    try:
        left_index, right_index = [int(value) for value in center_text.split(",")]
    except (TypeError, ValueError):
        return line_number, f"invalid center {center_text!r}"
    product = Chem.MolFromSmiles(product_smiles)
    left = Chem.MolFromSmiles(left_smiles)
    right = Chem.MolFromSmiles(right_smiles)
    if product is None or left is None or right is None:
        return line_number, "product or precursor fragment is not parseable"
    if not (0 <= left_index < right_index < product.GetNumAtoms()):
        return line_number, f"center indices out of range or unordered: {center_text}"
    if product.GetBondBetweenAtoms(left_index, right_index) is None:
        return line_number, f"center is not a product bond: {center_text}"
    left_map = product.GetAtomWithIdx(left_index).GetAtomMapNum()
    right_map = product.GetAtomWithIdx(right_index).GetAtomMapNum()
    if left_map not in _atom_map_indices(left):
        return line_number, f"left center map {left_map} is absent from R1"
    if right_map not in _atom_map_indices(right):
        return line_number, f"right center map {right_map} is absent from R2"
    return line_number, ""


def _validate_native(path: Path, workers: int, chunksize: int) -> dict[str, Any]:
    items = enumerate(_nonempty_lines(path), start=1)
    if workers == 1:
        results: Iterable[tuple[int, str]] = map(_validate_native_item, items)
        pool = None
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(workers)
        results = pool.imap(_validate_native_item, items, chunksize=chunksize)
    errors: list[dict[str, Any]] = []
    rows = 0
    try:
        for rows, message in results:
            if message and len(errors) < 20:
                errors.append({"line": rows, "error": message})
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {"rows": rows, "errors": errors, "valid": not errors}


def _validate_split(
    source: Path,
    output_dir: Path,
    workers: int,
    chunksize: int,
) -> tuple[dict[str, Any], set[str], set[str]]:
    split = source.stem
    index_path = output_dir / f"{split}_index.jsonl"
    targets_path = output_dir / f"{split}_targets.jsonl"
    native_path = output_dir / f"{split}.txt"
    errors: list[str] = []
    source_ids: list[str] = []
    target_ids: list[str] = []
    index_ids: list[str] = []
    heldout_label_keys: set[str] = set()
    supported_rows = 0
    expected_native_index = 0

    source_iter = _nonempty_lines(source)
    target_iter = _nonempty_lines(targets_path)
    index_iter = _nonempty_lines(index_path)
    row_count = 0
    while True:
        batch = [next(iterator, None) for iterator in (source_iter, target_iter, index_iter)]
        if batch == [None, None, None]:
            break
        row_count += 1
        if any(line is None for line in batch):
            errors.append("source, target, and index row counts differ")
            break
        source_row, target, index = (json.loads(line) for line in batch if line is not None)
        source_identifier = stable_id(source_row)
        source_ids.append(source_identifier)
        target_ids.append(str(target.get("stable_id", "")))
        index_ids.append(str(index.get("stable_id", "")))
        expected_source_index = row_count - 1
        if target.get("source_index") != expected_source_index:
            errors.append(f"target source_index mismatch at row {row_count}")
        if index.get("source_index") != expected_source_index:
            errors.append(f"index source_index mismatch at row {row_count}")
        if source_identifier != target.get("stable_id") or source_identifier != index.get("stable_id"):
            errors.append(f"stable_id mismatch at row {row_count}")
        if target.get("product_mapped") != source_row.get("product_mapped", ""):
            errors.append(f"mapped product changed at row {row_count}")
        if target.get("product_unmapped") != source_row.get("product_unmapped", ""):
            errors.append(f"unmapped product changed at row {row_count}")
        if target.get("reference_precursors") != source_row.get("precursor_unmapped", ""):
            errors.append(f"reference precursor changed at row {row_count}")
        if target.get("status") != index.get("status") or target.get("reason") != index.get("reason"):
            errors.append(f"target/index status mismatch at row {row_count}")
        if target.get("status") == "supported":
            supported_rows += 1
            heldout_label_keys.update((target["left_label_key"], target["right_label_key"]))
            if index.get("native_output_index") != expected_native_index:
                errors.append(f"native index is not contiguous at row {row_count}")
            expected_native_index += 1
        elif index.get("native_output_index") is not None:
            errors.append(f"unsupported row has native output at row {row_count}")
        if len(errors) >= 20:
            break

    native_validation = _validate_native(native_path, workers, chunksize)
    errors.extend(
        f"native line {item['line']}: {item['error']}"
        for item in native_validation["errors"]
    )
    checks = {
        "source_target_index_row_counts_equal": row_count == len(source_ids) == len(target_ids) == len(index_ids),
        "all_source_rows_retained_in_targets_and_index": len(source_ids) == row_count,
        "stable_id_order_preserved": source_ids == target_ids == index_ids,
        "stable_ids_unique": len(set(source_ids)) == len(source_ids),
        "native_row_count_matches_supported": native_validation["rows"] == supported_rows,
        "native_five_column_rows_valid": native_validation["valid"],
        "source_fields_preserved": not any(
            marker in error
            for error in errors
            for marker in ("product changed", "precursor changed")
        ),
    }
    return ({
        "input_rows": len(source_ids),
        "target_rows": len(target_ids),
        "index_rows": len(index_ids),
        "native_rows": native_validation["rows"],
        "supported_rows": supported_rows,
        "stable_id_sequence_sha256": _stable_id_digest(source_ids),
        "checks": checks,
        "errors": errors[:20],
        "valid": all(checks.values()) and not errors,
    }, set(source_ids), heldout_label_keys)


def validate_outputs(
    input_dir: Path,
    output_dir: Path,
    workers: int,
    chunksize: int,
) -> dict[str, Any]:
    split_reports: dict[str, Any] = {}
    split_ids: dict[str, set[str]] = {}
    split_label_keys: dict[str, set[str]] = {}
    for split in SPLITS:
        report, identifiers, label_keys = _validate_split(
            input_dir / f"{split}.jsonl", output_dir, workers, chunksize
        )
        split_reports[split] = report
        split_ids[split] = identifiers
        split_label_keys[split] = label_keys
    vocabulary = json.loads((output_dir / "label_vocabulary.json").read_text(encoding="utf-8"))
    train_keys = {item["key"] for item in vocabulary["labels"]}
    checks = {
        "all_splits_valid": all(report["valid"] for report in split_reports.values()),
        "no_cross_split_stable_id_overlap": not (
            split_ids["train"] & split_ids["valid"]
            or split_ids["train"] & split_ids["test"]
            or split_ids["valid"] & split_ids["test"]
        ),
        "label_vocabulary_declares_train_only": vocabulary.get("source_splits") == ["train"],
        "label_vocabulary_equals_supported_train_labels": train_keys == split_label_keys["train"],
    }
    for split in SPLITS:
        split_reports[split]["label_keys"] = len(split_label_keys[split])
        split_reports[split]["label_keys_unseen_from_train"] = len(
            split_label_keys[split] - train_keys
        )
    report = {
        "artifact_type": "retro_mtgr_native_processing_validation",
        "checks": checks,
        "splits": split_reports,
        "valid": all(checks.values()),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def process_dataset(
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    workers: int,
    chunksize: int,
    progress_every: int,
) -> dict[str, Any]:
    missing = [str(input_dir / f"{split}.jsonl") for split in SPLITS if not (input_dir / f"{split}.jsonl").is_file()]
    if missing:
        raise FileNotFoundError("missing frozen split files: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    train_specs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        report, specs = _process_split(
            input_dir / f"{split}.jsonl",
            output_dir,
            workers,
            chunksize,
            progress_every,
        )
        split_reports[split] = report
        if split == "train":
            train_specs = specs

    catalog = _label_catalog(train_specs)
    vocabulary_path = output_dir / "label_vocabulary.json"
    vocabulary_path.write_text(
        json.dumps({
            "source_splits": ["train"],
            "labels": catalog,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_outputs(input_dir, output_dir, workers, chunksize)
    for split in SPLITS:
        split_reports[split]["label_keys"] = validation["splits"][split]["label_keys"]
        split_reports[split]["label_keys_unseen_from_train"] = validation["splits"][split]["label_keys_unseen_from_train"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "retro_mtgr_native_processed_frozen_splits",
        "dataset": dataset,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "native_format": ["reaction_class", "product_mapped", "break_bond_indices", "reactant_1_mapped", "reactant_2_mapped"],
        "reaction_class_policy": "constant_zero; reaction-class features must be disabled",
        "target_contract": "single product-bond deletion plus two endpoint leaving-group labels",
        "source_splits_preserved": list(SPLITS),
        "resplit": False,
        "native_text_contains": "supported rows only",
        "all_rows_retained_in": ["*_index.jsonl", "*_targets.jsonl"],
        "unsupported_validation_and_test_policy": "retain stable IDs and score missing predictions as failures",
        "label_vocabulary_source_splits": ["train"],
        "label_vocabulary_size": len(catalog),
        "label_vocabulary_sha256": sha256_file(vocabulary_path),
        "splits": split_reports,
        "validation_passed": validation["valid"],
        "validation_report": str((output_dir / "validation_report.json").resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not validation["valid"]:
        raise RuntimeError(f"output validation failed; see {output_dir / 'validation_report.json'}")
    print(json.dumps({
        "dataset": dataset,
        "output_dir": str(output_dir.resolve()),
        "validation_passed": True,
        "label_vocabulary_size": len(catalog),
        "splits": {
            split: {
                "input_rows": split_reports[split]["input_rows"],
                "native_rows": split_reports[split]["native_rows"],
                "status_counts": split_reports[split]["status_counts"],
            }
            for split in SPLITS
        },
    }, indent=2, sort_keys=True))
    return manifest


def cli(dataset: str, default_input_dir: Path, default_output_dir: Path) -> int:
    parser = argparse.ArgumentParser(
        description=f"Prepare frozen {dataset} splits for Retro-MTGR."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")
    RDLogger.DisableLog("rdApp.warning")
    process_dataset(
        dataset=dataset,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        chunksize=args.chunksize,
        progress_every=args.progress_every,
    )
    return 0
