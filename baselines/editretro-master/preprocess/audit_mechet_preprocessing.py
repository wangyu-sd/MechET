#!/usr/bin/env python3
"""Validate EditRetro preprocessing, Fairseq data, IDs, and endpoint identity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(row: dict[str, Any]) -> str:
    value = str(row.get("stable_id") or row.get("id") or "").strip()
    if not value:
        raise ValueError("reference row missing stable_id/id")
    return value


def canonical_key(smiles: str) -> str:
    parts: list[str] = []
    for fragment in smiles.split("."):
        molecule = Chem.MolFromSmiles(fragment.strip())
        if molecule is None:
            raise ValueError(f"invalid SMILES: {smiles}")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
        parts.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return ".".join(sorted(parts))


def load_dictionary(path: Path) -> set[str]:
    vocabulary: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").rsplit(" ", 1)
            if len(fields) != 2:
                raise ValueError(f"invalid dictionary row at {path}:{line_number}")
            vocabulary.add(fields[0])
    vocabulary.update({"<s>", "<pad>", "</s>", "<unk>"})
    return vocabulary


def count_lines_and_oov(path: Path, vocabulary: set[str]) -> tuple[int, Counter[str]]:
    count = 0
    oov: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            count += 1
            for token in line.strip().split():
                if token not in vocabulary:
                    oov[token] += 1
    return count, oov


def expected_reference_rows(
    reference: Path, profile: str
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """Stream reference rows and canonicalize each endpoint exactly once."""
    identifiers: list[str] = []
    identities: dict[str, tuple[str, str]] = {}
    for row_index, row in enumerate(read_jsonl(reference)):
        if profile == "audit100" and row_index >= 100:
            break
        identifier = stable_id(row)
        if identifier in identities:
            raise ValueError(f"duplicate stable ID in {reference}: {identifier}")
        product = canonical_key(
            str(row.get("product_unmapped") or row.get("product_mapped") or "")
        )
        precursor = canonical_key(
            str(
                row.get("precursor_unmapped")
                or row.get("precursor_mapped")
                or ""
            )
        )
        identifiers.append(identifier)
        identities[identifier] = (product, precursor)
    return identifiers, identities


def audit_split(args: argparse.Namespace, logical_split: str) -> dict[str, Any]:
    native_split = "val" if logical_split == "valid" else logical_split
    reference_path = args.reference_dir / f"{logical_split}.jsonl"
    expected_ids, expected_identity = expected_reference_rows(
        reference_path, args.profile
    )
    expected_id_set = set(expected_ids)

    source_path = args.preprocessed_dir / f"{native_split}.src"
    target_path = args.preprocessed_dir / f"{native_split}.tgt"
    line_map_path = args.preprocessed_dir / f"{native_split}.line_map.jsonl"
    report_path = args.preprocessed_dir / f"{native_split}.preprocess_report.json"
    failures_path = args.preprocessed_dir / f"{native_split}.failures.jsonl"
    for path in (source_path, target_path, line_map_path, report_path, failures_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = list(read_jsonl(failures_path))
    source_count, source_oov = count_lines_and_oov(source_path, args.vocabulary)
    target_count, target_oov = count_lines_and_oov(target_path, args.vocabulary)

    errors: list[str] = []
    warnings: list[str] = []
    expected_views = len(expected_ids) * args.augmentation
    if source_count != expected_views:
        errors.append(f"emitted {source_count} views, expected {expected_views}")
    if failures:
        errors.append(f"native preprocessing has {len(failures)} failures")
    if source_oov:
        errors.append(
            f"dictionary source OOV tokens: {sum(source_oov.values())}"
        )
    if target_oov:
        message = f"dictionary target OOV tokens: {sum(target_oov.values())}"
        if logical_split == "train":
            errors.append(message)
        else:
            # Held-out targets are evaluator references, not model inputs or
            # vocabulary sources.  Recording their OOVs avoids test leakage;
            # Fairseq maps them to <unk> only in the optional target sidecar.
            warnings.append(message)

    views_by_id: Counter[str] = Counter()
    wrong_augmentation_ids: set[str] = set()
    identity_mismatch_count = 0
    identity_mismatch_samples: list[str] = []
    order_mismatch_count = 0
    order_mismatch_samples: list[int] = []
    unaligned_ids: set[str] = set()
    line_map_count = 0
    for line_index, metadata in enumerate(read_jsonl(line_map_path)):
        line_map_count += 1
        identifier = str(metadata.get("stable_id") or "")
        augmentation_index = int(metadata.get("augmentation_index", -1))
        views_by_id[identifier] += 1
        expected_row_index = line_index // args.augmentation
        expected_identifier = (
            expected_ids[expected_row_index]
            if expected_row_index < len(expected_ids)
            else ""
        )
        expected_augmentation = line_index % args.augmentation
        if (
            identifier != expected_identifier
            or augmentation_index != expected_augmentation
            or int(metadata.get("line_index", -1)) != line_index
        ):
            order_mismatch_count += 1
            if len(order_mismatch_samples) < 10:
                order_mismatch_samples.append(line_index)
        if augmentation_index != expected_augmentation:
            wrong_augmentation_ids.add(identifier or expected_identifier)
        identity = expected_identity.get(identifier)
        if identity is None:
            identity_mismatch_count += 1
            if len(identity_mismatch_samples) < 10:
                identity_mismatch_samples.append(identifier or f"line:{line_index}")
            continue
        expected_product, expected_precursor = identity
        if (
            metadata.get("product_canonical") != expected_product
            or metadata.get("precursor_canonical") != expected_precursor
        ):
            identity_mismatch_count += 1
            if len(identity_mismatch_samples) < 10:
                identity_mismatch_samples.append(identifier)
        if int(metadata.get("unaligned_precursor_fragment_count", 0)) > 0:
            unaligned_ids.add(identifier)

    if source_count != target_count or source_count != line_map_count:
        errors.append(
            "src/tgt/line-map counts differ: "
            f"{source_count}/{target_count}/{line_map_count}"
        )

    missing_ids = [identifier for identifier in expected_ids if identifier not in views_by_id]
    extra_ids = sorted(set(views_by_id) - expected_id_set)
    wrong_augmentation_ids.update(
        identifier
        for identifier in expected_ids
        if views_by_id.get(identifier, 0) != args.augmentation
    )
    wrong_augmentation_ids.discard("")
    if missing_ids:
        errors.append(f"missing stable IDs: {missing_ids[:10]}")
    if extra_ids:
        errors.append(f"unexpected stable IDs: {extra_ids[:10]}")
    if wrong_augmentation_ids:
        errors.append(f"bad augmentation coverage: {wrong_augmentation_ids[:10]}")
    if order_mismatch_count:
        errors.append(f"line order mismatches: {order_mismatch_samples}")
    if identity_mismatch_count:
        errors.append(f"endpoint identity mismatches: {identity_mismatch_samples}")
    if not bool(report.get("complete_coverage")):
        errors.append("preprocess report does not declare complete coverage")
    if int(report.get("rows_with_unaligned_precursor_fragments", 0)) != len(
        unaligned_ids
    ):
        errors.append("unaligned precursor fragment report disagrees with line map")

    binary_split = "valid" if logical_split == "valid" else native_split
    required_binary = (
        f"{binary_split}.src-tgt.src.bin",
        f"{binary_split}.src-tgt.src.idx",
        f"{binary_split}.src-tgt.tgt.bin",
        f"{binary_split}.src-tgt.tgt.idx",
    )
    missing_binary = [
        name for name in required_binary if not (args.preprocessed_dir / "data-bin" / name).is_file()
    ]
    if missing_binary and not args.allow_missing_binarization:
        errors.append(f"missing Fairseq binaries: {missing_binary}")

    return {
        "logical_split": logical_split,
        "native_split": native_split,
        "reference": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
        "reference_rows_used": len(expected_ids),
        "expected_views": expected_views,
        "source_lines": source_count,
        "target_lines": target_count,
        "line_map_rows": line_map_count,
        "failure_rows": len(failures),
        "stable_ids_with_views": len(views_by_id),
        "missing_stable_ids": len(missing_ids),
        "extra_stable_ids": len(extra_ids),
        "wrong_augmentation_ids": len(wrong_augmentation_ids),
        "line_order_mismatches": order_mismatch_count,
        "identity_mismatches": identity_mismatch_count,
        "rows_with_unaligned_precursor_fragments": len(unaligned_ids),
        "source_oov_occurrences": sum(source_oov.values()),
        "target_oov_occurrences": sum(target_oov.values()),
        "source_oov_types": source_oov.most_common(20),
        "target_oov_types": target_oov.most_common(20),
        "missing_fairseq_binaries": missing_binary,
        "heldout_target_oov_allowed": logical_split != "train",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("mech_uspto_31k_full", "flower_full"), required=True
    )
    parser.add_argument("--profile", choices=("full", "audit100"), default="full")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--augmentation", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing-binarization", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.augmentation <= 0:
        parser.error("--augmentation must be positive")
    args.reference_dir = args.reference_dir.resolve()
    args.preprocessed_dir = args.preprocessed_dir.resolve()
    args.dictionary = args.dictionary.resolve()
    args.vocabulary = load_dictionary(args.dictionary)
    reports = [audit_split(args, split) for split in ("train", "valid", "test")]
    passed = all(report["passed"] for report in reports)
    binarization_complete = all(
        not report["missing_fairseq_binaries"] for report in reports
    )
    training_ready = passed and binarization_complete
    output = args.output or args.preprocessed_dir / "mechet_preprocessing_audit.json"
    status_path = args.preprocessed_dir / "ARTIFACT_STATUS.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "editretro_mechet_preprocessing_audit",
        "dataset": args.dataset,
        "profile": args.profile,
        "augmentation": args.augmentation,
        "dictionary": str(args.dictionary),
        "dictionary_sha256": sha256_file(args.dictionary),
        "native_representation": "root_aligned_smiles_spe",
        "oracle_edit_targets": "generated_dynamically_by_editretro_nat",
        "iterative_decoder_preserved": True,
        "resplit": False,
        "mechanism_supervision": False,
        "native_preprocessing_passed": passed,
        "binarization_complete": binarization_complete,
        "training_ready": training_ready,
        "splits": {report["logical_split"]: report for report in reports},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status = {
        "status": (
            "validated"
            if training_ready
            else "preprocessing_validated_binarization_pending"
            if passed
            else "invalid"
        ),
        "training_allowed": training_ready,
        "artifact": str(args.preprocessed_dir),
        "audit": str(output.resolve()),
        "audit_sha256": sha256_file(output),
    }
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
