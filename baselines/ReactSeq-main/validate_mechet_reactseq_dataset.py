#!/usr/bin/env python3
"""Stream-validate a prepared MechET dataset for native ReactSeq training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} is not an object".format(path, line_number))
            yield value


def stable_id(row: Dict[str, Any]) -> str:
    value = str(row.get("id") or row.get("stable_id") or "").strip()
    if not value:
        raise ValueError("row is missing id/stable_id")
    return value


def product_smiles(row: Dict[str, Any]) -> str:
    return str(row.get("product_mapped") or row.get("target_smiles") or "").strip()


def precursor_smiles(row: Dict[str, Any]) -> str:
    return str(
        row.get("precursor_mapped")
        or row.get("structural_precursor")
        or row.get("expected_precursor")
        or ""
    ).strip()


def validate_source_equivalence(frozen_path: Path, conversion_path: Path) -> Dict[str, Any]:
    rows = 0
    id_mismatches = 0
    product_mismatches = 0
    precursor_mismatches = 0
    for frozen, conversion in zip_longest(
        read_jsonl(frozen_path), read_jsonl(conversion_path)
    ):
        if frozen is None or conversion is None:
            raise ValueError("source files have different row counts")
        rows += 1
        id_mismatches += int(stable_id(frozen) != stable_id(conversion))
        product_mismatches += int(product_smiles(frozen) != product_smiles(conversion))
        precursor_mismatches += int(
            precursor_smiles(frozen) != precursor_smiles(conversion)
        )
    return {
        "frozen_input": str(frozen_path.resolve()),
        "conversion_input": str(conversion_path.resolve()),
        "rows": rows,
        "stable_id_mismatches": id_mismatches,
        "mapped_product_mismatches": product_mismatches,
        "structural_precursor_mismatches": precursor_mismatches,
        "passed": id_mismatches == 0
        and product_mismatches == 0
        and precursor_mismatches == 0,
    }


def next_json(handle: TextIO, label: str) -> Dict[str, Any]:
    line = handle.readline()
    if line == "":
        raise ValueError("{} ended early".format(label))
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("{} contains a non-object row".format(label))
    return value


def next_line(handle: TextIO, label: str) -> str:
    line = handle.readline()
    if line == "":
        raise ValueError("{} ended early".format(label))
    return line.rstrip("\n")


def assert_exhausted(handle: TextIO, label: str) -> None:
    if handle.readline() != "":
        raise ValueError("{} contains extra rows".format(label))


def validate_split(
    split: str, input_path: Path, output_dir: Path, vocabulary: set
) -> Dict[str, Any]:
    reference_ids = [stable_id(row) for row in read_jsonl(input_path)]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("{} input contains duplicate stable IDs".format(split))

    native_report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    evaluation_only = bool(
        native_report.get("evaluation_only", split == "test")
    )
    augmentations = int(native_report["augmentations"])
    expected_attempts = len(reference_ids) * augmentations
    if int(native_report["input_rows"]) != len(reference_ids):
        raise ValueError("{} report input_rows is stale".format(split))
    if int(native_report["expected_attempts"]) != expected_attempts:
        raise ValueError("{} report expected_attempts is stale".format(split))

    emitted = 0
    failed = 0
    empty_sources = 0
    empty_targets = 0
    oov: Counter = Counter()
    status_counts: Counter = Counter()
    failed_reaction_ids = set()
    ledger_path = output_dir / "ledger.jsonl"
    map_path = output_dir / "line_map.jsonl"
    failure_path = output_dir / "failures.jsonl"
    src_path = output_dir / "src.txt"
    tgt_path = output_dir / "tgt.txt"

    with ledger_path.open(encoding="utf-8") as ledger_handle, map_path.open(
        encoding="utf-8"
    ) as map_handle, failure_path.open(encoding="utf-8") as failure_handle, src_path.open(
        encoding="utf-8"
    ) as src_handle, tgt_path.open(encoding="utf-8") as tgt_handle:
        for attempt_index, line in enumerate(ledger_handle):
            ledger = json.loads(line)
            source_index = attempt_index % len(reference_ids)
            augmentation_index = attempt_index // len(reference_ids)
            expected_id = reference_ids[source_index]
            if int(ledger["source_index"]) != source_index:
                raise ValueError("{} ledger source_index mismatch at {}".format(split, attempt_index))
            if int(ledger["augmentation_index"]) != augmentation_index:
                raise ValueError(
                    "{} ledger augmentation_index mismatch at {}".format(split, attempt_index)
                )
            if stable_id(ledger) != expected_id:
                raise ValueError("{} ledger stable-ID mismatch at {}".format(split, attempt_index))

            status = str(ledger.get("status") or "")
            status_counts[status] += 1
            if status == "emitted":
                metadata = next_json(map_handle, "{} line_map".format(split))
                if stable_id(metadata) != expected_id:
                    raise ValueError("{} line_map stable-ID mismatch".format(split))
                if int(metadata["line_index"]) != emitted:
                    raise ValueError("{} line_map line_index is not contiguous".format(split))
                if int(metadata["source_index"]) != source_index:
                    raise ValueError("{} line_map source_index mismatch".format(split))
                if int(metadata["augmentation_index"]) != augmentation_index:
                    raise ValueError("{} line_map augmentation mismatch".format(split))

                source = next_line(src_handle, "{} src".format(split))
                empty_sources += int(not source.strip())
                oov.update(token for token in source.split() if token not in vocabulary)
                if not evaluation_only:
                    target = next_line(tgt_handle, "{} tgt".format(split))
                    empty_targets += int(not target.strip())
                    oov.update(token for token in target.split() if token not in vocabulary)
                emitted += 1
            elif status == "conversion_error":
                failure = next_json(failure_handle, "{} failures".format(split))
                if failure != ledger:
                    raise ValueError("{} failure row does not match ledger".format(split))
                failed += 1
                failed_reaction_ids.add(expected_id)
            else:
                raise ValueError("{} unknown ledger status: {}".format(split, status))

        if attempt_index + 1 != expected_attempts:
            raise ValueError(
                "{} ledger has {} rows, expected {}".format(
                    split, attempt_index + 1, expected_attempts
                )
            )
        assert_exhausted(map_handle, "{} line_map".format(split))
        assert_exhausted(failure_handle, "{} failures".format(split))
        assert_exhausted(src_handle, "{} src".format(split))
        assert_exhausted(tgt_handle, "{} tgt".format(split))

    if emitted != int(native_report["emitted_rows"]):
        raise ValueError("{} emitted count disagrees with report".format(split))
    if failed != int(native_report["failed_attempts"]):
        raise ValueError("{} failure count disagrees with report".format(split))
    return {
        "input": str(input_path.resolve()),
        "input_rows": len(reference_ids),
        "augmentations": augmentations,
        "expected_attempts": expected_attempts,
        "ledger_rows": emitted + failed,
        "emitted_rows": emitted,
        "failed_attempts": failed,
        "conversion_coverage": emitted / float(max(expected_attempts, 1)),
        "status_counts": dict(sorted(status_counts.items())),
        "empty_sources": empty_sources,
        "empty_targets": empty_targets,
        "oov_token_occurrences": sum(oov.values()),
        "oov_tokens": dict(sorted(oov.items())),
        "evaluation_only": evaluation_only,
        "test_is_product_only": bool(native_report.get("test_is_product_only")),
        "evaluation_denominator_rows": len(reference_ids) if evaluation_only else None,
        "failed_reactions": len(failed_reaction_ids),
        "failures_counted_as_errors": evaluation_only,
        "reference_ids_filtered_from_evaluation": False if evaluation_only else None,
        "passed": empty_sources == 0 and empty_targets == 0 and not oov,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--conversion-source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    vocabulary = {
        line.strip().split()[0]
        for line in args.vocab.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    splits = {
        split: validate_split(
            split,
            args.input_root / "{}.jsonl".format(split),
            args.output_root / split,
            vocabulary,
        )
        for split in ("train", "valid", "test")
    }
    source_equivalence: Optional[Dict[str, Any]] = None
    if args.conversion_source_root is not None:
        source_equivalence = {
            split: validate_source_equivalence(
                args.input_root / "{}.jsonl".format(split),
                args.conversion_source_root / "{}.jsonl".format(split),
            )
            for split in ("train", "valid", "test")
        }
    report = {
        "artifact_type": "mechet_reactseq_full_dataset_validation",
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "vocab": str(args.vocab.resolve()),
        "vocab_size": len(vocabulary),
        "splits": splits,
        "source_equivalence": source_equivalence,
        "passed": all(value["passed"] for value in splits.values())
        and (
            source_equivalence is None
            or all(value["passed"] for value in source_equivalence.values())
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
