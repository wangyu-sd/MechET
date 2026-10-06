#!/usr/bin/env python3
"""Build full-denominator JevRetro single-step typed-decision targets.

Every input endpoint row is retained. The builder compiles a lossless endpoint
program and immediately reconstructs the structural precursor. Any failure is
recorded and the artifact is marked training_allowed=false; rows are never
silently filtered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.jevretro_endpoint import (
    attachment_templates,
    canonical_unmapped,
    derive_endpoint_program,
    program_summary,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def precursor(row: dict[str, Any]) -> str:
    value = str(
        row.get("structural_precursor")
        or row.get("expected_precursor")
        or ""
    ).strip()
    if not value:
        raise ValueError("row has no structural precursor")
    return value


def build_split(input_path: Path, output_path: Path, failure_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    successes = 0
    failures = 0
    programs: list[dict[str, Any]] = []
    histograms = {
        key: Counter()
        for key in (
            "atom_edits",
            "bond_edits",
            "delete_product_atoms",
            "attachments",
            "attachment_slots",
        )
    }
    max_values = {key: 0 for key in histograms}
    attachment_rows = 0

    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as out, failure_path.open("w", encoding="utf-8") as failed:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {input_path}:{line_number}")
            rows += 1
            row = json.loads(line)
            identifier = str(row.get("id") or f"row-{line_number}")
            product = str(row.get("target_smiles") or "").strip()
            try:
                if not product:
                    raise ValueError("row has no target_smiles")
                gold = precursor(row)
                program = derive_endpoint_program(product, gold)
                summary = program_summary(program)
                for key, value in summary.items():
                    histograms[key][str(value)] += 1
                    max_values[key] = max(max_values[key], value)
                attachment_rows += int(bool(program.get("attachments")))
                compiled = {
                    "id": identifier,
                    "source_id": row.get("source_id"),
                    "artifact_type": row.get("artifact_type"),
                    "task_type": "jevretro_single_step_typed_endpoint",
                    "target_smiles": product,
                    "structural_precursor": gold,
                    "canonical_structural_precursor": canonical_unmapped(gold),
                    "program": program,
                    "program_summary": summary,
                    "source_metadata": row.get("metadata") or {},
                }
                out.write(json.dumps(compiled, ensure_ascii=False) + "\n")
                programs.append(program)
                successes += 1
            except Exception as exc:
                failures += 1
                failed.write(
                    json.dumps(
                        {
                            "id": identifier,
                            "line_number": line_number,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return {
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "failures": str(failure_path.resolve()),
        "failures_sha256": sha256(failure_path),
        "rows": rows,
        "compiled_rows": successes,
        "failed_rows": failures,
        "coverage": successes / rows if rows else 0.0,
        "attachment_rows": attachment_rows,
        "attachment_template_vocab": sorted(attachment_templates(programs)),
        "histograms": {
            key: dict(sorted(value.items(), key=lambda item: int(item[0])))
            for key, value in histograms.items()
        },
        "maxima": max_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, path in (
        ("train", args.train),
        ("valid", args.valid),
        ("test", args.test),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        splits[split] = build_split(
            path,
            args.output_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.failures.jsonl",
        )

    train_vocab = set(splits["train"]["attachment_template_vocab"])
    for split in ("valid", "test"):
        path = args.output_dir / f"{split}.jsonl"
        oov_rows = 0
        oov_occurrences = 0
        oov_templates: Counter[str] = Counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                current = [
                    str(item["template"])
                    for item in row["program"].get("attachments") or []
                ]
                unseen = [template for template in current if template not in train_vocab]
                oov_rows += int(bool(unseen))
                oov_occurrences += len(unseen)
                oov_templates.update(unseen)
        splits[split]["attachment_oov_rows_vs_train"] = oov_rows
        splits[split]["attachment_oov_occurrences_vs_train"] = oov_occurrences
        splits[split]["attachment_oov_unique_vs_train"] = len(oov_templates)
        splits[split]["attachment_oov_top20_vs_train"] = oov_templates.most_common(20)

    for split in splits:
        splits[split]["attachment_template_vocab_size"] = len(
            splits[split].pop("attachment_template_vocab")
        )

    failed_total = sum(item["failed_rows"] for item in splits.values())
    manifest = {
        "schema_version": 1,
        "artifact_type": "jevretro_full_endpoint_typed_decisions_v1",
        "task": "single_step_retrosynthesis",
        "input_contract": "mapped_product_only",
        "target_contract": (
            "lossless atom-state/bond-state edits plus residual attachment templates"
        ),
        "mechanistic_trace_required": False,
        "row_filtering_allowed": False,
        "training_allowed": failed_total == 0,
        "failed_rows_total": failed_total,
        "splits": splits,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if failed_total:
        print(
            f"JevRetro endpoint compilation failed for {failed_total} rows; "
            "artifact is not trainable",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
