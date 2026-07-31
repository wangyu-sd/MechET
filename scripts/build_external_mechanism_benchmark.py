#!/usr/bin/env python3
"""Normalize external reactions/mechanisms into an auditable benchmark JSONL."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.data_audit import split_reaction_smiles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, input_format: str) -> list[dict]:
    if input_format == "jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-format", choices=["jsonl", "csv"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--reaction-field", default="reaction_smiles")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--date-field", default="publication_date")
    parser.add_argument("--patent-field", default="patent_id")
    parser.add_argument("--reference-proof-field", default="reference_proof")
    parser.add_argument("--cutoff-date", default="")
    parser.add_argument("--require-post-cutoff", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input, args.input_format)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = missing_date = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            reaction = str(row.get(args.reaction_field) or "")
            try:
                reactants, reagents, products = split_reaction_smiles(reaction)
            except Exception:
                skipped += 1
                continue
            date = str(row.get(args.date_field) or "")
            if not date:
                missing_date += 1
            if args.require_post_cutoff and args.cutoff_date:
                if not date or date <= args.cutoff_date:
                    skipped += 1
                    continue
            payload = {
                "id": str(row.get(args.id_field) or f"{args.source_name}:{index}"),
                "source": args.source_name,
                "reaction_smiles": reaction,
                "reactants": reactants,
                "reagents": reagents,
                "product": products,
                "publication_date": date,
                "patent_id": str(row.get(args.patent_field) or ""),
                "reference_proof": str(row.get(args.reference_proof_field) or ""),
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {
                        args.reaction_field,
                        args.id_field,
                        args.date_field,
                        args.patent_field,
                        args.reference_proof_field,
                    }
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    manifest = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "source_name": args.source_name,
        "n_input": len(rows),
        "n_written": written,
        "n_skipped": skipped,
        "n_missing_publication_date": missing_date,
        "cutoff_date": args.cutoff_date,
        "require_post_cutoff": args.require_post_cutoff,
        "warning": (
            "Rows without verified dates cannot support temporal-disjoint claims"
            if missing_date
            else ""
        ),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
