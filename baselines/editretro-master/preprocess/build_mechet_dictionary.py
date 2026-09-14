#!/usr/bin/env python3
"""Extend EditRetro's published dictionary using training tokens only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from rdkit import Chem, RDLogger


REACTION_COLUMN = "reactants>reagents>production"
SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)

RDLogger.DisableLog("rdApp.*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_base(path: Path) -> tuple[list[str], set[str]]:
    lines: list[str] = []
    tokens: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.rstrip("\n")
            fields = text.rsplit(" ", 1)
            if len(fields) != 2:
                raise ValueError(f"invalid dictionary row at {path}:{line_number}")
            token = fields[0]
            if token in tokens:
                raise ValueError(f"duplicate token in {path}: {token!r}")
            tokens.add(token)
            lines.append(text)
    return lines, tokens


def scan_raw_training_tokens(path: Path) -> tuple[Counter[str], int]:
    """Collect unmapped atom-level tokens from the complete frozen train split.

    The audit100 profile intentionally preprocesses only a prefix of training
    examples.  Its dictionary must nevertheless be valid for validation/test
    chemistry without inspecting either of those splits.  Scanning the full
    *training* CSV supplies that closed atom alphabet while retaining the IDs
    and frequencies of the published EditRetro dictionary.
    """
    counts: Counter[str] = Counter()
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or REACTION_COLUMN not in reader.fieldnames:
            raise ValueError(f"{path} is missing {REACTION_COLUMN!r}")
        for line_number, row in enumerate(reader, 2):
            reaction = str(row.get(REACTION_COLUMN) or "")
            parts = reaction.split(">")
            if len(parts) != 3:
                raise ValueError(f"invalid reaction at {path}:{line_number}")
            # Reagents are deliberately excluded, matching EditRetro's
            # product -> reactants training representation.
            for smiles in (parts[0], parts[2]):
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError(
                        f"RDKit could not parse {path}:{line_number}: {smiles}"
                    )
                for atom in molecule.GetAtoms():
                    atom.SetAtomMapNum(0)
                    if atom.HasProp("molAtomMapNumber"):
                        atom.ClearProp("molAtomMapNumber")
                unmapped = Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                )
                tokens = SMILES_TOKEN_PATTERN.findall(unmapped)
                if "".join(tokens) != unmapped:
                    raise ValueError(
                        f"SMILES tokenizer did not consume {path}:{line_number}: {smiles}"
                    )
                counts.update(tokens)
            rows += 1
    return counts, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dictionary", type=Path, required=True)
    parser.add_argument("--train-src", type=Path, required=True)
    parser.add_argument("--train-tgt", type=Path, required=True)
    parser.add_argument(
        "--raw-train-csv",
        type=Path,
        required=True,
        help="complete frozen training CSV; validation/test are never scanned",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (
        args.base_dictionary,
        args.train_src,
        args.train_tgt,
        args.raw_train_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")

    base_lines, base_tokens = read_base(args.base_dictionary)
    counts: Counter[str] = Counter()
    train_lines = 0
    for path in (args.train_src, args.train_tgt):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                train_lines += 1
                counts.update(line.strip().split())
    raw_counts, raw_train_rows = scan_raw_training_tokens(args.raw_train_csv)
    counts.update(raw_counts)
    additions = [
        (token, count)
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if token not in base_tokens
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for line in base_lines:
                handle.write(line + "\n")
            for token, count in additions:
                handle.write(f"{token} {count}\n")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    report = {
        "schema_version": 1,
        "artifact_type": "editretro_train_only_dictionary_extension",
        "base_dictionary": str(args.base_dictionary.resolve()),
        "base_dictionary_sha256": sha256_file(args.base_dictionary),
        "training_sources": [
            str(args.train_src.resolve()),
            str(args.train_tgt.resolve()),
            str(args.raw_train_csv.resolve()),
        ],
        "training_lines_scanned": train_lines,
        "complete_raw_training_rows_scanned": raw_train_rows,
        "complete_raw_training_token_types": len(raw_counts),
        "base_tokens": len(base_tokens),
        "added_tokens": len(additions),
        "additions": [
            {"token": token, "training_count": count} for token, count in additions
        ],
        "validation_or_test_tokens_used": False,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
