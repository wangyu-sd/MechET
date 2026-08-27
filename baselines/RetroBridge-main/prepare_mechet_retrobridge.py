#!/usr/bin/env python3
"""Convert frozen MechET reaction pairs to RetroBridge's native CSV format.

The generated metadata is derived from the selected training records only.  It
defines the atom vocabulary and the number of extra (dummy) nodes used by the
small-data RetroBridge adaptation.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm


PUBLISHED_ATOM_ORDER = [
    "C", "O", "N", "Cl", "F", "S", "Na", "Br", "K", "P", "H", "I", "B", "Li", "Si", "Pd",
    "Cs", "Al", "Cu", "Mg", "Sn", "Zn", "Fe", "Cr", "Mn", "Ti", "Pt", "Ca", "Ag", "Se", "Ni",
    "Ru", "Rh", "Co", "Os", "Ce", "Pb", "Ba", "Hg", "Zr", "As", "Yb", "W", "Bi", "Ge", "In",
    "Sb", "Sc", "Tl", "Mo", "Sm", "Re", "Ir", "Au", "Cd", "Ga", "Xe", "Nd", "Ta", "V", "La",
    "Rb", "Dy", "Hf", "Y", "Te", "Ar", "Pr", "He", "Be", "Eu", "Sr",
]
SUPPORTED_BONDS = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}
SPLIT_TO_CSV = {
    "train": "uspto50k_train.csv",
    "valid": "uspto50k_val.csv",
    "test": "uspto50k_test.csv",
}


def read_jsonl(path, limit=None):
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def mapped_atoms(molecule):
    return {atom.GetAtomMapNum(): atom.GetSymbol() for atom in molecule.GetAtoms()}


def inspect_row(row):
    reactant = Chem.MolFromSmiles(row["precursor_mapped"])
    product = Chem.MolFromSmiles(row["product_mapped"])
    errors = []
    if reactant is None:
        errors.append("invalid_precursor_mapped")
    if product is None:
        errors.append("invalid_product_mapped")
    if errors:
        return {"errors": errors}

    reactant_maps = [atom.GetAtomMapNum() for atom in reactant.GetAtoms()]
    product_maps = [atom.GetAtomMapNum() for atom in product.GetAtoms()]
    if 0 in reactant_maps or len(reactant_maps) != len(set(reactant_maps)):
        errors.append("precursor_atom_maps_not_unique_positive")
    if 0 in product_maps or len(product_maps) != len(set(product_maps)):
        errors.append("product_atom_maps_not_unique_positive")

    reactant_by_map = mapped_atoms(reactant)
    product_by_map = mapped_atoms(product)
    if not set(product_by_map).issubset(reactant_by_map):
        errors.append("product_maps_not_subset_of_precursor")
    elif any(reactant_by_map[key] != value for key, value in product_by_map.items()):
        errors.append("mapped_atom_element_mismatch")

    atom_delta = reactant.GetNumAtoms() - product.GetNumAtoms()
    if atom_delta < 0:
        errors.append("product_has_more_atoms_than_precursor")

    unsupported_bonds = sorted({
        str(bond.GetBondType())
        for molecule in (reactant, product)
        for bond in molecule.GetBonds()
        if bond.GetBondType() not in SUPPORTED_BONDS
    })
    if unsupported_bonds:
        errors.append("unsupported_bond_types:" + ",".join(unsupported_bonds))

    return {
        "errors": errors,
        "atom_delta": atom_delta,
        "reactant_atoms": reactant.GetNumAtoms(),
        "product_atoms": product.GetNumAtoms(),
        "elements": sorted({atom.GetSymbol() for atom in reactant.GetAtoms()} |
                           {atom.GetSymbol() for atom in product.GetAtoms()}),
    }


def sha256_lines(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["stable_id"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def analyze_rows(rows, split):
    element_counts = Counter()
    atom_delta_counts = Counter()
    failures = []
    max_delta = 0
    max_nodes = 0
    for row in tqdm(rows, desc=f"audit {split}", unit="rxn"):
        result = inspect_row(row)
        if result["errors"]:
            failures.append({
                "stable_id": row.get("stable_id"),
                "source_line": row["_source_line"],
                "errors": result["errors"],
            })
            continue
        element_counts.update(result["elements"])
        atom_delta_counts[result["atom_delta"]] += 1
        max_delta = max(max_delta, result["atom_delta"])
        max_nodes = max(max_nodes, result["reactant_atoms"])

    return {
        "split": split,
        "count": len(rows),
        "stable_id_sha256": sha256_lines(rows),
        "elements": sorted(element_counts),
        "element_reaction_counts": dict(sorted(element_counts.items())),
        "atom_delta_counts": {str(key): value for key, value in sorted(atom_delta_counts.items())},
        "max_atom_delta": max_delta,
        "max_reactant_atoms": max_nodes,
        "failure_count": len(failures),
        "failures": failures,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "class", "reactants>reagents>production"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["stable_id"],
                "class": row.get("reaction_class", ""),
                "reactants>reagents>production": (
                    f'{row["precursor_mapped"]}>>{row["product_mapped"]}'
                ),
            })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument(
        "--atom-vocabulary",
        choices=["train", "published-mit"],
        default="train",
        help=(
            "Derive atom types from train, or use RetroBridge's published fixed "
            "USPTO-MIT vocabulary (needed when evaluation contains train-unseen elements)."
        ),
    )
    parser.add_argument(
        "--overfit-size",
        type=int,
        help="Use the first N train rows for all three splits (diagnostic only).",
    )
    parser.add_argument(
        "--drop-incompatible",
        action="store_true",
        help=(
            "Exclude reactions that fail the RetroBridge compatibility audit from "
            "the converted output. The source JSONL files are never modified, and "
            "all exclusions are recorded in metadata.json."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    if args.overfit_size is not None:
        train_rows = read_jsonl(args.source_dir / "train.jsonl", args.overfit_size)
        split_rows = {"train": train_rows, "valid": train_rows, "test": train_rows}
        diagnostic_duplicate_splits = True
    else:
        split_rows = {
            "train": read_jsonl(args.source_dir / "train.jsonl", args.train_limit),
            "valid": read_jsonl(args.source_dir / "valid.jsonl", args.valid_limit),
            "test": read_jsonl(args.source_dir / "test.jsonl", args.test_limit),
        }
        train_rows = split_rows["train"]
        diagnostic_duplicate_splits = False

    source_audits = {split: analyze_rows(rows, split) for split, rows in split_rows.items()}
    all_failures = [
        failure
        for audit in source_audits.values()
        for failure in audit["failures"]
    ]
    if all_failures and not args.drop_incompatible:
        preview = json.dumps(all_failures[:10], indent=2)
        raise ValueError(
            f"Input contains {len(all_failures)} incompatible reactions:\n{preview}\n"
            "Re-run with --drop-incompatible only if changing the split "
            "denominators is explicitly intended."
        )

    excluded_by_split = {
        split: list(audit["failures"])
        for split, audit in source_audits.items()
    }
    if args.drop_incompatible:
        filtered_split_rows = {}
        audits = {}
        for split, rows in split_rows.items():
            excluded_source_lines = {
                failure["source_line"] for failure in excluded_by_split[split]
            }
            filtered_rows = [
                row for row in rows if row["_source_line"] not in excluded_source_lines
            ]
            filtered_split_rows[split] = filtered_rows

            # analyze_rows already computes chemistry statistics from compatible
            # rows only. Adjust its provenance fields to describe the rows that
            # will actually be written without repeating the full RDKit scan.
            audit = dict(source_audits[split])
            audit["input_count"] = audit["count"]
            audit["count"] = len(filtered_rows)
            audit["stable_id_sha256"] = sha256_lines(filtered_rows)
            audit["excluded_incompatible_count"] = len(excluded_by_split[split])
            audit["failure_count"] = 0
            audit["failures"] = []
            audits[split] = audit
        split_rows = filtered_split_rows
        train_rows = split_rows["train"]
    else:
        audits = source_audits

    train_elements = set(audits["train"]["elements"])
    if args.atom_vocabulary == "published-mit":
        atom_decoder = list(PUBLISHED_ATOM_ORDER)
        allowed_elements = set(PUBLISHED_ATOM_ORDER)
        vocabulary_source_splits = []
        vocabulary_source = "published_RetroBridgeMITDataset.types"
    else:
        atom_decoder = [element for element in PUBLISHED_ATOM_ORDER if element in train_elements]
        atom_decoder.extend(sorted(train_elements - set(PUBLISHED_ATOM_ORDER)))
        allowed_elements = train_elements
        vocabulary_source_splits = ["train"]
        vocabulary_source = "selected_train_rows"
    atom_decoder.append("*")

    unseen_by_split = {
        split: sorted(set(audit["elements"]) - allowed_elements)
        for split, audit in audits.items()
    }
    excessive_delta_by_split = {
        split: max(0, audit["max_atom_delta"] - audits["train"]["max_atom_delta"])
        for split, audit in audits.items()
    }
    incompatibilities = {
        split: {
            "unseen_elements": unseen_by_split[split],
            "atom_delta_above_train_max": excessive_delta_by_split[split],
        }
        for split in split_rows
        if unseen_by_split[split] or excessive_delta_by_split[split]
    }
    if incompatibilities:
        raise ValueError(
            "Validation/test rows do not fit the train-derived graph space: "
            + json.dumps(incompatibilities, indent=2)
        )

    raw_dir = args.output_dir / "raw"
    for split, rows in split_rows.items():
        write_csv(raw_dir / SPLIT_TO_CSV[split], rows)

    metadata = {
        "format": "mechet-retrobridge-v1",
        "strict": True,
        "atom_decoder": atom_decoder,
        "max_n_dummy_nodes": audits["train"]["max_atom_delta"],
        "vocabulary_source": vocabulary_source,
        "vocabulary_source_splits": vocabulary_source_splits,
        "capacity_source_splits": ["train"],
        "source_dir": str(args.source_dir.resolve()),
        "diagnostic_duplicate_splits": diagnostic_duplicate_splits,
        "source_filtering": {
            "mode": "drop_incompatible" if args.drop_incompatible else "strict",
            "source_files_modified": False,
            "input_counts": {
                split: audit["count"] for split, audit in source_audits.items()
            },
            "output_counts": {
                split: len(rows) for split, rows in split_rows.items()
            },
            "excluded_incompatible_count": len(all_failures) if args.drop_incompatible else 0,
            "excluded_by_split": excluded_by_split if args.drop_incompatible else {},
            "changes_evaluation_denominator": bool(
                args.drop_incompatible and excluded_by_split.get("test")
            ),
        },
        "splits": audits,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "excluded_incompatible": (
            {split: len(failures) for split, failures in excluded_by_split.items()}
            if args.drop_incompatible else {}
        ),
        "atom_decoder": atom_decoder,
        "max_n_dummy_nodes": metadata["max_n_dummy_nodes"],
    }, indent=2))


if __name__ == "__main__":
    main()
