#!/usr/bin/env python3
"""Prepare frozen MechET reaction pairs with EditRetro's native representation.

This adapter deliberately keeps the published EditRetro representation:

* atom-mapped reactant/product pairs are used only to construct root-aligned
  SMILES;
* atom maps are removed before tokenization;
* the published ChEMBL SPE vocabulary is used by default;
* oracle reposition/insertion targets remain generated inside ``editretro_nat``
  during training and are not replaced by a generic sequence target.

The official preprocessor silently drops incompatible rows and loses reaction
identifiers.  This wrapper makes every attempt auditable and writes a line map
for restoring ``stable_id`` after Fairseq binarization and generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import statistics
from typing import Any, Iterable, Iterator

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIR.parent
REACTION_COLUMN = "reactants>reagents>production"
MAP_PATTERN = re.compile(r":(\d+)]")
TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)

_SPE_TOKENIZER: Any = None


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


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    first = True
    for value in values:
        if not first:
            digest.update(b"\n")
        digest.update(value.encode("utf-8"))
        first = False
    return digest.hexdigest()


def stable_seed(base_seed: int, stable_id: str, split: str) -> int:
    payload = f"{base_seed}\n{split}\n{stable_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def get_spe_tokenizer() -> Any:
    global _SPE_TOKENIZER
    if _SPE_TOKENIZER is None:
        try:
            from SmilesPE.tokenizer import SPE_Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "SmilesPE is required for --tokenization spe; install "
                "requirements.txt in the EditRetro Python 3.10 environment"
            ) from exc
        vocabulary = (SCRIPT_DIR / "SPE_ChEMBL.txt").open(encoding="utf-8")
        _SPE_TOKENIZER = SPE_Tokenizer(vocabulary, merges=-1)
    return _SPE_TOKENIZER


def tokenize_smiles(smiles: str, mode: str, dropout: float) -> str:
    if mode == "spe":
        return str(get_spe_tokenizer().tokenize(smiles, dropout=dropout))
    if mode == "selfies":
        try:
            import selfies as sf
        except ImportError as exc:
            raise RuntimeError("selfies is required for --tokenization selfies") from exc
        return " ".join(sf.split_selfies(sf.encoder(smiles)))
    tokens = TOKEN_PATTERN.findall(smiles)
    if smiles != "".join(tokens):
        raise ValueError(f"SMILES tokenizer did not consume input: {smiles}")
    return " ".join(tokens)


def clear_map_canonical_smiles(
    smiles: str, *, canonical: bool = True, root: int = -1
) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(
        molecule,
        isomericSmiles=True,
        rootedAtAtom=root,
        canonical=canonical,
    )


def canonical_key(smiles: str) -> str:
    fragments: list[str] = []
    for fragment in smiles.split("."):
        fragment = fragment.strip()
        if not fragment:
            raise ValueError(f"empty fragment in SMILES: {smiles}")
        fragments.append(clear_map_canonical_smiles(fragment))
    return ".".join(sorted(fragments))


def get_canonical_map_numbers(smiles: str, root: int = -1) -> list[int] | None:
    atom_mapped = Chem.MolFromSmiles(smiles)
    if atom_mapped is None:
        return None
    canonical = Chem.MolFromSmiles(clear_map_canonical_smiles(smiles, root=root))
    if canonical is None:
        return None
    canonical_to_mapped = atom_mapped.GetSubstructMatch(canonical)
    correct = [
        canonical.GetAtomWithIdx(index).GetSymbol()
        == atom_mapped.GetAtomWithIdx(mapped_index).GetSymbol()
        for index, mapped_index in enumerate(canonical_to_mapped)
    ]
    atom_count = canonical.GetNumAtoms()
    if sum(correct) < atom_count or len(canonical_to_mapped) < atom_count:
        inverse = canonical.GetSubstructMatch(atom_mapped)
        if len(inverse) != atom_count:
            return None
        canonical_to_mapped = [0] * atom_count
        for mapped_index, canonical_index in enumerate(inverse):
            canonical_to_mapped[canonical_index] = mapped_index
    mapped_numbers = [atom.GetAtomMapNum() for atom in atom_mapped.GetAtoms()]
    return [mapped_numbers[canonical_to_mapped[index]] for index in range(atom_count)]


def get_root_id(molecule: Chem.Mol, root_map_number: int) -> int:
    if root_map_number == -1:
        return -1
    for index, atom in enumerate(molecule.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            return index
    return -1


def select_product_roots(
    product_map_numbers: list[int], augmentation: int, rng: random.Random
) -> list[int]:
    if augmentation == 999:
        return list(product_map_numbers)
    if augmentation <= 0:
        raise ValueError("augmentation must be positive or 999")
    roots = [-1]
    target_count = min(augmentation, len(product_map_numbers))
    if target_count < augmentation:
        roots.extend(product_map_numbers)
        roots.extend(rng.choices(roots, k=augmentation - len(roots)))
    else:
        while len(roots) < target_count:
            root = rng.choice(product_map_numbers)
            if root not in roots:
                roots.append(root)
    if len(roots) != augmentation:
        raise AssertionError(
            f"root augmentation produced {len(roots)} views, expected {augmentation}"
        )
    return roots


def parse_reaction(value: str) -> tuple[str, str]:
    parts = value.strip().split(">")
    if len(parts) != 3:
        raise ValueError("reaction must have reactants>reagents>product form")
    reactants = parts[0].split(" ", 1)[0].strip()
    product = parts[2].split(" ", 1)[0].strip()
    return reactants, product


def validate_pair(reactants: str, product: str) -> tuple[str, list[int]]:
    if not product:
        return "empty_p", []
    if not reactants:
        return "empty_r", []
    reactant_molecule = Chem.MolFromSmiles(reactants)
    if reactant_molecule is None:
        return "invalid_r", []
    product_molecule = Chem.MolFromSmiles(product)
    if product_molecule is None:
        return "invalid_p", []
    reactant_maps = MAP_PATTERN.findall(reactants)
    product_maps = MAP_PATTERN.findall(product)
    if len(set(reactant_maps)) != len(reactant_maps):
        return "duplicate_reactant_mapping", []
    if len(set(product_maps)) != len(product_maps):
        return "duplicate_product_mapping", []
    if not all(atom.GetAtomMapNum() > 0 for atom in product_molecule.GetAtoms()):
        return "error_mapping_p", []
    return "ok", [int(value) for value in product_maps]


def process_row(task: dict[str, Any]) -> dict[str, Any]:
    stable_id = str(task["stable_id"])
    result: dict[str, Any] = {
        "status": "ok",
        "stable_id": stable_id,
        "source_row_index": int(task["source_row_index"]),
        "views": [],
        "edit_distances": [],
    }
    try:
        reactants, product = parse_reaction(str(task["reaction"]))
        if "." in product:
            raise ValueError("multiple_product")
        status, product_map_numbers = validate_pair(reactants, product)
        if status != "ok":
            result["status"] = status
            return result

        product_molecule = Chem.MolFromSmiles(product)
        assert product_molecule is not None
        fragments = reactants.split(".")
        fragment_molecules = [Chem.MolFromSmiles(fragment) for fragment in fragments]
        if any(molecule is None for molecule in fragment_molecules):
            result["status"] = "invalid_r"
            return result
        fragment_maps = [
            [int(value) for value in MAP_PATTERN.findall(fragment)]
            for fragment in fragments
        ]
        expected_product_key = canonical_key(product)
        # The common handoff has already separated structural precursors from
        # auxiliary/reagent fragments.  Preserve every structural fragment,
        # including rare upstream mapping anomalies with no product-map overlap.
        expected_precursor_key = canonical_key(reactants)

        product_map_set = set(product_map_numbers)
        unaligned_fragment_indices = [
            index
            for index, maps in enumerate(fragment_maps)
            if not product_map_set.intersection(maps)
        ]
        result["unaligned_precursor_fragment_count"] = len(
            unaligned_fragment_indices
        )

        rng = random.Random(
            stable_seed(int(task["seed"]), stable_id, str(task["split"]))
        )
        roots = select_product_roots(
            product_map_numbers, int(task["augmentation"]), rng
        )
        for augmentation_index, root_map_number in enumerate(roots):
            product_root = get_root_id(product_molecule, root_map_number)
            canonical_maps = get_canonical_map_numbers(product, root=product_root)
            if canonical_maps is None:
                result["status"] = "error_mapping"
                result["views"] = []
                return result
            product_smiles = clear_map_canonical_smiles(product, root=product_root)

            aligned: list[tuple[str, int]] = []
            aligned_indices: set[int] = set()
            for fragment_index, (fragment, molecule, maps) in enumerate(zip(
                fragments, fragment_molecules, fragment_maps
            )):
                for product_position, map_number in enumerate(canonical_maps):
                    if map_number in maps:
                        assert molecule is not None
                        fragment_root = get_root_id(molecule, map_number)
                        aligned.append(
                            (
                                clear_map_canonical_smiles(
                                    fragment, root=fragment_root
                                ),
                                product_position,
                            )
                        )
                        aligned_indices.add(fragment_index)
                        break
            # Official EditRetro discards reactant fragments without a shared
            # product map because its raw inputs conflate reagents and
            # reactants.  MechET's handoff already removed auxiliaries, so an
            # unaligned fragment here is a structural mapping anomaly and must
            # remain in the target/denominator.  Normal rows never take this
            # fallback.
            for fragment_index, (fragment, molecule) in enumerate(
                zip(fragments, fragment_molecules)
            ):
                if fragment_index not in aligned_indices:
                    assert molecule is not None
                    aligned.append(
                        (
                            clear_map_canonical_smiles(fragment),
                            len(canonical_maps) + fragment_index,
                        )
                    )
            aligned.sort(key=lambda item: item[1])
            precursor_parts = [item[0] for item in aligned]
            if task["shuffle"]:
                rng.shuffle(precursor_parts)
            precursor_smiles = ".".join(precursor_parts)
            if not precursor_smiles:
                result["status"] = "empty_native_precursor"
                result["views"] = []
                return result

            source_key = canonical_key(product_smiles)
            target_key = canonical_key(precursor_smiles)
            if source_key != expected_product_key:
                result["status"] = "product_identity_mismatch"
                result["views"] = []
                return result
            if target_key != expected_precursor_key:
                result["status"] = "precursor_identity_mismatch"
                result["views"] = []
                return result

            source = tokenize_smiles(
                product_smiles, str(task["tokenization"]), float(task["dropout"])
            )
            target = tokenize_smiles(
                precursor_smiles,
                str(task["tokenization"]),
                float(task["dropout"]),
            )
            if task["character"]:
                source = " ".join("".join(source.split()))
                target = " ".join("".join(target.split()))
            distance = levenshtein_distance(source.split(), target.split())
            result["edit_distances"].append(distance)
            result["views"].append(
                {
                    "stable_id": stable_id,
                    "source_row_index": int(task["source_row_index"]),
                    "augmentation_index": augmentation_index,
                    "root_map_number": root_map_number,
                    "source": source,
                    "target": target,
                    "product_canonical": source_key,
                    "precursor_canonical": target_key,
                    "unaligned_precursor_fragment_count": len(
                        unaligned_fragment_indices
                    ),
                }
            )
    except Exception as exc:  # preserve a row-level failure ledger
        result["status"] = str(exc) or type(exc).__name__
        result["error_type"] = type(exc).__name__
        result["views"] = []
    return result


def levenshtein_distance(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_token in enumerate(right, 1):
        current = [right_index]
        for left_index, left_token in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def load_source_index(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    indexed: dict[int, str] = {}
    for row in read_jsonl(path):
        index = int(row["row_index"])
        identifier = str(row["stable_id"])
        if index in indexed:
            raise ValueError(f"duplicate row_index {index} in {path}")
        indexed[index] = identifier
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError(f"non-contiguous row_index values in {path}")
    return [indexed[index] for index in range(len(indexed))]


def load_tasks(args: argparse.Namespace, split: str) -> tuple[list[dict[str, Any]], Path]:
    raw_path = args.data_dir / f"raw_{split}.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    tasks: list[dict[str, Any]] = []
    with raw_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or REACTION_COLUMN not in reader.fieldnames:
            raise ValueError(f"{raw_path} is missing {REACTION_COLUMN!r}")
        seen: set[str] = set()
        for row_index, row in enumerate(reader):
            identifier = str(row.get("stable_id") or row.get("id") or "").strip()
            if not identifier:
                raise ValueError(f"{raw_path}:{row_index + 2} is missing stable_id")
            if identifier in seen:
                raise ValueError(f"duplicate stable_id in {raw_path}: {identifier}")
            seen.add(identifier)
            tasks.append(
                {
                    "stable_id": identifier,
                    "source_row_index": row_index,
                    "reaction": str(row.get(REACTION_COLUMN) or ""),
                    "split": split,
                    "augmentation": args.augmentation,
                    "seed": args.seed,
                    "tokenization": args.tokenization,
                    "dropout": args.dropout,
                    "shuffle": args.shuffle,
                    "character": args.character,
                }
            )
    source_index = load_source_index(args.data_dir / f"{split}_index.jsonl")
    task_ids = [str(task["stable_id"]) for task in tasks]
    if source_index is not None and source_index != task_ids:
        raise ValueError(f"raw CSV and source index disagree for split {split}")
    return tasks, raw_path


def output_paths(output_dir: Path, split: str) -> dict[str, Path]:
    return {
        "src": output_dir / f"{split}.src",
        "tgt": output_dir / f"{split}.tgt",
        "line_map": output_dir / f"{split}.line_map.jsonl",
        "failures": output_dir / f"{split}.failures.jsonl",
        "report": output_dir / f"{split}.preprocess_report.json",
    }


def ensure_writable(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {existing[0]}; pass --overwrite"
        )


def temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def prepare_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    tasks, raw_path = load_tasks(args, split)
    paths = output_paths(args.output_dir, split)
    ensure_writable(paths.values(), args.overwrite)
    temp = {name: temporary_path(path) for name, path in paths.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    distances: list[int] = []
    rows_with_unaligned_fragments = 0
    unaligned_fragment_count = 0
    emitted = 0
    emitted_ids: list[str] = []
    results: Iterable[dict[str, Any]]
    pool: mp.pool.Pool | None = None
    try:
        if args.processes == 1:
            results = map(process_row, tasks)
        else:
            processes = mp.cpu_count() if args.processes < 0 else args.processes
            pool = mp.Pool(processes)
            results = pool.imap(process_row, tasks, chunksize=args.chunksize)

        with temp["src"].open("w", encoding="utf-8") as source_handle, temp[
            "tgt"
        ].open("w", encoding="utf-8") as target_handle, temp["line_map"].open(
            "w", encoding="utf-8"
        ) as map_handle, temp["failures"].open("w", encoding="utf-8") as failure_handle:
            for result in results:
                status = str(result["status"])
                status_counts[status] += 1
                if status != "ok":
                    failure = {
                        key: value
                        for key, value in result.items()
                        if key not in {"views", "edit_distances"}
                    }
                    failures.append(failure)
                    failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    continue
                views = list(result["views"])
                if len(views) != args.augmentation:
                    raise ValueError(
                        f"{result['stable_id']} emitted {len(views)} views, "
                        f"expected {args.augmentation}"
                    )
                distances.extend(int(value) for value in result["edit_distances"])
                row_unaligned = int(
                    result.get("unaligned_precursor_fragment_count", 0)
                )
                if row_unaligned:
                    rows_with_unaligned_fragments += 1
                    unaligned_fragment_count += row_unaligned
                for view in views:
                    source_handle.write(str(view.pop("source")) + "\n")
                    target_handle.write(str(view.pop("target")) + "\n")
                    view["line_index"] = emitted
                    map_handle.write(json.dumps(view, ensure_ascii=False) + "\n")
                    emitted_ids.append(str(view["stable_id"]))
                    emitted += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    expected_views = len(tasks) * args.augmentation
    report = {
        "schema_version": 1,
        "artifact_type": "editretro_native_preprocessing",
        "dataset": args.dataset,
        "split": split,
        "raw_input": str(raw_path.resolve()),
        "raw_input_sha256": sha256_file(raw_path),
        "input_rows": len(tasks),
        "input_stable_ids_sha256": sha256_lines(
            str(task["stable_id"]) for task in tasks
        ),
        "augmentation": args.augmentation,
        "tokenization": args.tokenization,
        "root_alignment": True,
        "seed": args.seed,
        "expected_views": expected_views,
        "emitted_views": emitted,
        "emitted_stable_ids_sha256": sha256_lines(emitted_ids),
        "failed_rows": len(failures),
        "rows_with_unaligned_precursor_fragments": rows_with_unaligned_fragments,
        "unaligned_precursor_fragments": unaligned_fragment_count,
        "status_counts": dict(sorted(status_counts.items())),
        "mean_token_levenshtein_distance": (
            statistics.fmean(distances) if distances else None
        ),
        "complete_coverage": emitted == expected_views and not failures,
        "oracle_edit_targets": "generated_dynamically_by_editretro_nat",
        "mechanism_supervision": False,
        "resplit": False,
    }
    temp["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, path in paths.items():
        os.replace(temp[name], path)
    if failures and not args.allow_failures:
        raise RuntimeError(
            f"{split}: {len(failures)} rows failed native preprocessing; "
            f"see {paths['failures']}"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test")
    )
    parser.add_argument("--augmentation", type=int, default=10)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--processes", type=int, default=-1)
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument(
        "--tokenization", choices=("spe", "token", "selfies"), default="spe"
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--character", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.augmentation <= 0 or args.augmentation == 999:
        parser.error("MechET preprocessing requires a fixed positive augmentation")
    if args.processes == 0:
        parser.error("--processes cannot be zero")
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    args.data_dir = (
        args.data_dir.resolve()
        if args.data_dir
        else (REPOSITORY / "datasets" / args.dataset / "raw").resolve()
    )
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            REPOSITORY / "datasets" / args.dataset / f"aug{args.augmentation}"
        ).resolve()
    )
    reports = [prepare_split(args, split) for split in args.splits]
    summary = {
        "dataset": args.dataset,
        "output_dir": str(args.output_dir),
        "splits": {str(report["split"]): report for report in reports},
    }
    summary_path = args.output_dir / "preprocess_manifest.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
