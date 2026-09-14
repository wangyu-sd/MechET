#!/usr/bin/env python3
"""Prepare frozen MechET endpoint JSONL files for native ReactSeq training.

The script deliberately keeps atom mapping separate from ReactSeq conversion:

* ``map`` maps unmapped precursor>>product pairs with the official RXNMapper;
* ``prepare`` performs ReactSeq's native random-SMILES/kekulization conversion.

Every source reaction is represented in a ledger.  Supervised rows that
ReactSeq cannot encode are omitted from src/tgt but remain explicit in the
ledger.  Validation/test evaluation inputs can be generated from the product
alone, preserving the complete reference denominator without exposing the
reference precursor.  Any missing or failed evaluation prediction is scored as
an error downstream rather than filtering the reference row.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


_PREPARE_ROWS: List[Dict[str, Any]] = []


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} is not a JSON object".format(path, line_number))
            rows.append(value)
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def stable_id(row: Dict[str, Any]) -> str:
    value = str(row.get("id") or row.get("stable_id") or "").strip()
    if not value:
        raise ValueError("row is missing id/stable_id")
    return value


def product_smiles(row: Dict[str, Any]) -> str:
    return str(
        row.get("product_mapped")
        or row.get("target_smiles")
        or row.get("product_unmapped")
        or row.get("product")
        or ""
    ).strip()


def precursor_smiles(row: Dict[str, Any]) -> str:
    return str(
        row.get("precursor_mapped")
        or row.get("structural_precursor")
        or row.get("expected_precursor")
        or row.get("precursor_unmapped")
        or row.get("reactants")
        or ""
    ).strip()


def reaction_smiles(row: Dict[str, Any]) -> str:
    precursor = precursor_smiles(row)
    product = product_smiles(row)
    if precursor and product:
        return precursor + ">>" + product
    value = str(row.get("reaction_mapped") or "").strip()
    if ">>" in value:
        return value
    raise ValueError("{} is missing precursor/product".format(stable_id(row)))


def unmapped_reaction_smiles(row: Dict[str, Any]) -> str:
    value = str(row.get("reaction_unmapped") or "").strip()
    if value:
        return value
    precursor = str(
        row.get("precursor_unmapped")
        or row.get("structural_precursor")
        or row.get("expected_precursor")
        or row.get("precursor_mapped")
        or ""
    ).strip()
    product = str(
        row.get("product_unmapped")
        or row.get("target_smiles")
        or row.get("product_mapped")
        or ""
    ).strip()
    if not precursor or not product:
        raise ValueError("{} is missing precursor/product".format(stable_id(row)))
    return precursor + ">>" + product


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _validate_mapped_reaction(mapped: str, original: str) -> None:
    from rdkit import Chem

    mapped_precursor, mapped_product = mapped.split(">>", 1)
    original_precursor, original_product = original.split(">>", 1)
    if _canonical_unmapped(mapped_precursor) != _canonical_unmapped(original_precursor):
        raise ValueError("mapped precursor changed structure")
    if _canonical_unmapped(mapped_product) != _canonical_unmapped(original_product):
        raise ValueError("mapped product changed structure")

    precursor_mol = Chem.MolFromSmiles(mapped_precursor)
    product_mol = Chem.MolFromSmiles(mapped_product)
    if precursor_mol is None or product_mol is None:
        raise ValueError("invalid mapped reaction")
    precursor_maps: Dict[int, str] = {}
    for atom in precursor_mol.GetAtoms():
        value = int(atom.GetAtomMapNum())
        if value <= 0:
            continue
        if value in precursor_maps:
            raise ValueError("duplicate positive precursor atom map")
        precursor_maps[value] = atom.GetSymbol()
    product_maps: Dict[int, str] = {}
    for atom in product_mol.GetAtoms():
        value = int(atom.GetAtomMapNum())
        if value <= 0:
            raise ValueError("product contains an unmapped atom")
        if value in product_maps:
            raise ValueError("duplicate product atom map")
        product_maps[value] = atom.GetSymbol()
    for value in set(precursor_maps).intersection(product_maps):
        if precursor_maps[value] != product_maps[value]:
            raise ValueError("shared atom map changes element")


def command_map(args: argparse.Namespace) -> int:
    try:
        from rxnmapper import RXNMapper
    except ImportError as exc:
        raise SystemExit(
            "RXNMapper is required for `map`; activate the MechET environment"
        ) from exc

    source_rows = read_jsonl(args.input, args.limit)
    mapper = RXNMapper()
    mapped_rows: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []
    mapped_successes = 0

    for batch in chunks(source_rows, args.batch_size):
        reactions: List[str] = []
        valid_rows: List[Dict[str, Any]] = []
        for row in batch:
            identifier = stable_id(row)
            try:
                reactions.append(unmapped_reaction_smiles(row))
                valid_rows.append(row)
            except Exception as exc:  # preserve the failed stable ID
                failed = dict(row)
                failed["id"] = identifier
                failed["stable_id"] = identifier
                failed["mapping_status"] = "input_error"
                failed["mapping_error"] = str(exc)
                failed["reaction_mapped"] = ""
                mapped_rows.append(failed)
                ledger.append(
                    {"stable_id": identifier, "status": "input_error", "error": str(exc)}
                )

        if not reactions:
            continue
        try:
            outputs = mapper.get_attention_guided_atom_maps(
                reactions, canonicalize_rxns=False, detailed_output=True
            )
        except Exception:
            outputs = []
            for reaction in reactions:
                try:
                    result = mapper.get_attention_guided_atom_maps(
                        [reaction], canonicalize_rxns=False, detailed_output=True
                    )[0]
                except Exception as exc:
                    result = {"mapped_rxn": "", "mapping_error": str(exc)}
                outputs.append(result)

        if len(outputs) != len(valid_rows):
            raise RuntimeError(
                "RXNMapper returned {} results for {} inputs".format(
                    len(outputs), len(valid_rows)
                )
            )

        for row, result in zip(valid_rows, outputs):
            identifier = stable_id(row)
            mapped = str(result.get("mapped_rxn") or "").strip()
            mapping_error = ""
            if ">>" in mapped:
                try:
                    _validate_mapped_reaction(mapped, unmapped_reaction_smiles(row))
                except Exception as exc:
                    mapping_error = "{}: {}".format(type(exc).__name__, exc)
                    mapped = ""
            if ">>" not in mapped:
                failed = dict(row)
                failed["id"] = identifier
                failed["stable_id"] = identifier
                failed["mapping_status"] = "mapping_error"
                failed["mapping_error"] = str(
                    mapping_error or result.get("mapping_error") or "missing mapped_rxn"
                )
                failed["reaction_mapped"] = ""
                mapped_rows.append(failed)
                ledger.append(
                    {
                        "stable_id": identifier,
                        "status": "mapping_error",
                        "error": str(
                            mapping_error
                            or result.get("mapping_error")
                            or "missing mapped_rxn"
                        ),
                    }
                )
                continue
            precursor, product = mapped.split(">>", 1)
            output = dict(row)
            output["id"] = identifier
            output["stable_id"] = identifier
            output["reaction_mapped"] = mapped
            output["precursor_mapped"] = precursor
            output["product_mapped"] = product
            output["rxnmapper_confidence"] = result.get("confidence")
            output["mapping_status"] = "mapped"
            mapped_rows.append(output)
            mapped_successes += 1
            ledger.append(
                {
                    "stable_id": identifier,
                    "status": "mapped",
                    "confidence": result.get("confidence"),
                }
            )

    source_ids = [stable_id(row) for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("mapping input contains duplicate stable IDs")
    source_order = {identifier: index for index, identifier in enumerate(source_ids)}
    mapped_rows.sort(key=lambda row: source_order[stable_id(row)])
    ledger.sort(key=lambda row: source_order[str(row["stable_id"])])
    if [stable_id(row) for row in mapped_rows] != source_ids:
        raise RuntimeError("mapped cache did not preserve the complete input ID order")

    write_jsonl(args.output, mapped_rows)
    ledger_path = args.output.with_suffix(".ledger.jsonl")
    write_jsonl(ledger_path, ledger)
    report = {
        "command": "map",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "input_rows": len(source_rows),
        "output_rows": len(mapped_rows),
        "mapped_rows": mapped_successes,
        "failed_rows": len(source_rows) - mapped_successes,
        "ledger": str(ledger_path.resolve()),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _seed_for(base_seed: int, identifier: str, augmentation: int) -> int:
    digest = hashlib.sha256(
        "{}\n{}\n{}".format(base_seed, identifier, augmentation).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _clear_maps_product(product: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(product, sanitize=False)
    if mol is None:
        raise ValueError("invalid product SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=False, kekuleSmiles=True)


def _remove_amap_not_in_product(reaction: str) -> str:
    from rdkit import Chem

    reactants, product = reaction.split(">>", 1)
    product_mol = Chem.MolFromSmiles(product)
    reactant_mol = Chem.MolFromSmiles(reactants)
    if product_mol is None or reactant_mol is None:
        raise ValueError("invalid reaction SMILES")
    product_maps = {atom.GetAtomMapNum() for atom in product_mol.GetAtoms()}
    if not product_maps or min(product_maps) <= 0:
        raise ValueError("product requires complete positive atom maps")
    next_map = max(product_maps) + 1
    for atom in reactant_mol.GetAtoms():
        if atom.GetAtomMapNum() not in product_maps:
            atom.SetAtomMapNum(next_map)
            next_map += 1
    return Chem.MolToSmiles(reactant_mol) + ">>" + product


def _random_smiles_with_map(smiles_with_map: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles_with_map)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("invalid mapped SMILES")
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) > 1:
        randomized = [
            _random_smiles_with_map(
                Chem.MolToSmiles(fragment, canonical=False, isomericSmiles=True)
            )
            for fragment in fragments
        ]
        random.shuffle(randomized)
        return ".".join(randomized)
    atom_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
    root = random.randint(0, len(atom_maps) - 1)
    Chem.Kekulize(mol)
    Chem.MolToSmiles(mol, rootedAtAtom=root)
    order_value = mol.GetProp("_smilesAtomOutputOrder")
    order = list(ast.literal_eval(order_value))
    ordered = Chem.RenumberAtoms(mol, order)
    for index, original_index in enumerate(order):
        ordered.GetAtomWithIdx(index).SetAtomMapNum(atom_maps[original_index])
    return Chem.MolToSmiles(ordered, canonical=False, kekuleSmiles=True)


def _canonical_smiles_with_map(smiles_with_map: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles_with_map)
    if mol is None:
        raise ValueError("invalid mapped SMILES")
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) > 1:
        canonicalized = [
            _canonical_smiles_with_map(
                Chem.MolToSmiles(fragment, canonical=False, isomericSmiles=True)
            )
            for fragment in fragments
        ]
        return ".".join(sorted(canonicalized))
    atom_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
    Chem.Kekulize(mol)
    order = list(Chem.CanonicalRankAtoms(mol, includeChirality=True))
    ordered = Chem.RenumberAtoms(mol, order)
    for index, original_index in enumerate(order):
        ordered.GetAtomWithIdx(index).SetAtomMapNum(atom_maps[original_index])
    return Chem.MolToSmiles(ordered, canonical=False, kekuleSmiles=True)


def _ensure_temporary_product_maps(product: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(product)
    if mol is None:
        raise ValueError("invalid product SMILES")
    maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if any(value <= 0 for value in maps) or len(set(maps)) != len(maps):
        for index, atom in enumerate(mol.GetAtoms(), 1):
            atom.SetAtomMapNum(index)
    return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)


def _augment_reaction(reaction: str) -> str:
    from rdkit import Chem

    reaction = _remove_amap_not_in_product(reaction)
    reactants, product = reaction.split(">>", 1)
    product = _random_smiles_with_map(product)
    product_mol = Chem.MolFromSmiles(product, sanitize=False)
    old_maps = [atom.GetAtomMapNum() for atom in product_mol.GetAtoms()]
    for index, atom in enumerate(product_mol.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    product = Chem.MolToSmiles(product_mol, canonical=False, kekuleSmiles=True)
    new_maps = [
        atom.GetAtomMapNum()
        for atom in Chem.MolFromSmiles(product, sanitize=False).GetAtoms()
    ]
    if new_maps != list(range(1, len(new_maps) + 1)):
        raise ValueError("product atom-map renumbering failed")
    old_to_new = dict(zip(old_maps, new_maps))

    reactant_mol = Chem.MolFromSmiles(reactants)
    if reactant_mol is None:
        raise ValueError("invalid precursor SMILES")
    for atom in reactant_mol.GetAtoms():
        old = atom.GetAtomMapNum()
        if old in old_to_new:
            atom.SetAtomMapNum(old_to_new[old])
    reactants = _canonical_smiles_with_map(Chem.MolToSmiles(reactant_mol))
    return reactants + ">>" + product


def _random_test_source(product: str) -> str:
    mapped_product = _ensure_temporary_product_maps(product)
    randomized = _random_smiles_with_map(mapped_product)
    return _clear_maps_product(randomized)


def _canonical_unmapped(smiles: str) -> str:
    from rdkit import Chem

    parts: List[str] = []
    for text in str(smiles or "").split("."):
        if not text.strip():
            continue
        mol = Chem.MolFromSmiles(text.strip())
        if mol is None:
            return ""
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ".".join(sorted(parts))


def _round_trip_matches(reactseq: str, reference: str) -> bool:
    from e_smiles import merge_smiles_only

    reconstructed = merge_smiles_only(reactseq)
    return bool(reconstructed) and _canonical_unmapped(reconstructed) == _canonical_unmapped(reference)


def _initialize_prepare_worker(rows: List[Dict[str, Any]]) -> None:
    global _PREPARE_ROWS
    _PREPARE_ROWS = rows


def _prepare_one(task: Tuple[int, int, str, int, bool]) -> Dict[str, Any]:
    source_index, augmentation, split, base_seed, evaluation_only = task
    row = _PREPARE_ROWS[source_index]
    identifier = stable_id(row)
    random.seed(_seed_for(base_seed, identifier, augmentation))
    result: Dict[str, Any] = {
        "stable_id": identifier,
        "source_index": source_index,
        "augmentation_index": augmentation,
        "split": split,
    }
    try:
        if evaluation_only:
            source = _random_test_source(product_smiles(row))
            target = ""
        else:
            from e_smiles import get_e_smiles_with_check

            reaction = _augment_reaction(reaction_smiles(row))
            reactseq = get_e_smiles_with_check(reaction)
            source, target = reactseq.split(">>>", 1)
            if not _round_trip_matches(reactseq, precursor_smiles(row)):
                raise ValueError("ReactSeq round-trip does not match precursor")
        result["status"] = "emitted"
        result["source"] = source
        result["target"] = target
    except Exception as exc:
        result["status"] = "conversion_error"
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    return result


def command_prepare(args: argparse.Namespace) -> int:
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError as exc:
        raise SystemExit("ReactSeq preprocessing requires RDKit") from exc
    if args.evaluation_only and args.split == "train":
        raise ValueError("--evaluation-only is only valid for validation/test data")
    evaluation_only = bool(args.evaluation_only or args.split == "test")
    if not evaluation_only:
        try:
            import indigo  # noqa: F401
            import e_smiles  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "train/valid conversion requires epam.indigo; activate rdkit2019"
            ) from exc

    rows = read_jsonl(args.input, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    src_path = args.output_dir / "src.txt"
    tgt_path = args.output_dir / "tgt.txt"
    line_map_path = args.output_dir / "line_map.jsonl"
    ledger_path = args.output_dir / "ledger.jsonl"
    failures_path = args.output_dir / "failures.jsonl"

    emitted = 0
    failures = 0
    ledger_count = 0
    tasks = (
        (source_index, augmentation, args.split, args.seed, evaluation_only)
        for augmentation in range(args.augmentations)
        for source_index in range(len(rows))
    )
    pool: Optional[Any] = None
    if args.workers > 1:
        pool = mp.Pool(
            args.workers, initializer=_initialize_prepare_worker, initargs=(rows,)
        )
        results: Iterable[Dict[str, Any]] = pool.imap(
            _prepare_one, tasks, chunksize=args.chunksize
        )
    else:
        _initialize_prepare_worker(rows)
        results = map(_prepare_one, tasks)

    try:
        with src_path.open("w", encoding="utf-8") as src_handle, tgt_path.open(
            "w", encoding="utf-8"
        ) as tgt_handle, line_map_path.open("w", encoding="utf-8") as map_handle, ledger_path.open(
            "w", encoding="utf-8"
        ) as ledger_handle, failures_path.open("w", encoding="utf-8") as failure_handle:
            for ledger in results:
                if ledger["status"] == "emitted":
                    source = str(ledger.pop("source"))
                    target = str(ledger.pop("target"))
                    src_handle.write(" ".join(list(source)) + "\n")
                    if not evaluation_only:
                        tgt_handle.write(" ".join(list(target)) + "\n")
                    map_handle.write(
                        json.dumps(
                            {
                                "line_index": emitted,
                                "stable_id": ledger["stable_id"],
                                "source_index": ledger["source_index"],
                                "augmentation_index": ledger["augmentation_index"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    ledger["line_index"] = emitted
                    emitted += 1
                else:
                    failure_handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
                    failures += 1
                ledger_handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
                ledger_count += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    report = {
        "command": "prepare",
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "split": args.split,
        "input_rows": len(rows),
        "augmentations": args.augmentations,
        "expected_attempts": len(rows) * args.augmentations,
        "ledger_rows": ledger_count,
        "emitted_rows": emitted,
        "failed_attempts": failures,
        "seed": args.seed,
        "workers": args.workers,
        "evaluation_only": evaluation_only,
        "test_is_product_only": args.split == "test" and evaluation_only,
        "evaluation_denominator_rows": len(rows) if evaluation_only else None,
        "failures_counted_as_errors": evaluation_only,
        "reference_ids_filtered_from_evaluation": False if evaluation_only else None,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if evaluation_only and emitted != len(rows) * args.augmentations:
        print(
            "WARNING: evaluation conversion failures remain explicit and must be scored as errors",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    map_parser = subparsers.add_parser("map", help="atom-map endpoint reactions")
    map_parser.add_argument("--input", type=Path, required=True)
    map_parser.add_argument("--output", type=Path, required=True)
    map_parser.add_argument("--batch-size", type=int, default=32)
    map_parser.add_argument("--limit", type=int, default=0)
    map_parser.set_defaults(func=command_map)

    prepare_parser = subparsers.add_parser(
        "prepare", help="generate native ReactSeq src/tgt files"
    )
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--split", choices=("train", "valid", "test"), required=True
    )
    prepare_parser.add_argument("--augmentations", type=int, required=True)
    prepare_parser.add_argument("--seed", type=int, default=3435)
    prepare_parser.add_argument("--limit", type=int, default=0)
    prepare_parser.add_argument("--workers", type=int, default=1)
    prepare_parser.add_argument("--chunksize", type=int, default=16)
    prepare_parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="generate product-only inputs and preserve the complete validation/test denominator",
    )
    prepare_parser.set_defaults(func=command_prepare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
