"""Conservative multi-source data normalization for the forward expert.

The normalizer understands the public FlowER/mech-USPTO mechanistic-SMILES
layout, PMechDB arrow codes, generic JSON/CSV/Parquet/Arrow tables and ORD
protobuf rows. It never invents a source-sink arrow. Ambiguous or missing arrows
remain unlabeled and may train only the reaction-compatibility head.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from rdkit import Chem

from .forward_expert import (
    ElectronContainer,
    ElectronMove,
    verify_electron_step,
)

REACTION_FIELDS = (
    "reaction_smiles", "rxn_smiles", "mapped_reaction_smiles",
    "mapped_reaction", "reaction", "rxn", "smirks", "reaction_smirks",
)
REACTANT_FIELDS = (
    "reactants", "reactant_smiles", "reactants_smiles", "precursors",
    "elem_reac_spe", "elem_reac_ori", "elem_reac_equ", "elem_reac_min",
)
PRODUCT_FIELDS = (
    "products", "product", "product_smiles", "target_product", "target",
    "rxn_prod_spe", "rxn_prod_ori", "elem_prod_spe", "elem_prod_ori",
    "rxn_prod_equ", "rxn_prod_min", "elem_prod_equ", "elem_prod_min",
)
REAGENT_FIELDS = ("reagents", "agents", "reagent_smiles", "conditions_smiles")
CLASS_FIELDS = (
    "mechanism_class", "reaction_class", "class", "rxn_class",
    "name_reaction", "orbital_pair_classification", "orbital_pair",
)
CONDITION_FIELDS = ("conditions", "condition", "procedure", "solvent", "temperature")
STEP_FIELDS = ("steps", "mechanism_steps", "electron_flow", "mechanism", "trajectory")
MOVE_FIELDS = (
    "moves", "arrows", "arrow_codes", "arrow_code", "electron_moves",
    "source_sink_pairs", "edits",
)
COMPETITOR_FIELDS = (
    "competitor_products", "negative_products", "side_products", "alternatives",
)
MECHANISTIC_VARIANTS = (
    ("ori", "mech_smi_ori", "elem_reac_ori", "elem_prod_ori"),
    ("spe", "mech_smi_spe", "elem_reac_spe", "elem_prod_spe"),
    ("equ", "mech_smi_equ", "elem_reac_equ", "elem_prod_equ"),
    ("min", "mech_smi_min", "elem_reac_min", "elem_prod_min"),
)


@dataclass(frozen=True)
class StandardizationReport:
    source: str
    read: int
    written: int
    quarantined: int
    with_moves: int
    without_moves: int
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first(row: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "[{" or text in {"null", "true", "false"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return value


def split_reaction_smiles(value: str) -> tuple[str, str, str]:
    text = str(value or "").strip()
    if ">>" in text:
        left, right = text.split(">>", 1)
        return left.strip(), "", right.strip()
    parts = text.split(">")
    if len(parts) == 3:
        return tuple(part.strip() for part in parts)  # type: ignore[return-value]
    raise ValueError("reaction SMILES must contain '>>' or two '>' separators")



def _parse_mol(smiles: str) -> Chem.Mol:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError("empty SMILES")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(text, params)
    if mol is None:
        raise ValueError(f"invalid SMILES: {text}")
    return mol


def _canonical(smiles: str, *, require_maps: bool = True) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    mol = _parse_mol(text)
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    positive_maps = [value for value in maps if value > 0]
    if len(positive_maps) != len(set(positive_maps)):
        raise ValueError("positive atom maps must be unique")
    if require_maps and len(positive_maps) != len(maps):
        raise ValueError("unique positive atom maps are required")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _complete_atom_maps(smiles: str) -> tuple[str, dict[str, Any]]:
    """Fill missing maps without changing arrow-referenced positive map labels."""
    mol = _parse_mol(smiles)
    existing = [
        atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    ]
    if len(existing) != len(set(existing)):
        raise ValueError("duplicate positive atom maps in mechanistic state")
    rank_mol = Chem.Mol(mol)
    for atom in rank_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    ranks = list(Chem.CanonicalRankAtoms(rank_mol, breakTies=True))
    unmapped = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() <= 0
    ]
    next_map = max(existing, default=0) + 1
    assigned: dict[str, int] = {}
    for atom_idx in sorted(unmapped, key=lambda idx: (ranks[idx], idx)):
        while next_map in existing:
            next_map += 1
        mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(next_map)
        assigned[str(atom_idx)] = next_map
        existing.append(next_map)
        next_map += 1
    output = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    _canonical(output, require_maps=True)
    return output, {
        "completed": bool(unmapped),
        "assigned_maps": assigned,
        "original_mapped_atoms": len(existing) - len(unmapped),
        "completed_atoms": len(unmapped),
    }


def _unmapped_canonical(smiles: str) -> str:
    mol = _parse_mol(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _same_structure_unmapped(left: str, right: str) -> bool:
    return _unmapped_canonical(left) == _unmapped_canonical(right)

def _stable_split(identity: str) -> str:
    fraction = int(
        hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12], 16
    ) / float(16**12)
    if fraction < 0.90:
        return "train"
    if fraction < 0.95:
        return "valid"
    return "test"


def _container_text(text: str, role: str) -> dict[str, Any]:
    cleaned = text.strip().replace("(", ":").replace(")", "").replace(" ", "")
    if ":" in cleaned:
        kind, atom_text = cleaned.split(":", 1)
    else:
        kind, atom_text = ("LP" if role == "source" else "ATOM"), cleaned
    atoms = [int(part) for part in atom_text.replace("-", ",").split(",") if part]
    return {"kind": kind, "atoms": atoms}


def _pmech_arrow(value: str) -> dict[str, Any]:
    if "=" not in value:
        raise ValueError("not a PMechDB arrow code")
    source_text, sink_text = (part.strip() for part in value.split("=", 1))

    def endpoint(part: str) -> int | tuple[int, ...]:
        atoms = [int(token) for token in part.split(",") if token.strip()]
        return atoms[0] if len(atoms) == 1 else tuple(atoms)

    source = endpoint(source_text)
    sink = endpoint(sink_text)
    source_container = (
        {"kind": "LP", "atoms": [source]}
        if isinstance(source, int)
        else {"kind": "BOND", "atoms": list(source)}
    )
    if isinstance(sink, int):
        sink_container = (
            {"kind": "BOND", "atoms": [source, sink]}
            if isinstance(source, int)
            else {"kind": "ATOM", "atoms": [sink]}
        )
    else:
        sink_container = {"kind": "BOND", "atoms": list(sink)}
    return ElectronMove.parse(
        {"source": source_container, "sink": sink_container}
    ).to_dict()


def _normalise_move(value: Any) -> dict[str, Any]:
    value = _jsonish(value)
    if isinstance(value, dict) and "source" in value and "sink" in value:
        return ElectronMove.parse(value).to_dict()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return ElectronMove.parse(
            {"source": value[0], "sink": value[1]}
        ).to_dict()
    if isinstance(value, str) and "->" in value:
        source, sink = value.split("->", 1)
        return ElectronMove.parse(
            {
                "source": _container_text(source, "source"),
                "sink": _container_text(sink, "sink"),
            }
        ).to_dict()
    if isinstance(value, str) and "=" in value:
        return _pmech_arrow(value)
    raise ValueError(f"cannot parse electron move: {value!r}")


def parse_mechanistic_smiles(value: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse ``state|(source,sink);...`` used by mech-USPTO/FlowER."""
    text = str(value or "").strip()
    if "|" not in text:
        return text, []
    state, encoded = text.rsplit("|", 1)
    moves = []
    for token in encoded.split(";"):
        try:
            source, sink = ast.literal_eval(token.strip())
        except (SyntaxError, ValueError, TypeError):
            continue
        if isinstance(source, int):
            source_container = {"kind": "LP", "atoms": [source]}
        elif isinstance(source, (list, tuple)) and len(source) == 2:
            source_container = {"kind": "BOND", "atoms": list(source)}
        else:
            continue
        if isinstance(sink, int):
            sink_container = (
                {"kind": "BOND", "atoms": [source, sink]}
                if isinstance(source, int)
                else {"kind": "ATOM", "atoms": [sink]}
            )
        elif isinstance(sink, (list, tuple)) and len(sink) == 2:
            sink_container = {"kind": "BOND", "atoms": list(sink)}
        else:
            continue
        try:
            moves.append(
                ElectronMove.parse(
                    {"source": source_container, "sink": sink_container}
                ).to_dict()
            )
        except ValueError:
            continue
    return state, moves


def _moves(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("mech_smi_ori", "mech_smi_spe", "mech_smi_equ", "mech_smi_min"):
        if row.get(key):
            _, parsed = parse_mechanistic_smiles(str(row[key]))
            if parsed:
                return parsed
    raw = _jsonish(_first(row, MOVE_FIELDS, []))
    if isinstance(raw, dict):
        raw = raw.get("moves") or raw.get("arrows") or [raw]
    if isinstance(raw, str) and ";" in raw:
        raw = [token for token in raw.split(";") if token.strip()]
    if raw in (None, ""):
        raw = []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    output = []
    for value in raw:
        try:
            output.append(_normalise_move(value))
        except (TypeError, ValueError, KeyError):
            continue
    if output:
        return output
    sources = _jsonish(row.get("sources") or row.get("source"))
    sinks = _jsonish(row.get("sinks") or row.get("sink"))
    if sources is None or sinks is None:
        return []
    if not isinstance(sources, (list, tuple)) or (
        sources and isinstance(sources[0], int)
    ):
        sources = [sources]
    if not isinstance(sinks, (list, tuple)) or (
        sinks and isinstance(sinks[0], int)
    ):
        sinks = [sinks]
    for source, sink in zip(sources, sinks):
        try:
            output.append(
                ElectronMove(
                    ElectronContainer.parse(_jsonish(source), "source"),
                    ElectronContainer.parse(_jsonish(sink), "sink"),
                ).to_dict()
            )
        except (TypeError, ValueError, KeyError):
            continue
    return output



def _mechanistic_step(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover a fully mapped elementary state pair from partial-map mech_smi."""
    failures: list[str] = []
    for variant, mech_key, reactant_key, product_key in MECHANISTIC_VARIANTS:
        raw = row.get(mech_key)
        if not raw:
            continue
        state_partial, moves = parse_mechanistic_smiles(str(raw))
        if not moves:
            failures.append(f"{variant}: no arrow pairs")
            continue
        try:
            state, map_info = _complete_atom_maps(state_partial)
            referenced_maps = {
                atom_map
                for move in moves
                for side in (move["source"], move["sink"])
                for atom_map in side["atoms"]
            }
            state_maps = {
                atom.GetAtomMapNum() for atom in _parse_mol(state).GetAtoms()
            }
            missing = referenced_maps - state_maps
            if missing:
                raise ValueError(
                    f"arrow atom maps missing from state: {sorted(missing)}"
                )
            executed = verify_electron_step(state, moves)
            if not executed.get("ok"):
                raise ValueError(
                    "formal step execution failed: "
                    + str(executed.get("message") or executed.get("code"))
                )
            target = _canonical(str(executed["state_smiles"]), require_maps=True)
            reactant_reference = str(row.get(reactant_key) or "")
            product_reference = str(row.get(product_key) or "")
            reactant_match = (
                not reactant_reference
                or _same_structure_unmapped(state, reactant_reference)
            )
            product_match = (
                not product_reference
                or _same_structure_unmapped(target, product_reference)
            )
            if not reactant_match or not product_match:
                raise ValueError(
                    "reference mismatch "
                    f"reactant={reactant_match} product={product_match}"
                )
            return {
                "variant": variant,
                "step_index": int(
                    row.get("step_idx_forward", row.get("step_index", 0)) or 0
                ),
                "state_smiles": state,
                "target_product": target,
                "moves": moves,
                "map_completion": map_info,
                "reference_reactant": reactant_reference,
                "reference_product": product_reference,
                "reference_reactant_match": reactant_match,
                "reference_product_match": product_match,
            }
        except Exception as exc:
            failures.append(f"{variant}: {exc}")
    if failures:
        raise ValueError(
            "mechanistic row could not be reconstructed; " + " | ".join(failures)
        )
    return None


def _steps(
    row: Mapping[str, Any],
    reactants: str,
    product: str,
) -> list[dict[str, Any]]:
    raw = _jsonish(_first(row, STEP_FIELDS, []))
    if isinstance(raw, dict):
        raw = raw.get("steps") or raw.get("trajectory") or [raw]
    output = []
    for index, item in enumerate(raw if isinstance(raw, (list, tuple)) else []):
        if not isinstance(item, Mapping):
            continue
        state = _first(
            item,
            ("state_smiles", "state_before", "reactants", "input_smiles"),
            reactants,
        )
        target = _first(
            item,
            ("target_product", "state_after", "product", "output_smiles"),
            product,
        )
        moves = _moves(item)
        if moves:
            output.append(
                {
                    "step_index": index,
                    "state_smiles": _canonical(str(state), require_maps=True),
                    "target_product": _canonical(str(target), require_maps=True),
                    "moves": moves,
                }
            )
    if not output:
        moves = _moves(row)
        if moves:
            output.append(
                {
                    "step_index": 0,
                    "state_smiles": reactants,
                    "target_product": product,
                    "moves": moves,
                }
            )
    return output

def _adapt_ord_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("reaction")
    if raw is None or isinstance(raw, str):
        return row
    try:
        from google.protobuf.json_format import MessageToDict
        from ord_schema.message_helpers import get_reaction_smiles
        from ord_schema.proto import reaction_pb2
    except ImportError as exc:
        raise RuntimeError(
            "install mechet[ord] to decode ORD protobuf rows"
        ) from exc
    message = reaction_pb2.Reaction()
    message.ParseFromString(bytes(raw))
    reaction_smiles = get_reaction_smiles(
        message,
        generate_if_missing=True,
        allow_incomplete=True,
        allow_unspecified_roles=True,
        validate=False,
        canonical=True,
    )
    if not reaction_smiles:
        raise ValueError("ORD reaction has no usable structural reaction SMILES")
    output = dict(row)
    output["reaction_smiles"] = reaction_smiles
    output["reaction_id"] = message.reaction_id or row.get("reaction_id")
    output["conditions"] = MessageToDict(
        message.conditions,
        preserving_proto_field_name=True,
    )
    return output



def normalize_reaction_row(
    row: Mapping[str, Any],
    *,
    source: str,
    row_index: int,
    require_maps: bool = True,
) -> dict[str, Any]:
    row = _adapt_ord_row(row)
    aliases = {
        str(key).strip().lower().replace(" ", "_"): value
        for key, value in row.items()
    }
    row = {**dict(row), **aliases}
    metadata: dict[str, Any] = {
        "original_keys": sorted(str(key) for key in row)
    }
    mechanistic = _mechanistic_step(row)
    if mechanistic is not None:
        reactants = mechanistic["state_smiles"]
        products = mechanistic["target_product"]
        reagents = ""
        steps = [
            {
                key: mechanistic[key]
                for key in (
                    "step_index", "state_smiles", "target_product", "moves"
                )
            }
        ]
        metadata.update(
            {
                "mechanistic_variant": mechanistic["variant"],
                "atom_map_policy": "complete_partial_mechanistic_maps",
                "map_completion": mechanistic["map_completion"],
                "reference_reactant": mechanistic["reference_reactant"],
                "reference_product": mechanistic["reference_product"],
                "reference_reactant_match": mechanistic[
                    "reference_reactant_match"
                ],
                "reference_product_match": mechanistic[
                    "reference_product_match"
                ],
                "mapped_target_source": "formal_electron_step_execution",
            }
        )
    else:
        reaction = str(_first(row, REACTION_FIELDS, "") or "")
        if reaction:
            reactants, reagents, products = split_reaction_smiles(reaction)
        else:
            reactants = str(_first(row, REACTANT_FIELDS, "") or "")
            products = str(_first(row, PRODUCT_FIELDS, "") or "")
            reagents = str(_first(row, REAGENT_FIELDS, "") or "")
        reactants = _canonical(reactants, require_maps=require_maps)
        products = _canonical(products, require_maps=require_maps)
        if not reactants or not products:
            raise ValueError("reactants and products are required")
        reagents = _canonical(reagents, require_maps=False) if reagents else ""
        steps = _steps(row, reactants, products)
    reaction = f"{reactants}>{reagents}>{products}"
    identity = str(
        row.get("id")
        or row.get("reaction_id")
        or row.get("rxn_idx")
        or hashlib.sha1(reaction.encode()).hexdigest()[:16]
    )
    split = str(row.get("split") or row.get("set") or "").lower()
    if split in {"validation", "val", "dev"}:
        split = "valid"
    if split not in {"train", "valid", "test"}:
        split = _stable_split(f"{source}:{identity}")
    competitors = _jsonish(_first(row, COMPETITOR_FIELDS, []))
    if isinstance(competitors, str):
        competitors = [value for value in competitors.split("|") if value]
    normalized_competitors = []
    for value in competitors or []:
        try:
            normalized_competitors.append(
                _canonical(str(value), require_maps=require_maps)
            )
        except ValueError:
            continue
    return {
        "id": identity,
        "source": source,
        "source_row": int(row_index),
        "reaction_smiles": reaction,
        "reactants": reactants,
        "reagents": reagents,
        "products": products,
        "mechanism_class": str(_first(row, CLASS_FIELDS, "") or ""),
        "conditions": _jsonish(_first(row, CONDITION_FIELDS, {})),
        "competitor_products": normalized_competitors,
        "steps": steps,
        "split": split,
        "metadata": metadata,
    }

def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield dict(json.loads(line))
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        for row in value:
            yield dict(row)
    elif isinstance(value, dict):
        rows = value.get("data") or value.get("rows") or value.get("reactions")
        if isinstance(rows, list):
            for row in rows:
                yield dict(row)
        else:
            yield dict(value)


def iter_source_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    root = Path(path)
    if root.is_dir():
        if (root / "dataset_dict.json").exists() or (root / "state.json").exists():
            try:
                from datasets import Dataset, DatasetDict, load_from_disk
            except ImportError as exc:
                raise RuntimeError("install mechet[data] for Arrow datasets") from exc
            dataset = load_from_disk(str(root))
            if isinstance(dataset, DatasetDict):
                for split, table in dataset.items():
                    for row in table:
                        yield {**dict(row), "split": split}
            elif isinstance(dataset, Dataset):
                for row in dataset:
                    yield dict(row)
            return
        for file in sorted(
            file
            for file in root.rglob("*")
            if file.suffix.lower() in {".jsonl", ".json", ".csv", ".parquet", ".arrow"}
        ):
            yield from iter_source_rows(file)
        return
    suffix = root.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        yield from _iter_json(root)
    elif suffix == ".csv":
        with root.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield dict(row)
    elif suffix in {".parquet", ".arrow"}:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("install mechet[data] for Parquet/Arrow") from exc
        dataset = (
            Dataset.from_parquet(str(root))
            if suffix == ".parquet"
            else Dataset.from_file(str(root))
        )
        for row in dataset:
            yield dict(row)
    else:
        raise ValueError(f"unsupported data file: {root}")


def standardize_path(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source: str,
    quarantine_path: str | Path | None = None,
    require_maps: bool = True,
    limit: int = 0,
) -> StandardizationReport:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine = (
        Path(quarantine_path)
        if quarantine_path
        else output.with_name(output.stem + ".quarantine.jsonl")
    )
    read = written = with_moves = without_moves = 0
    split_counts = {"train": 0, "valid": 0, "test": 0}
    with output.open("w", encoding="utf-8") as good, quarantine.open(
        "w", encoding="utf-8"
    ) as bad:
        for index, row in enumerate(iter_source_rows(input_path)):
            if limit and read >= limit:
                break
            read += 1
            try:
                normalized = normalize_reaction_row(
                    row,
                    source=source,
                    row_index=index,
                    require_maps=require_maps,
                )
                good.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                written += 1
                split_counts[normalized["split"]] += 1
                if normalized["steps"]:
                    with_moves += 1
                else:
                    without_moves += 1
            except Exception as exc:
                bad.write(
                    json.dumps(
                        {"source_row": index, "error": str(exc), "row": row},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
    return StandardizationReport(
        source,
        read,
        written,
        read - written,
        with_moves,
        without_moves,
        split_counts,
    )


def iter_standardized(
    path: str | Path,
    split: str | None = None,
) -> Iterator[dict[str, Any]]:
    for row in _iter_json(Path(path)):
        if split is None or row.get("split") == split:
            yield row


def flatten_step_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_outcomes: bool = True,
) -> Iterator[dict[str, Any]]:
    for row in rows:
        steps = list(row.get("steps") or [])
        if not steps and include_outcomes:
            steps = [
                {
                    "step_index": 0,
                    "state_smiles": row["reactants"],
                    "target_product": row["products"],
                    "moves": [],
                }
            ]
        for step in steps:
            yield {
                "id": f"{row['id']}:step{step.get('step_index', 0)}",
                "reaction_id": row["id"],
                "state_smiles": step["state_smiles"],
                "target_product": step["target_product"],
                "moves": step["moves"],
                "reactants": row["reactants"],
                "products": row["products"],
                "competitor_products": list(row.get("competitor_products") or []),
                "mechanism_class": row.get("mechanism_class", ""),
                "conditions": row.get("conditions", {}),
                "split": row.get("split", "train"),
            }
