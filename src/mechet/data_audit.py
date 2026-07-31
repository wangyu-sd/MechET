"""Dataset fingerprinting, overlap auditing, and decontamination utilities.

The module deliberately separates normalization from policy. A benchmark is
first frozen into immutable chemical keys; training rows are then quarantined
according to an explicit policy. This makes train--test overlap measurable and
reproducible instead of hiding it inside ad-hoc preprocessing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class NormalizationConfig:
    keep_stereo: bool = True
    remove_atom_maps: bool = True
    canonical: bool = True
    include_environment_in_full_key: bool = True
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ReactionRecord:
    record_id: str
    product: str
    reactants: str
    reagents: str = ""
    proof: str = ""
    patent_id: str = ""
    publication_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReactionKeys:
    exact_full: str
    exact_structural: str
    product: str
    scaffold: str
    reaction_center: str
    proof_composition: str = ""
    patent: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RoleSplit:
    structural: list[str]
    environment: list[str]
    mapped: bool

    @property
    def structural_smiles(self) -> str:
        return ".".join(sorted(self.structural))

    @property
    def environment_smiles(self) -> str:
        return ".".join(sorted(self.environment))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_reaction_smiles(reaction: str) -> tuple[str, str, str]:
    """Return ``(reactants, reagents, products)`` for common reaction syntax."""
    text = (reaction or "").strip()
    if ">>" in text:
        left, right = text.split(">>", 1)
        return left.strip(), "", right.strip()
    parts = text.split(">")
    if len(parts) == 3:
        return tuple(part.strip() for part in parts)  # type: ignore[return-value]
    raise ValueError("reaction must use reactants>>products or reactants>reagents>products")


def _mol(smiles: str, *, sanitize: bool = True) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = sanitize
    mol = Chem.MolFromSmiles((smiles or "").strip(), params)
    if mol is None:
        raise ValueError(f"invalid SMILES: {(smiles or '')[:120]}")
    return mol


def _clear_maps(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")


def canonical_smiles(smiles: str, config: NormalizationConfig) -> str:
    mol = _mol(smiles)
    if config.remove_atom_maps:
        _clear_maps(mol)
    return Chem.MolToSmiles(
        mol,
        canonical=config.canonical,
        isomericSmiles=config.keep_stereo,
    )


def canonical_multiset(smiles: str, config: NormalizationConfig) -> str:
    parts: list[str] = []
    for fragment in (smiles or "").replace(";", ".").split("."):
        fragment = fragment.strip()
        if fragment:
            parts.append(canonical_smiles(fragment, config))
    return ".".join(sorted(parts))


def atom_maps(smiles: str) -> set[int]:
    maps: set[int] = set()
    for fragment in (smiles or "").split("."):
        if not fragment.strip():
            continue
        mol = _mol(fragment)
        for atom in mol.GetAtoms():
            value = atom.GetAtomMapNum()
            if value > 0:
                maps.add(value)
    return maps


def split_structural_and_environment(reactants: str, product: str) -> RoleSplit:
    """Split fragments by whether they contain atoms contributing to the product.

    For atom-mapped reactions, an entire reactant fragment is structural when it
    shares at least one map number with the product. This retains synthetic
    equivalents such as an alkyl bromide while excluding free solvents, salts,
    catalysts, and spectators.

    Unmapped input cannot support this distinction; all fragments are retained as
    structural and ``mapped`` is false so the limitation remains auditable.
    """
    product_maps = atom_maps(product)
    fragments = [part.strip() for part in (reactants or "").split(".") if part.strip()]
    if not product_maps:
        return RoleSplit(fragments, [], False)
    structural: list[str] = []
    environment: list[str] = []
    for fragment in fragments:
        maps = atom_maps(fragment)
        (structural if maps & product_maps else environment).append(fragment)
    return RoleSplit(structural, environment, True)


def _bond_dict(smiles: str) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for fragment in (smiles or "").split("."):
        if not fragment.strip():
            continue
        mol = _mol(fragment)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        for bond in mol.GetBonds():
            a = bond.GetBeginAtom().GetAtomMapNum()
            b = bond.GetEndAtom().GetAtomMapNum()
            if a <= 0 or b <= 0:
                continue
            out[tuple(sorted((a, b)))] = float(bond.GetBondTypeAsDouble())
    return out


def _atom_feature_by_map(smiles: str) -> dict[int, tuple[str, int, bool, int]]:
    out: dict[int, tuple[str, int, bool, int]] = {}
    for fragment in (smiles or "").split("."):
        if not fragment.strip():
            continue
        mol = _mol(fragment)
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap > 0:
                out[amap] = (
                    atom.GetSymbol(),
                    int(atom.GetFormalCharge()),
                    bool(atom.GetIsAromatic()),
                    int(atom.GetDegree()),
                )
    return out


def reaction_center_key(reactants: str, product: str) -> str:
    """Map-label-invariant local bond-change signature.

    This is intentionally named a reaction-center key rather than a full reaction
    template. It captures changed bond orders and local atom roles while avoiding
    arbitrary atom-map identifiers.
    """
    rb = _bond_dict(reactants)
    pb = _bond_dict(product)
    features = _atom_feature_by_map(product)
    features.update({k: v for k, v in _atom_feature_by_map(reactants).items() if k not in features})
    changes: list[tuple[Any, ...]] = []
    for pair in sorted(set(rb) | set(pb)):
        old = rb.get(pair, 0.0)
        new = pb.get(pair, 0.0)
        if abs(old - new) < 1e-6:
            continue
        left = features.get(pair[0], ("?", 0, False, 0))
        right = features.get(pair[1], ("?", 0, False, 0))
        role_pair = tuple(sorted((left, right)))
        changes.append((role_pair, round(old, 2), round(new, 2)))
    payload = json.dumps(changes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() if changes else ""


def scaffold_key(product: str, config: NormalizationConfig) -> str:
    mol = _mol(product)
    if config.remove_atom_maps:
        _clear_maps(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return canonical_smiles(product, config)
    return Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=config.keep_stereo,
    )


def proof_composition_key(proof: str) -> str:
    if not proof:
        return ""
    try:
        from mechet.proof_equivalence import composition_signature

        return composition_signature(proof)
    except Exception:
        return ""


def reaction_keys(record: ReactionRecord, config: NormalizationConfig) -> ReactionKeys:
    roles = split_structural_and_environment(record.reactants, record.product)
    product_key = canonical_multiset(record.product, config)
    structural = canonical_multiset(roles.structural_smiles, config)
    full_left = canonical_multiset(record.reactants, config)
    reagent_key = canonical_multiset(record.reagents, config)
    full_payload = f"{full_left}>{reagent_key}>{product_key}"
    structural_payload = f"{structural}>>{product_key}"
    return ReactionKeys(
        exact_full=hashlib.sha256(full_payload.encode("utf-8")).hexdigest(),
        exact_structural=hashlib.sha256(structural_payload.encode("utf-8")).hexdigest(),
        product=hashlib.sha256(product_key.encode("utf-8")).hexdigest(),
        scaffold=hashlib.sha256(scaffold_key(record.product, config).encode("utf-8")).hexdigest(),
        reaction_center=reaction_center_key(record.reactants, record.product),
        proof_composition=proof_composition_key(record.proof),
        patent=(hashlib.sha256(record.patent_id.strip().encode("utf-8")).hexdigest() if record.patent_id.strip() else ""),
    )


def record_from_mechet_row(row: Mapping[str, Any]) -> ReactionRecord:
    messages = row.get("messages") or []
    product = ""
    proof = ""
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user" and content.startswith("TARGET:"):
            product = content.split("\n", 1)[0].replace("TARGET:", "", 1).strip()
        if role == "assistant":
            proof = content
    metadata = dict(row.get("metadata") or {})
    reactants = str(
        metadata.get("initial_reactants")
        or metadata.get("derived_precursor")
        or metadata.get("reactants")
        or ""
    )
    return ReactionRecord(
        record_id=str(row.get("id") or metadata.get("id") or ""),
        product=product,
        reactants=reactants,
        reagents=str(metadata.get("reagents") or ""),
        proof=proof,
        patent_id=str(metadata.get("patent_id") or metadata.get("patent") or metadata.get("publication_id") or ""),
        publication_date=str(metadata.get("publication_date") or metadata.get("date") or ""),
        metadata=metadata,
    )


def iter_mechet_jsonl(path: str | Path) -> Iterator[ReactionRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            record = record_from_mechet_row(row)
            if not record.record_id:
                record.record_id = f"{Path(path).name}:{line_no}"
            yield record


def iter_reaction_table(
    path: str | Path,
    *,
    reaction_field: str = "reaction_smiles",
    id_field: str = "id",
) -> Iterator[ReactionRecord]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows: Iterable[Mapping[str, Any]] = (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    else:
        import csv

        handle = path.open("r", encoding="utf-8", newline="")
        rows = csv.DictReader(handle)
    for index, row in enumerate(rows):
        reaction = str(row.get(reaction_field) or row.get("rxn_smiles") or row.get("reaction") or "")
        if not reaction:
            continue
        reactants, reagents, products = split_reaction_smiles(reaction)
        yield ReactionRecord(
            record_id=str(row.get(id_field) or row.get("reaction_id") or index),
            product=products,
            reactants=reactants,
            reagents=reagents,
            patent_id=str(row.get("patent_id") or row.get("patent") or row.get("publication_id") or ""),
            publication_date=str(row.get("publication_date") or row.get("date") or ""),
            metadata=dict(row),
        )


def fingerprint_product(product: str, config: NormalizationConfig):
    mol = _mol(product)
    if config.remove_atom_maps:
        _clear_maps(mol)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.fingerprint_radius,
        fpSize=config.fingerprint_bits,
    )
    return generator.GetFingerprint(mol)


def max_tanimoto_against_reference(
    product: str,
    reference_products: Sequence[str],
    config: NormalizationConfig,
    *,
    chunk_size: int = 4096,
) -> float:
    """Exact nearest-neighbour similarity for a bounded reference sample."""
    query = fingerprint_product(product, config)
    best = 0.0
    for start in range(0, len(reference_products), chunk_size):
        fps = [fingerprint_product(value, config) for value in reference_products[start : start + chunk_size]]
        if fps:
            best = max(best, max(DataStructs.BulkTanimotoSimilarity(query, fps)))
    return float(best)


KEY_LEVELS = (
    "exact_full",
    "exact_structural",
    "product",
    "scaffold",
    "reaction_center",
    "proof_composition",
    "patent",
)


def overlap_summary(
    reference: Mapping[str, set[str]],
    query_rows: Iterable[tuple[str, ReactionKeys]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = {level: 0 for level in KEY_LEVELS}
    conflicts: list[dict[str, Any]] = []
    for record_id, keys in query_rows:
        key_dict = keys.as_dict()
        matched = [
            level
            for level in KEY_LEVELS
            if key_dict[level] and key_dict[level] in reference.get(level, set())
        ]
        for level in matched:
            counts[level] += 1
        if matched:
            conflicts.append({"id": record_id, "levels": matched, "keys": key_dict})
    return counts, conflicts


def build_key_index(rows: Iterable[tuple[str, ReactionKeys]]) -> dict[str, set[str]]:
    index = {level: set() for level in KEY_LEVELS}
    for _, keys in rows:
        values = keys.as_dict()
        for level in KEY_LEVELS:
            if values[level]:
                index[level].add(values[level])
    return index


def quarantine_reason(keys: ReactionKeys, benchmark_index: Mapping[str, set[str]], policy: Sequence[str]) -> list[str]:
    values = keys.as_dict()
    return [
        level
        for level in policy
        if level in values and values[level] and values[level] in benchmark_index.get(level, set())
    ]
