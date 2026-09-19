"""Typed environment imports and open-vocabulary reactive-fragment programs.

Environment/context molecules are selected from a catalog.  A fragment whose
private atoms participate in any reference electron move is instead compiled
to a deterministic graph-construction program.  Atom maps are used only to
derive supervision and are absent from the program seen by the model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from rdkit import Chem


REACTIVE_ROLES = (
    "NUCLEOPHILE",
    "ELECTROPHILE",
    "BOND_DONOR",
    "LEAVING_GROUP",
    "PROTON_TRANSFER",
    "RADICAL",
    "REDOX",
    "OTHER_REACTIVE",
)


@dataclass(frozen=True)
class FragmentAtom:
    atomic_num: int
    formal_charge: int = 0
    isotope: int = 0
    explicit_h: int = 0
    no_implicit: bool = False
    radical_electrons: int = 0
    chiral_tag: int = 0
    parent: int | None = None
    parent_bond_type: int | None = None


@dataclass(frozen=True)
class FragmentBond:
    atoms: tuple[int, int]
    bond_type: int
    stereo: int = 0
    stereo_atoms: tuple[int, int] = ()
    bond_dir: int = 0


@dataclass(frozen=True)
class ReactiveFragmentProgram:
    role: str
    atoms: tuple[FragmentAtom, ...]
    extra_bonds: tuple[FragmentBond, ...]
    active_atoms: tuple[int, ...]
    source_unmapped_smiles: str

    def __post_init__(self) -> None:
        if self.role not in REACTIVE_ROLES:
            raise ValueError(f"unknown reactive role: {self.role}")
        if not self.atoms:
            raise ValueError("reactive fragment cannot be empty")
        if not self.active_atoms:
            raise ValueError("reactive fragment requires at least one active atom")
        if set(self.active_atoms) - set(range(len(self.atoms))):
            raise ValueError("active atom outside fragment")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportSupervision:
    fragment: str
    kind: str
    participating_maps: tuple[int, ...]
    role: str | None


@dataclass(frozen=True)
class ReactiveImportGuard:
    """One-event obligation created after importing a reactive fragment.

    ``active_maps`` are executor-assigned private identifiers, not policy
    features.  The immediately following committed electron event must touch
    at least one of them.  This prevents a generated reagent from being added
    as inert endpoint decoration while retaining map-invariant generation.
    """

    active_maps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.active_maps or any(int(value) <= 0 for value in self.active_maps):
            raise ValueError("reactive import guard requires positive private maps")

    def validate(self, moves: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        touched = electron_event_maps(moves)
        used = tuple(sorted(set(self.active_maps) & touched))
        if used:
            return {"ok": True, "code": "REACTIVE_IMPORT_USED", "used_maps": used}
        return {
            "ok": False,
            "code": "UNUSED_REACTIVE_IMPORT",
            "message": "the next committed electron event must use an active atom from the imported fragment",
            "required_maps": self.active_maps,
            "touched_maps": tuple(sorted(touched)),
        }


def electron_event_maps(moves: Sequence[Mapping[str, Any]]) -> set[int]:
    """Return all private atom maps touched by one FLOW/BE_DELTA event."""

    touched: set[int] = set()
    for move in moves:
        if move.get("mode") == "BE_DELTA":
            for item in move.get("bond_deltas") or ():
                touched.update(int(value) for value in item.get("atoms") or ())
            for item in move.get("charge_actions") or ():
                atom_map = int(item.get("atom_map") or 0)
                if atom_map > 0:
                    touched.add(atom_map)
            continue
        for side in ("source", "sink"):
            touched.update(
                int(value) for value in (move.get(side) or {}).get("atoms") or ()
            )
    return {value for value in touched if value > 0}


def bind_reactive_import(
    program: ReactiveFragmentProgram,
    assigned_maps: Sequence[int],
) -> ReactiveImportGuard:
    """Bind executor-assigned maps to a map-free generated fragment program."""

    maps = tuple(int(value) for value in assigned_maps)
    if len(maps) != len(program.atoms):
        raise ValueError("executor map count does not match generated fragment")
    if any(value <= 0 for value in maps) or len(set(maps)) != len(maps):
        raise ValueError("executor maps must be unique positive integers")
    return ReactiveImportGuard(tuple(maps[index] for index in program.active_atoms))


def _fragment_maps(smiles: str) -> tuple[int, ...]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError(f"invalid imported fragment: {smiles!r}")
    maps = tuple(int(atom.GetAtomMapNum()) for atom in mol.GetAtoms())
    if any(value <= 0 for value in maps) or len(set(maps)) != len(maps):
        raise ValueError("imported fragment atoms require unique private maps")
    return maps


def electron_participating_maps(plan: Mapping[str, Any]) -> set[int]:
    maps: set[int] = set()
    for step in plan.get("steps") or ():
        for move in step.get("moves") or ():
            if move.get("mode") == "BE_DELTA":
                for item in move.get("bond_deltas") or ():
                    maps.update(int(value) for value in item.get("atoms") or ())
                for item in move.get("charge_actions") or ():
                    atom_map = int(item.get("atom_map") or 0)
                    if atom_map > 0:
                        maps.add(atom_map)
                continue
            for side in ("source", "sink"):
                maps.update(
                    int(value)
                    for value in (move.get(side) or {}).get("atoms") or ()
                )
    return maps


def infer_reactive_role(plan: Mapping[str, Any], fragment_maps: Sequence[int]) -> str:
    wanted = set(int(value) for value in fragment_maps)
    # Source-side evidence is more informative than membership of a new bond,
    # where both the donor and acceptor occur in the sink container.
    for step in plan.get("steps") or ():
        for move in step.get("moves") or ():
            if move.get("mode") == "BE_DELTA":
                for item in move.get("bond_deltas") or ():
                    atoms = set(int(value) for value in item.get("atoms") or ())
                    if not atoms & wanted:
                        continue
                    return "BOND_DONOR" if int(item.get("delta") or 0) > 0 else "LEAVING_GROUP"
                for item in move.get("charge_actions") or ():
                    if int(item.get("atom_map") or 0) in wanted:
                        return "REDOX"
                continue
            source = move.get("source") or {}
            source_atoms = set(int(value) for value in source.get("atoms") or ())
            if source_atoms & wanted:
                kind = str(source.get("kind") or "")
                if kind == "LP":
                    return "NUCLEOPHILE"
                if kind == "RADICAL_PAIR":
                    return "RADICAL"
                if kind == "BOND":
                    # Hydrogen-containing bond donors are overwhelmingly proton
                    # transfer events; other bonds donate an electron pair.
                    return "BOND_DONOR"
            sink = move.get("sink") or {}
            sink_atoms = set(int(value) for value in sink.get("atoms") or ())
            if sink_atoms & wanted:
                return "ELECTROPHILE"
    return "OTHER_REACTIVE"


def classify_imports(plan: Mapping[str, Any]) -> tuple[ImportSupervision, ...]:
    participating = electron_participating_maps(plan)
    fragments = list(plan.get("initial_imports") or ())
    for step in plan.get("steps") or ():
        fragments.extend(step.get("imports") or ())
    rows: list[ImportSupervision] = []
    for fragment in fragments:
        maps = _fragment_maps(str(fragment))
        active = tuple(sorted(set(maps) & participating))
        rows.append(
            ImportSupervision(
                fragment=str(fragment),
                kind="IMPORT_REACTIVE" if active else "IMPORT_ENV",
                participating_maps=active,
                role=infer_reactive_role(plan, active) if active else None,
            )
        )
    return tuple(rows)


def _bond_type(value: Chem.Bond) -> int:
    # Aromatic graphs are Kekulized before compilation, leaving integer bond
    # orders for ordinary organic fragments while retaining uncommon RDKit bond
    # enums (e.g. dative) without collapsing their identity.
    return int(value.GetBondType())


def _canonical_connected_order(mol: Chem.Mol) -> tuple[tuple[int, ...], dict[int, int | None]]:
    ranks = tuple(int(value) for value in Chem.CanonicalRankAtoms(mol, breakTies=True))
    unvisited = set(range(mol.GetNumAtoms()))
    order: list[int] = []
    parents: dict[int, int | None] = {}
    while unvisited:
        root = min(unvisited, key=lambda index: (ranks[index], index))
        queue = [root]
        parents[root] = None
        unvisited.remove(root)
        while queue:
            atom_index = queue.pop(0)
            order.append(atom_index)
            neighbors = sorted(
                (
                    neighbor.GetIdx()
                    for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
                    if neighbor.GetIdx() in unvisited
                ),
                key=lambda index: (ranks[index], index),
            )
            for neighbor in neighbors:
                if neighbor not in unvisited:
                    continue
                unvisited.remove(neighbor)
                parents[neighbor] = atom_index
                queue.append(neighbor)
    return tuple(order), parents


def decompose_reactive_fragment(
    mapped_smiles: str,
    *,
    participating_maps: Sequence[int],
    role: str,
) -> ReactiveFragmentProgram:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    source = Chem.MolFromSmiles(str(mapped_smiles or ""), params)
    if source is None:
        raise ValueError(f"invalid reactive fragment: {mapped_smiles!r}")
    source_maps = tuple(int(atom.GetAtomMapNum()) for atom in source.GetAtoms())
    active_source_indices = {
        index
        for index, atom_map in enumerate(source_maps)
        if atom_map in set(int(value) for value in participating_maps)
    }
    if not active_source_indices:
        raise ValueError("reactive fragment has no participating atom")
    work = Chem.Mol(source)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(0)
    source_unmapped = Chem.MolToSmiles(work, canonical=True, isomericSmiles=True)
    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:
        pass
    order, parent_by_source = _canonical_connected_order(work)
    program_index = {source_index: index for index, source_index in enumerate(order)}
    tree_edges: set[tuple[int, int]] = set()
    atoms: list[FragmentAtom] = []
    for source_index in order:
        atom = work.GetAtomWithIdx(source_index)
        parent_source = parent_by_source[source_index]
        parent = program_index[parent_source] if parent_source is not None else None
        parent_bond_type = None
        if parent_source is not None:
            bond = work.GetBondBetweenAtoms(source_index, parent_source)
            if bond is None:
                raise ValueError("fragment traversal parent lacks bond")
            parent_bond_type = _bond_type(bond)
            tree_edges.add(tuple(sorted((source_index, parent_source))))
        atoms.append(
            FragmentAtom(
                atomic_num=int(atom.GetAtomicNum()),
                formal_charge=int(atom.GetFormalCharge()),
                isotope=int(atom.GetIsotope()),
                explicit_h=int(atom.GetNumExplicitHs()),
                no_implicit=bool(atom.GetNoImplicit()),
                radical_electrons=int(atom.GetNumRadicalElectrons()),
                chiral_tag=int(atom.GetChiralTag()),
                parent=parent,
                parent_bond_type=parent_bond_type,
            )
        )
    extra: list[FragmentBond] = []
    for bond in work.GetBonds():
        source_pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if source_pair in tree_edges:
            continue
        stereo_atoms = tuple(program_index[int(value)] for value in bond.GetStereoAtoms())
        extra.append(
            FragmentBond(
                atoms=tuple(sorted((program_index[source_pair[0]], program_index[source_pair[1]]))),
                bond_type=_bond_type(bond),
                stereo=int(bond.GetStereo()),
                stereo_atoms=stereo_atoms if len(stereo_atoms) == 2 else (),
                bond_dir=int(bond.GetBondDir()),
            )
        )
    extra.sort(key=lambda item: (item.atoms, item.bond_type))
    active = tuple(sorted(program_index[index] for index in active_source_indices))
    return ReactiveFragmentProgram(
        role=role,
        atoms=tuple(atoms),
        extra_bonds=tuple(extra),
        active_atoms=active,
        source_unmapped_smiles=source_unmapped,
    )


def replay_reactive_fragment(program: ReactiveFragmentProgram) -> str:
    rw = Chem.RWMol()
    for index, spec in enumerate(program.atoms):
        atom = Chem.Atom(int(spec.atomic_num))
        atom.SetFormalCharge(int(spec.formal_charge))
        atom.SetIsotope(int(spec.isotope))
        atom.SetNumExplicitHs(int(spec.explicit_h))
        atom.SetNoImplicit(bool(spec.no_implicit))
        atom.SetNumRadicalElectrons(int(spec.radical_electrons))
        atom.SetChiralTag(Chem.ChiralType(int(spec.chiral_tag)))
        added = rw.AddAtom(atom)
        if added != index:
            raise ValueError("unexpected RDKit atom insertion order")
        if spec.parent is not None:
            rw.AddBond(index, int(spec.parent), Chem.BondType(int(spec.parent_bond_type)))
    for spec in program.extra_bonds:
        left, right = spec.atoms
        rw.AddBond(int(left), int(right), Chem.BondType(int(spec.bond_type)))
        bond = rw.GetBondBetweenAtoms(int(left), int(right))
        if bond is not None:
            bond.SetBondDir(Chem.BondDir(int(spec.bond_dir)))
            if spec.stereo_atoms:
                bond.SetStereoAtoms(*map(int, spec.stereo_atoms))
            bond.SetStereo(Chem.BondStereo(int(spec.stereo)))
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
