"""GT-independent, factorized action space for electron-flow events.

The current molecular state determines a finite inventory of electron sources.
After one source is selected, topological electron-flow rules determine the
compatible sinks.  This avoids enumerating the Cartesian product of complete
multi-arrow events.  The resulting choices are *admissible prefixes*; a full
event is formally legal only after :func:`verify_electron_step` accepts the
coupled arrows atomically.

No function in this module accepts a product answer, proof, or gold event.
Gold moves are used only by the separate audit helper to measure whether the
state-derived action space contains the reference decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .forward_expert import ElectronContainer, ElectronMove, _lp_electrons


SUPPORTED_POLAR_SIGNATURES = frozenset(
    {
        ("LP", "BOND"),
        ("BOND", "ATOM"),
        ("BOND", "BOND"),
        ("RADICAL_PAIR", "BOND"),
        ("BOND", "RADICAL_PAIR"),
    }
)


def _is_transition_metal(atomic_number: int) -> bool:
    return (
        21 <= atomic_number <= 30
        or 39 <= atomic_number <= 48
        or 57 <= atomic_number <= 80
        or 89 <= atomic_number <= 112
    )


def _radical_pair_candidate(left: Chem.Atom, right: Chem.Atom) -> bool:
    """Conservative GT-independent support for corpus radical-pair actions.

    FlowER uses RADICAL_PAIR both for metal-ligand coordination and a small
    number of halogen transfer events.  Keeping this as a separate family
    prevents quadratic all-atom radical enumeration in ordinary polar steps.
    """

    z_left, z_right = left.GetAtomicNum(), right.GetAtomicNum()
    halogens = {9, 17, 35, 53}
    partners = {6, 7, 8, 15, 16}
    metal_partners = partners | halogens
    if _is_transition_metal(z_left):
        return z_right in metal_partners
    if _is_transition_metal(z_right):
        return z_left in metal_partners
    return (z_left in halogens and z_right in partners) or (
        z_right in halogens and z_left in partners
    )


def _unique(values: Sequence[ElectronContainer]) -> tuple[ElectronContainer, ...]:
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class MoveInventory:
    """One state-derived source inventory and its conditional sink operator."""

    state_smiles: str
    sources: tuple[ElectronContainer, ...]
    polar_sources: tuple[ElectronContainer, ...]
    radical_sources: tuple[ElectronContainer, ...]
    atom_sinks: tuple[ElectronContainer, ...]
    bond_sinks: tuple[ElectronContainer, ...]

    @classmethod
    def from_state(cls, state_smiles: str) -> "MoveInventory":
        # The executor applies arrows on a Kekule graph.  Inventory construction
        # must use the same bond-order representation; otherwise pyridine-like
        # aromatic nitrogens lose a lone-pair candidate because Python rounds
        # aromatic bond order 1.5 while the executor sees alternating 1/2 bonds.
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(state_smiles, params)
        if mol is None:
            raise ValueError(f"invalid molecular state: {state_smiles!r}")
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        atoms = list(mol.GetAtoms())
        polar_sources: list[ElectronContainer] = []
        radical_sources: list[ElectronContainer] = []
        sinks: list[ElectronContainer] = []
        for atom in atoms:
            atom_map = atom.GetAtomMapNum()
            if atom_map <= 0:
                raise ValueError("all atoms require positive private maps")
            sinks.append(ElectronContainer("ATOM", (atom_map,)))
            if _lp_electrons(atom) >= 2:
                polar_sources.append(ElectronContainer("LP", (atom_map,)))
        for bond in mol.GetBonds():
            pair = (
                bond.GetBeginAtom().GetAtomMapNum(),
                bond.GetEndAtom().GetAtomMapNum(),
            )
            polar_sources.append(ElectronContainer("BOND", pair))
            if round(bond.GetBondTypeAsDouble()) < 3:
                sinks.append(ElectronContainer("BOND", pair))
        for index, left in enumerate(atoms):
            for right in atoms[index + 1 :]:
                if mol.GetBondBetweenAtoms(left.GetIdx(), right.GetIdx()) is None:
                    sinks.append(
                        ElectronContainer(
                            "BOND", (left.GetAtomMapNum(), right.GetAtomMapNum())
                        )
                    )
                if _radical_pair_candidate(left, right):
                    radical_sources.append(
                        ElectronContainer(
                            "RADICAL_PAIR",
                            (left.GetAtomMapNum(), right.GetAtomMapNum()),
                        )
                    )
        return cls(
            state_smiles=state_smiles,
            sources=_unique(tuple(polar_sources + radical_sources)),
            polar_sources=_unique(tuple(polar_sources)),
            radical_sources=_unique(tuple(radical_sources)),
            atom_sinks=_unique(tuple(item for item in sinks if item.kind == "ATOM")),
            bond_sinks=_unique(tuple(item for item in sinks if item.kind == "BOND")),
        )

    def compatible_sinks(
        self, source: ElectronContainer
    ) -> tuple[ElectronContainer, ...]:
        """Return sinks allowed by electron-container topology.

        These are deliberately not claimed to be chemically correct.  The
        coupled event still has to pass the deterministic executor.
        """

        if source.kind == "LP":
            donor = source.atoms[0]
            return _unique(
                tuple(item for item in self.bond_sinks if donor in item.atoms)
            )
        if source.kind == "BOND":
            endpoints = set(source.atoms)
            cleavage = tuple(
                item for item in self.atom_sinks if item.atoms[0] in endpoints
            )
            shifts = tuple(
                item
                for item in self.bond_sinks
                if item != source and len(endpoints & set(item.atoms)) == 1
            )
            radical = (ElectronContainer("RADICAL_PAIR", source.atoms),)
            return _unique(cleavage + shifts + radical)
        if source.kind == "RADICAL_PAIR":
            pair = ElectronContainer("BOND", source.atoms)
            return (pair,) if pair in self.bond_sinks else ()
        return ()

    def contains(self, move: ElectronMove | Mapping[str, Any]) -> bool:
        parsed = move if isinstance(move, ElectronMove) else ElectronMove.parse(move)
        return parsed.source in self.sources and parsed.sink in self.compatible_sinks(
            parsed.source
        )


def audit_reference_event(
    state_smiles: str, moves: Sequence[ElectronMove | Mapping[str, Any]]
) -> dict[str, Any]:
    """Measure reference coverage without using it to construct candidates."""

    if any(isinstance(item, Mapping) and item.get("mode") == "BE_DELTA" for item in moves):
        return {
            "covered": False,
            "reason": "BE_DELTA_OUTSIDE_POLAR_SPACE",
            "move_coverage": [],
            "source_candidates": 0,
            "polar_source_candidates": 0,
            "radical_source_candidates": 0,
            "gold_conditioned_sink_candidates": [],
        }
    inventory = MoveInventory.from_state(state_smiles)
    parsed = [item if isinstance(item, ElectronMove) else ElectronMove.parse(item) for item in moves]
    rows = []
    sink_counts = []
    for move in parsed:
        sinks = inventory.compatible_sinks(move.source)
        signature = (move.source.kind, move.sink.kind)
        supported = signature in SUPPORTED_POLAR_SIGNATURES
        source_covered = move.source in inventory.sources
        sink_covered = move.sink in sinks
        rows.append(
            {
                "signature": list(signature),
                "supported": supported,
                "source_covered": source_covered,
                "sink_covered": sink_covered,
                "covered": supported and source_covered and sink_covered,
            }
        )
        sink_counts.append(len(sinks))
    return {
        "covered": bool(rows) and all(item["covered"] for item in rows),
        "reason": "PASS" if rows and all(item["covered"] for item in rows) else "MOVE_OUTSIDE_POLAR_SPACE",
        "move_coverage": rows,
        "source_candidates": len(inventory.sources),
        "polar_source_candidates": len(inventory.polar_sources),
        "radical_source_candidates": len(inventory.radical_sources),
        "gold_conditioned_sink_candidates": sink_counts,
    }
