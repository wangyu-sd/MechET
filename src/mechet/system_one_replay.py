"""Executor replay of System-One pair decisions at visible current states.

Temporary atom maps are reconstructed from the public annotated SMILES solely
for the executor. They are never fed to the decision policy. The reconstructed
inventory must preserve each alias's graph address and chemical stereochemistry;
byte identity is preferred but RDKit can normalize redundant stereo symbols.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from rdkit import Chem

from .electron_pointer import PointerObservation, candidate_keys
from .natural_language_electron_flow import build_inventory, execute_event_arguments
from .structural_overlap import canonical_unmapped_smiles


def _alias_graph_key(smiles: str) -> tuple:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("invalid visible molecular state")
    atoms = tuple(
        (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
         atom.GetNumRadicalElectrons(), atom.GetIsAromatic())
        for atom in mol.GetAtoms()
    )
    bonds = tuple(sorted(
        (min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
         max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
         str(bond.GetBondType()), bond.GetIsAromatic())
        for bond in mol.GetBonds()
    ))
    return atoms, bonds


def reconstruct_mapped_state(observation: PointerObservation) -> str:
    visible = re.sub(r"<A\d+>", "", observation.annotated)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(visible, params)
    if mol is None or mol.GetNumAtoms() != len(observation.atom_names):
        raise ValueError(f"{observation.row_id}: visible atom inventory does not parse")
    for index, atom in enumerate(mol.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    mapped = Chem.MolToSmiles(mol, canonical=False)
    rebuilt = build_inventory(mapped).prompt.split("ANNOTATED CURRENT STATE: ", 1)[1]
    if rebuilt != observation.annotated:
        rebuilt_names = tuple(re.findall(r"<A\d+>", rebuilt))
        if (
            rebuilt_names != tuple(f"<{name}>" for name in observation.atom_names)
            or _alias_graph_key(re.sub(r"<A\d+>", "", rebuilt)) != _alias_graph_key(visible)
            or canonical_unmapped_smiles(re.sub(r"<A\d+>", "", rebuilt))
            != canonical_unmapped_smiles(visible)
        ):
            raise ValueError(
                f"{observation.row_id}: reconstructed inventory changes an atom address or chemistry"
            )
    return mapped


def pair_indices_to_arguments(
    observation: PointerObservation, flat_indices: Sequence[int]
) -> dict:
    """Turn ranked coupled pair IDs into an inference-time event, without GT."""
    source = candidate_keys(len(observation.atom_names), observation.bonds, source=True)
    sink = candidate_keys(len(observation.atom_names), observation.bonds, source=False)
    sink_count = len(sink)
    if not flat_indices or len(set(flat_indices)) != len(flat_indices):
        raise ValueError("at least one unique source/sink pair is required")
    flows = []
    for flat in flat_indices:
        if not 0 <= int(flat) < len(source) * sink_count:
            raise ValueError(f"source/sink pair outside current candidate space: {flat}")
        src = source[int(flat) // sink_count]
        dst = sink[int(flat) % sink_count]
        if src[0] == "atom":
            src_text = f"a lone pair on atom {observation.atom_names[src[1]]}"
        else:
            src_text = (
                f"the bond between atoms {observation.atom_names[src[1]]} "
                f"and {observation.atom_names[src[2]]}"
            )
        if dst[0] == "atom":
            dst_text = f"atom {observation.atom_names[dst[1]]}"
        else:
            prefix = (
                "the bond between atoms" if (dst[1], dst[2]) in observation.bonds
                else "the bond to form between atoms"
            )
            dst_text = (
                f"{prefix} {observation.atom_names[dst[1]]} "
                f"and {observation.atom_names[dst[2]]}"
            )
        flows.append({"source": src_text, "destination": dst_text})
    return {
        "direction": "retrosynthetic",
        "electron_flow": flows,
        "bond_order_changes": [],
        "charge_changes": [],
    }


def execute_pair_indices(
    mapped_state: str, observation: PointerObservation, flat_indices: Sequence[int]
) -> dict:
    return execute_event_arguments(
        mapped_state, pair_indices_to_arguments(observation, flat_indices)
    )
