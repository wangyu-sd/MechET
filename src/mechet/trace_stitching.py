"""Stitch independently mapped elementary steps into globally mapped traces."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .forward_expert import ElectronMove, verify_electron_step


def complete_reaction_ids(
    raw_reaction_ids: Sequence[str | int],
    standardized_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, int | str]]:
    """Select reactions for which every raw elementary step was standardized."""
    expected = Counter(str(value) for value in raw_reaction_ids)
    observed: Counter[str] = Counter()
    for row in standardized_rows:
        observed[str(row.get("id") or "")] += len(row.get("steps") or [])
    complete = [
        {
            "reaction_id": reaction_id,
            "expected_steps": expected_steps,
            "standardized_steps": observed[reaction_id],
        }
        for reaction_id, expected_steps in expected.items()
        if observed[reaction_id] == expected_steps
    ]
    return sorted(
        complete,
        key=lambda value: (
            0,
            int(str(value["reaction_id"])),
        )
        if str(value["reaction_id"]).isdigit()
        else (1, str(value["reaction_id"])),
    )


def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("invalid mapped SMILES")
    return mol


def _mapped_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _without_maps(mol: Chem.Mol) -> Chem.Mol:
    value = Chem.Mol(mol)
    for atom in value.GetAtoms():
        atom.SetAtomMapNum(0)
    return value


def _same_graph(left_smiles: str, right_smiles: str) -> bool:
    """Compare constitution/formal state while tolerating omitted stereo tags."""
    left = _without_maps(_mol(left_smiles))
    right = _without_maps(_mol(right_smiles))
    if left.GetNumAtoms() != right.GetNumAtoms():
        return False
    matches = left.GetSubstructMatches(
        right, uniquify=True, useChirality=False, maxMatches=1
    )
    return bool(matches and len(matches[0]) == right.GetNumAtoms())


def map_correspondence(
    reference_smiles: str,
    local_smiles: str,
    *,
    max_matches: int = 4096,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Map local atom-map labels onto an isomorphic reference state."""
    reference = _mol(reference_smiles)
    local = _mol(local_smiles)
    if reference.GetNumAtoms() != local.GetNumAtoms():
        raise ValueError("adjacent states have different atom counts")
    reference_maps = [atom.GetAtomMapNum() for atom in reference.GetAtoms()]
    local_maps = [atom.GetAtomMapNum() for atom in local.GetAtoms()]
    if not reference_maps or 0 in reference_maps or 0 in local_maps:
        raise ValueError("stitching requires every atom to have a map label")
    if len(set(reference_maps)) != len(reference_maps) or len(set(local_maps)) != len(local_maps):
        raise ValueError("atom-map labels must be unique within each state")

    matches = _without_maps(reference).GetSubstructMatches(
        _without_maps(local),
        uniquify=False,
        useChirality=True,
        maxMatches=max_matches,
    )
    full = [match for match in matches if len(match) == local.GetNumAtoms()]
    if not full:
        raise ValueError("adjacent mapped states are not graph-isomorphic")
    if len(full) >= max_matches:
        raise ValueError(f"atom correspondence exceeds max_matches={max_matches}")

    local_order = sorted(range(len(local_maps)), key=local_maps.__getitem__)
    chosen = min(full, key=lambda match: tuple(reference_maps[match[i]] for i in local_order))
    mapping = {
        local_maps[local_index]: reference_maps[reference_index]
        for local_index, reference_index in enumerate(chosen)
    }
    return mapping, {
        "isomorphism_matches": len(full),
        "ambiguous": len(full) > 1,
    }


def remap_smiles(smiles: str, mapping: Mapping[int, int]) -> str:
    mol = _mol(smiles)
    for atom in mol.GetAtoms():
        old = atom.GetAtomMapNum()
        if old not in mapping:
            raise ValueError(f"missing remap for atom {old}")
        atom.SetAtomMapNum(int(mapping[old]))
    return _mapped_smiles(mol)


def remap_moves(
    moves: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, int],
) -> list[dict[str, Any]]:
    output = deepcopy(list(moves))
    for move in output:
        for side in ("source", "sink"):
            atoms = move[side]["atoms"]
            move[side]["atoms"] = [int(mapping[int(value)]) for value in atoms]
    # Rebuild all cached container/move IDs after changing atom labels.
    return [ElectronMove.parse(move).to_dict() for move in output]


def stitch_steps(
    steps: Sequence[Mapping[str, Any]],
    *,
    max_matches: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a replay-verified trace with one atom-map namespace."""
    ordered = sorted(steps, key=lambda value: int(value.get("step_index", 0)))
    if not ordered:
        raise ValueError("trace contains no steps")
    stitched: list[dict[str, Any]] = []
    link_metadata: list[dict[str, Any]] = []
    previous_target = ""
    for position, raw in enumerate(ordered):
        step = deepcopy(dict(raw))
        if position:
            mapping, metadata = map_correspondence(
                previous_target,
                str(step["state_smiles"]),
                max_matches=max_matches,
            )
            local_state = remap_smiles(str(step["state_smiles"]), mapping)
            expected_target = remap_smiles(str(step["target_product"]), mapping)
            step["moves"] = remap_moves(step.get("moves") or [], mapping)
            if not _same_graph(local_state, previous_target):
                raise ValueError("remapped adjacent state is not the previous target graph")
            # The environment-owned prior target is the causal next state.  This
            # also preserves stereo that an independently encoded row omitted.
            step["state_smiles"] = previous_target
            link_metadata.append(metadata)
        else:
            step["state_smiles"] = _mapped_smiles(_mol(str(step["state_smiles"])))
            expected_target = _mapped_smiles(_mol(str(step["target_product"])))

        replay = verify_electron_step(step["state_smiles"], step.get("moves") or [])
        if not replay.get("ok"):
            raise ValueError(
                "stitched step replay failed: "
                + str(replay.get("message") or replay.get("code"))
            )
        replay_target = _mapped_smiles(_mol(str(replay["state_smiles"])))
        if not _same_graph(replay_target, expected_target):
            raise ValueError("stitched replay target mismatch")
        step["target_product"] = replay_target
        stitched.append(step)
        previous_target = step["target_product"]
    return stitched, {
        "links": len(link_metadata),
        "ambiguous_links": sum(bool(value["ambiguous"]) for value in link_metadata),
        "max_isomorphism_matches": max(
            (int(value["isomorphism_matches"]) for value in link_metadata),
            default=1,
        ),
    }
