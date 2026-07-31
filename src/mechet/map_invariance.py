"""Atom-map permutation utilities for leakage and invariance controls."""
from __future__ import annotations

import random
import re
from typing import Iterable, Mapping

from rdkit import Chem

_MAP_PATTERN = re.compile(r"(?<=:)(\d+)(?=\])")
_ACTION_PATTERN = re.compile(r"^(\s*(?:BOND|LP|CHARGE)\s+)(.*)$")


def map_numbers_in_smiles(smiles: str) -> list[int]:
    values = {int(value) for value in _MAP_PATTERN.findall(smiles or "")}
    if values:
        return sorted(values)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles[:100]}")
    return sorted({atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0})


def map_numbers_in_proof(text: str) -> list[int]:
    values = {int(value) for value in _MAP_PATTERN.findall(text or "")}
    for raw in (text or "").splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        if parts[0] == "BOND" and len(parts) >= 3:
            values.update((int(parts[1]), int(parts[2])))
        elif parts[0] in {"LP", "CHARGE"} and len(parts) >= 2:
            values.add(int(parts[1]))
    return sorted(values)


def random_map_permutation_from_values(values: Iterable[int], *, seed: int) -> dict[int, int]:
    ordered = sorted({int(value) for value in values if int(value) > 0})
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    return dict(zip(ordered, shuffled))


def random_map_permutation(smiles: str, *, seed: int) -> dict[int, int]:
    return random_map_permutation_from_values(map_numbers_in_smiles(smiles), seed=seed)


def record_map_permutation(
    *,
    product: str,
    proof: str = "",
    precursor: str = "",
    seed: int,
) -> dict[int, int]:
    values = set(map_numbers_in_smiles(product))
    values.update(map_numbers_in_proof(proof))
    if precursor:
        values.update(map_numbers_in_smiles(precursor))
    return random_map_permutation_from_values(values, seed=seed)


def remap_smiles(smiles: str, mapping: Mapping[int, int]) -> str:
    def repl(match: re.Match[str]) -> str:
        old = int(match.group(1))
        return str(mapping.get(old, old))

    return _MAP_PATTERN.sub(repl, smiles)


def remap_proof_text(text: str, mapping: Mapping[int, int]) -> str:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if stripped.startswith(("TARGET_SMILES ", "IMPORT ")):
            lines.append(remap_smiles(raw, mapping))
            continue
        match = _ACTION_PATTERN.match(raw)
        if not match:
            lines.append(raw)
            continue
        prefix, payload = match.groups()
        tokens = payload.split()
        if stripped.startswith("BOND ") and len(tokens) >= 3:
            tokens[0] = str(mapping.get(int(tokens[0]), int(tokens[0])))
            tokens[1] = str(mapping.get(int(tokens[1]), int(tokens[1])))
        elif stripped.startswith(("LP ", "CHARGE ")) and tokens:
            tokens[0] = str(mapping.get(int(tokens[0]), int(tokens[0])))
        lines.append(prefix + " ".join(tokens))
    return "\n".join(lines)
