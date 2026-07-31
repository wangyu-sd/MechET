"""Verified proof-equivalence transformations used for data augmentation."""
from __future__ import annotations

import copy
import hashlib
import random

from mechet.map_invariance import record_map_permutation, remap_proof_text
from mechet.proof_equivalence import proofs_equivalent
from mechet.proof_program import (
    ProofProgram,
    ProofProgramError,
    execute_proof,
    format_proof_output,
    parse_proof_program,
)


def _state_ids(program: ProofProgram) -> list[str]:
    values = set(program.roots)
    values.add(program.precursor_state_id)
    for edge in program.edges:
        values.update((edge.src, edge.dst))
    return sorted(values)


def rename_states(program_or_text: ProofProgram | str, *, seed: int) -> str:
    program = copy.deepcopy(
        parse_proof_program(program_or_text)
        if isinstance(program_or_text, str)
        else program_or_text
    )
    ids = _state_ids(program)
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    mapping = {old: f"q{index}" for index, old in enumerate(shuffled)}
    program.roots = {mapping[key]: list(value) for key, value in program.roots.items()}
    program.precursor_state_id = mapping[program.precursor_state_id]
    for edge in program.edges:
        edge.src = mapping[edge.src]
        edge.dst = mapping[edge.dst]
    return format_proof_output(program)


def reorder_edges(program_or_text: ProofProgram | str, *, seed: int) -> str:
    program = copy.deepcopy(
        parse_proof_program(program_or_text)
        if isinstance(program_or_text, str)
        else program_or_text
    )
    random.Random(seed).shuffle(program.edges)
    return format_proof_output(program)


def build_equivalent_variants(
    proof_text: str,
    *,
    n_variants: int = 4,
    seed: int = 0,
) -> list[str]:
    reference = format_proof_output(parse_proof_program(proof_text))
    execution = execute_proof(reference)
    if not execution.ok:
        raise ProofProgramError("reference proof is not executable")
    variants = [reference]
    seen = {hashlib.sha256(reference.encode("utf-8")).hexdigest()}
    attempts = max(8, n_variants * 8)
    for index in range(attempts):
        local_seed = seed + 104729 * (index + 1)
        candidate = rename_states(reference, seed=local_seed)
        candidate = reorder_edges(candidate, seed=local_seed + 1)
        program = parse_proof_program(candidate)
        interim = execute_proof(program)
        if not interim.ok:
            continue
        mapping = record_map_permutation(
            product=program.target_smiles,
            proof=candidate,
            precursor=interim.precursor_smiles,
            seed=local_seed + 2,
        )
        candidate = remap_proof_text(candidate, mapping)
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        if execute_proof(candidate).ok and proofs_equivalent(reference, candidate):
            variants.append(candidate)
            seen.add(digest)
        if len(variants) >= n_variants:
            break
    return variants
