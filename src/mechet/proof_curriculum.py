"""Proof-centric curriculum data for equivalence, falsification, preference, and repair.

Every negative example is validated by the deterministic executor. Executable
proofs with alternative endpoints are never labelled as negative by default.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Any, Iterable

from mechet.map_invariance import (
    record_map_permutation,
    remap_proof_text,
)
from mechet.proof_diagnostics import diagnose_proof, format_repair_feedback
from mechet.proof_equivalence import (
    canonical_partial_order_signature,
    edge_touched_maps,
    proofs_equivalent,
)
from mechet.proof_program import (
    ProofEdge,
    ProofProgram,
    ProofProgramError,
    execute_proof,
    extract_proof_body,
    format_proof_output,
    parse_proof_program,
)


@dataclass(frozen=True)
class ProofCorruption:
    source_id: str
    corruption_type: str
    valid_proof: str
    corrupted_proof: str
    expected_execute: bool
    observed_execute: bool
    failure_code: str = ""
    failure_edge: str = ""
    repairable: bool = False
    changed_lines: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def proof_text_from_row(row: dict[str, Any]) -> str:
    """Extract the assistant proof from a standard chat row."""
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            text = str(message.get("content") or "")
            if extract_proof_body(text):
                return text
    return str(row.get("proof") or row.get("prediction") or "")


def _stable_digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _state_ids(program: ProofProgram) -> list[str]:
    ids = set(program.roots)
    ids.add(program.precursor_state_id)
    for edge in program.edges:
        ids.update((edge.src, edge.dst))
    return sorted(ids)


def rename_states(program_or_text: ProofProgram | str, *, seed: int) -> str:
    """Rename all internal state identifiers while preserving proof semantics."""
    program = copy.deepcopy(
        parse_proof_program(program_or_text)
        if isinstance(program_or_text, str)
        else program_or_text
    )
    ids = _state_ids(program)
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    mapping = {old: f"q{index}_{name}" for index, name in enumerate(shuffled)}
    program.roots = {mapping[key]: list(value) for key, value in program.roots.items()}
    program.precursor_state_id = mapping[program.precursor_state_id]
    for edge in program.edges:
        edge.src = mapping[edge.src]
        edge.dst = mapping[edge.dst]
    return format_proof_output(program)


def reorder_edges(program_or_text: ProofProgram | str, *, seed: int) -> str:
    """Change only textual edge order; execution still follows dependencies."""
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
    """Generate verified serialization/map variants of one executable proof."""
    reference = format_proof_output(parse_proof_program(proof_text))
    if not execute_proof(reference).ok:
        raise ProofProgramError("reference proof is not executable")
    variants: list[str] = [reference]
    seen = {_stable_digest(reference)}
    for index in range(max(n_variants - 1, 0) * 4 + 4):
        candidate = reference
        local_seed = seed + 104729 * (index + 1)
        candidate = rename_states(candidate, seed=local_seed)
        candidate = reorder_edges(candidate, seed=local_seed + 1)
        program = parse_proof_program(candidate)
        execution = execute_proof(program)
        precursor = execution.precursor_smiles if execution.ok else ""
        mapping = record_map_permutation(
            product=program.target_smiles,
            proof=candidate,
            precursor=precursor,
            seed=local_seed + 2,
        )
        candidate = remap_proof_text(candidate, mapping)
        digest = _stable_digest(candidate)
        if digest in seen:
            continue
        result = execute_proof(candidate)
        if result.ok and proofs_equivalent(reference, candidate):
            variants.append(candidate)
            seen.add(digest)
        if len(variants) >= n_variants:
            break
    return variants


def _format_changed(lines_before: Iterable[str], lines_after: Iterable[str]) -> tuple[str, ...]:
    before = list(lines_before)
    after = list(lines_after)
    out: list[str] = []
    width = max(len(before), len(after))
    for index in range(width):
        left = before[index] if index < len(before) else ""
        right = after[index] if index < len(after) else ""
        if left != right:
            out.append(right or f"<DELETE:{left}>")
    return tuple(out)


def _first_edge_with(program: ProofProgram, field: str) -> ProofEdge | None:
    for edge in program.edges:
        if getattr(edge, field):
            return edge
    return None


def _corrupt_program(program: ProofProgram, corruption_type: str, seed: int) -> ProofProgram:
    rng = random.Random(seed)
    out = copy.deepcopy(program)
    if corruption_type == "LP_WRONG_DELTA":
        edge = _first_edge_with(out, "lone_pairs")
        if edge is None:
            raise ValueError("proof has no LP action")
        index = rng.randrange(len(edge.lone_pairs))
        atom_map, delta = edge.lone_pairs[index]
        edge.lone_pairs[index] = (atom_map, delta + (2 if delta >= 0 else -2))
    elif corruption_type == "LP_DELETE":
        edge = _first_edge_with(out, "lone_pairs")
        if edge is None:
            raise ValueError("proof has no LP action")
        del edge.lone_pairs[rng.randrange(len(edge.lone_pairs))]
    elif corruption_type == "BOND_DELETE":
        edge = _first_edge_with(out, "bonds")
        if edge is None:
            raise ValueError("proof has no BOND action")
        del edge.bonds[rng.randrange(len(edge.bonds))]
    elif corruption_type == "BOND_WRONG_DELTA":
        edge = _first_edge_with(out, "bonds")
        if edge is None:
            raise ValueError("proof has no BOND action")
        index = rng.randrange(len(edge.bonds))
        i, j, delta = edge.bonds[index]
        edge.bonds[index] = (i, j, -delta if delta else 1)
    elif corruption_type == "CHARGE_WRONG_PRECONDITION":
        edge = _first_edge_with(out, "charges")
        if edge is None:
            raise ValueError("proof has no CHARGE action")
        index = rng.randrange(len(edge.charges))
        action = edge.charges[index]
        edge.charges[index] = type(action)(action.atom_map, action.q0 + 1, action.q1)
    elif corruption_type == "CHARGE_DELETE":
        edge = _first_edge_with(out, "charges")
        if edge is None:
            raise ValueError("proof has no CHARGE action")
        del edge.charges[rng.randrange(len(edge.charges))]
    elif corruption_type == "IMPORT_DELETE":
        locations: list[tuple[str, Any]] = []
        locations.extend(("root", key) for key, value in out.roots.items() if value)
        locations.extend(("edge", edge) for edge in out.edges if edge.imports)
        if not locations:
            raise ValueError("proof has no IMPORT")
        kind, location = rng.choice(locations)
        if kind == "root":
            out.roots[location].pop(0)
        else:
            location.imports.pop(0)
    elif corruption_type == "ATOM_MAP_REPLACE":
        edge = _first_edge_with(out, "bonds") or _first_edge_with(out, "lone_pairs") or _first_edge_with(out, "charges")
        if edge is None:
            raise ValueError("proof has no mapped action")
        all_maps = set()
        for item in out.edges:
            all_maps.update(edge_touched_maps(item))
        missing = max(all_maps or {0}) + 1009
        if edge.bonds:
            i, j, delta = edge.bonds[0]
            edge.bonds[0] = (missing, j, delta)
        elif edge.lone_pairs:
            _, delta = edge.lone_pairs[0]
            edge.lone_pairs[0] = (missing, delta)
        else:
            action = edge.charges[0]
            edge.charges[0] = type(action)(missing, action.q0, action.q1)
    elif corruption_type == "UNREACHABLE_EDGE":
        if not out.edges:
            raise ValueError("proof has no edge")
        out.edges[-1].src = "__missing_state__"
    elif corruption_type == "PRECURSOR_NOT_DERIVED":
        out.precursor_state_id = "__missing_precursor__"
    elif corruption_type == "EDGE_DEPENDENCY_REWIRE":
        if len(out.edges) < 2:
            raise ValueError("proof has fewer than two edges")
        second = out.edges[1]
        second.src = out.edges[0].src
    elif corruption_type == "COMMUTING_ORDER_CONTROL":
        rng.shuffle(out.edges)
    elif corruption_type == "STATE_RENAME_CONTROL":
        return parse_proof_program(rename_states(out, seed=seed))
    else:
        raise ValueError(f"unsupported corruption type: {corruption_type}")
    return out


DEFAULT_CORRUPTIONS = (
    "LP_WRONG_DELTA",
    "LP_DELETE",
    "BOND_DELETE",
    "BOND_WRONG_DELTA",
    "CHARGE_WRONG_PRECONDITION",
    "CHARGE_DELETE",
    "IMPORT_DELETE",
    "ATOM_MAP_REPLACE",
    "UNREACHABLE_EDGE",
    "PRECURSOR_NOT_DERIVED",
    "EDGE_DEPENDENCY_REWIRE",
)


def corrupt_proof(
    proof_text: str,
    *,
    corruption_type: str,
    seed: int = 0,
    source_id: str = "",
) -> ProofCorruption:
    """Create one controlled corruption and record the observed verifier result."""
    valid = format_proof_output(parse_proof_program(proof_text))
    if not execute_proof(valid).ok:
        raise ProofProgramError("source proof is not executable")
    corrupted_program = _corrupt_program(parse_proof_program(valid), corruption_type, seed)
    corrupted = format_proof_output(corrupted_program)
    observed = execute_proof(corrupted).ok
    expected = corruption_type.endswith("_CONTROL")
    certificate = diagnose_proof(corrupted)
    before = valid.splitlines()
    after = corrupted.splitlines()
    return ProofCorruption(
        source_id=source_id,
        corruption_type=corruption_type,
        valid_proof=valid,
        corrupted_proof=corrupted,
        expected_execute=expected,
        observed_execute=observed,
        failure_code=certificate.code if certificate else "",
        failure_edge=certificate.edge if certificate else "",
        repairable=bool(certificate and certificate.repairable),
        changed_lines=_format_changed(before, after),
    )


def build_corruption_set(
    proof_text: str,
    *,
    source_id: str = "",
    corruption_types: Iterable[str] = DEFAULT_CORRUPTIONS,
    seed: int = 0,
    require_observed_label: bool = True,
) -> list[ProofCorruption]:
    """Build all applicable corruptions, optionally dropping ambiguous outcomes."""
    out: list[ProofCorruption] = []
    for index, corruption_type in enumerate(corruption_types):
        try:
            item = corrupt_proof(
                proof_text,
                corruption_type=corruption_type,
                seed=seed + index,
                source_id=source_id,
            )
        except (ValueError, ProofProgramError):
            continue
        if require_observed_label and item.observed_execute != item.expected_execute:
            continue
        out.append(item)
    return out


def preference_pair_from_corruption(
    corruption: ProofCorruption,
    *,
    prompt_messages: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Return a safe verifier preference pair, or None for an ambiguous pair."""
    if not execute_proof(corruption.valid_proof).ok:
        return None
    if corruption.observed_execute:
        return None
    return {
        "id": f"{corruption.source_id}:{corruption.corruption_type}",
        "prompt_messages": prompt_messages,
        "chosen": corruption.valid_proof,
        "rejected": corruption.corrupted_proof,
        "chosen_verdict": "EXECUTABLE",
        "rejected_verdict": corruption.failure_code or "NON_EXECUTABLE",
        "metadata": corruption.to_dict(),
    }


def repair_row_from_corruption(
    corruption: ProofCorruption,
    *,
    product: str,
) -> dict[str, Any] | None:
    """Create certificate-conditioned proof repair SFT data."""
    certificate = diagnose_proof(corruption.corrupted_proof)
    if certificate is None:
        return None
    feedback = format_repair_feedback(certificate)
    return {
        "id": f"{corruption.source_id}:{corruption.corruption_type}:repair",
        "task_type": "mech_proof_repair",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Repair the MECH_PROOF v1 program using the deterministic "
                    "failure certificate. Return only one complete <proof> block."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TARGET: {product}\nINVALID_PROOF:\n"
                    f"{corruption.corrupted_proof}\nCERTIFICATE:\n{feedback}"
                ),
            },
            {"role": "assistant", "content": corruption.valid_proof},
        ],
        "metadata": {
            **corruption.to_dict(),
            "failure_feedback": feedback,
            "changed_lines": list(corruption.changed_lines),
        },
    }


def equivalence_metadata(proof_text: str) -> dict[str, Any]:
    signature = canonical_partial_order_signature(proof_text)
    return {
        "equivalence_digest": signature.digest(),
        "equivalence_signature": json.loads(signature.to_json()),
    }
