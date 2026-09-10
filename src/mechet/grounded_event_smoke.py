"""Pure-Qwen grounded electron-event smoke utilities.

This module is deliberately diagnostic.  It does not expose atom maps or ask a
language model to author SMARTS.  The executor owns site grounding and produces
a small set of formally executable elementary electron-flow events.  The model
only ranks randomized event labels from chemistry-bearing local descriptions.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .a7_rescue import canonical_mapped_state, executor_candidate_containers
from .forward_expert import ElectronContainer, ElectronMove, verify_electron_step


LABELS = tuple("ABCDEFGH")


def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError("invalid mapped molecular state")
    return mol


def _map_index(mol: Chem.Mol) -> dict[int, int]:
    output: dict[int, int] = {}
    for atom in mol.GetAtoms():
        atom_map = int(atom.GetAtomMapNum())
        if atom_map <= 0 or atom_map in output:
            raise ValueError("state atoms require unique positive maps")
        output[atom_map] = atom.GetIdx()
    return output


def _maps(smiles: str) -> set[int]:
    return set(_map_index(_mol(smiles)))


def state_with_imports(step: Mapping[str, Any]) -> str:
    """Materialize the executor state seen by an event, including first-use imports.

    Trace transitions keep imports as a separate field.  Older artifacts may
    already contain those atoms in ``state_before``; disjointness checks avoid
    duplicating them while still making corrected first-use traces executable.
    """

    state = str(step.get("state_before") or "")
    if not state:
        raise ValueError("step missing state_before")
    present = _maps(state)
    pieces = [state]
    for raw in step.get("imports") or []:
        fragment = str(raw or "").strip()
        if not fragment:
            continue
        incoming = _maps(fragment)
        overlap = present & incoming
        if overlap:
            if incoming <= present:
                continue
            raise ValueError(f"partial import-map overlap: {sorted(overlap)}")
        pieces.append(fragment)
        present.update(incoming)
    return canonical_mapped_state(".".join(pieces))


def unmap_state(smiles: str) -> str:
    mol = _mol(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _bond_name(bond: Chem.Bond | None) -> str:
    if bond is None:
        return "unbonded"
    if bond.GetIsAromatic():
        return "aromatic"
    value = float(bond.GetBondTypeAsDouble())
    if abs(value - 1.0) < 1e-6:
        return "single"
    if abs(value - 2.0) < 1e-6:
        return "double"
    if abs(value - 3.0) < 1e-6:
        return "triple"
    return str(bond.GetBondType())


def _atom_descriptor(mol: Chem.Mol, atom_map: int) -> str:
    indices = _map_index(mol)
    if atom_map not in indices:
        return "missing-atom"
    atom = mol.GetAtomWithIdx(indices[atom_map])
    neighbors = []
    for neighbor in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        neighbors.append(
            f"{_bond_name(bond)}-{neighbor.GetSymbol()}"
            f"(q={neighbor.GetFormalCharge()},aro={int(neighbor.GetIsAromatic())})"
        )
    neighbors.sort()
    second_shell = []
    for neighbor in atom.GetNeighbors():
        for outer in neighbor.GetNeighbors():
            if outer.GetIdx() == atom.GetIdx():
                continue
            second_shell.append(outer.GetSymbol())
    second_shell.sort()
    return (
        f"{atom.GetSymbol()}(q={atom.GetFormalCharge()},"
        f"aro={int(atom.GetIsAromatic())},deg={atom.GetDegree()},"
        f"H={atom.GetTotalNumHs()}; nbr=[{','.join(neighbors)}];"
        f" r2=[{','.join(second_shell)}])"
    )


def container_descriptor(smiles: str, container: ElectronContainer) -> str:
    mol = _mol(smiles)
    if len(container.atoms) == 1:
        atom = _atom_descriptor(mol, int(container.atoms[0]))
        return f"{container.kind} on {atom}"
    left, right = (int(container.atoms[0]), int(container.atoms[1]))
    indices = _map_index(mol)
    left_desc = _atom_descriptor(mol, left)
    right_desc = _atom_descriptor(mol, right)
    bond = None
    if left in indices and right in indices:
        bond = mol.GetBondBetweenAtoms(indices[left], indices[right])
    return (
        f"{container.kind} pair [{left_desc}] --({_bond_name(bond)})-- "
        f"[{right_desc}]"
    )


def event_descriptor(smiles: str, moves: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for index, raw in enumerate(moves, 1):
        if raw.get("mode") == "BE_DELTA":
            raise ValueError("BE_DELTA is outside the grounded-event smoke")
        move = ElectronMove.parse(raw)
        source = container_descriptor(smiles, move.source)
        sink = container_descriptor(smiles, move.sink)
        lines.append(f"move {index}: {source}  ->  {sink}")
    return "\n".join(lines)


def _signature(container: ElectronContainer) -> tuple[str, int]:
    return container.kind, len(container.atoms)


def _distance_to_center(
    mol: Chem.Mol, container: ElectronContainer, center_maps: set[int]
) -> int:
    indices = _map_index(mol)
    best = 10**6
    for atom_map in container.atoms:
        if atom_map not in indices:
            continue
        for center in center_maps:
            if center not in indices:
                continue
            path = Chem.GetShortestPath(mol, indices[int(atom_map)], indices[int(center)])
            if path:
                best = min(best, len(path) - 1)
    return best


def _container_dict(container: ElectronContainer) -> dict[str, Any]:
    return {"kind": container.kind, "atoms": list(container.atoms)}


def _candidate_successor(
    state: str, moves: Sequence[Mapping[str, Any]]
) -> str | None:
    result = verify_electron_step(state, list(moves))
    if not result.get("ok"):
        return None
    try:
        return canonical_mapped_state(str(result.get("state_smiles") or ""))
    except Exception:
        return None


def executable_event_candidates(
    *,
    state: str,
    gold_moves: Sequence[Mapping[str, Any]],
    max_candidates: int = 8,
    max_attempts: int = 512,
) -> tuple[list[dict[str, Any]], str]:
    """Return one gold event plus hard formally executable local alternatives.

    Alternatives preserve the event arity and replace one source/sink container
    at a time with an executor-grounded container of the same semantic type.
    Candidate successor states are deduplicated, so symmetry-equivalent arrows
    are not counted as different answers.
    """

    if max_candidates < 2 or max_candidates > len(LABELS):
        raise ValueError("max_candidates must be in [2, 8]")
    if any(item.get("mode") == "BE_DELTA" for item in gold_moves):
        raise ValueError("BE_DELTA_EVENT")
    parsed = [ElectronMove.parse(item) for item in gold_moves]
    if not parsed:
        raise ValueError("empty gold event")
    gold_successor = _candidate_successor(state, gold_moves)
    if not gold_successor:
        raise ValueError("gold event is not executable from materialized state")

    sources, sinks = executor_candidate_containers(state)
    mol = _mol(state)
    center_maps = {
        int(atom_map)
        for move in parsed
        for atom_map in (*move.source.atoms, *move.sink.atoms)
    }
    source_lists: list[list[ElectronContainer]] = []
    sink_lists: list[list[ElectronContainer]] = []
    for move in parsed:
        source_lists.append(
            sorted(
                [item for item in sources if _signature(item) == _signature(move.source)],
                key=lambda item: (
                    _distance_to_center(mol, item, center_maps),
                    container_descriptor(state, item),
                ),
            )
        )
        sink_lists.append(
            sorted(
                [item for item in sinks if _signature(item) == _signature(move.sink)],
                key=lambda item: (
                    _distance_to_center(mol, item, center_maps),
                    container_descriptor(state, item),
                ),
            )
        )

    negatives: dict[str, list[dict[str, Any]]] = {}
    attempts = 0

    def maybe_add(candidate: list[dict[str, Any]]) -> None:
        nonlocal attempts
        if attempts >= max_attempts or len(negatives) >= max_candidates - 1:
            return
        attempts += 1
        successor = _candidate_successor(state, candidate)
        if successor is None or successor == gold_successor or successor in negatives:
            return
        negatives[successor] = candidate

    # First pass: single-site substitutions are deliberately hard negatives.
    for move_index, move in enumerate(parsed):
        for source in source_lists[move_index]:
            if source == move.source:
                continue
            candidate = deepcopy(list(gold_moves))
            candidate[move_index] = dict(candidate[move_index])
            candidate[move_index]["source"] = _container_dict(source)
            maybe_add(candidate)
            if len(negatives) >= max_candidates - 1:
                break
        if len(negatives) >= max_candidates - 1:
            break
        for sink in sink_lists[move_index]:
            if sink == move.sink:
                continue
            candidate = deepcopy(list(gold_moves))
            candidate[move_index] = dict(candidate[move_index])
            candidate[move_index]["sink"] = _container_dict(sink)
            maybe_add(candidate)
            if len(negatives) >= max_candidates - 1:
                break
        if len(negatives) >= max_candidates - 1:
            break

    # Second pass: source+sink substitutions increase coverage when the local
    # executable action space is sparse under one-coordinate changes.
    if len(negatives) < max_candidates - 1:
        for move_index, move in enumerate(parsed):
            for source in source_lists[move_index][:16]:
                if source == move.source:
                    continue
                for sink in sink_lists[move_index][:24]:
                    if sink == move.sink:
                        continue
                    candidate = deepcopy(list(gold_moves))
                    candidate[move_index] = dict(candidate[move_index])
                    candidate[move_index]["source"] = _container_dict(source)
                    candidate[move_index]["sink"] = _container_dict(sink)
                    maybe_add(candidate)
                    if attempts >= max_attempts or len(negatives) >= max_candidates - 1:
                        break
                if attempts >= max_attempts or len(negatives) >= max_candidates - 1:
                    break
            if attempts >= max_attempts or len(negatives) >= max_candidates - 1:
                break

    candidates = [
        {
            "moves": [dict(item) for item in gold_moves],
            "successor": gold_successor,
            "is_gold": True,
        }
    ]
    candidates.extend(
        {"moves": moves, "successor": successor, "is_gold": False}
        for successor, moves in negatives.items()
    )
    return candidates, gold_successor


def grounded_event_task(
    row: Mapping[str, Any],
    event_index: int,
    *,
    seed: int = 17,
    max_candidates: int = 8,
) -> dict[str, Any]:
    plan = (row.get("metadata") or {}).get("trace_plan") or {}
    steps = list(plan.get("steps") or [])
    if not 0 <= event_index < len(steps):
        raise IndexError(event_index)
    step = steps[event_index]
    state = state_with_imports(step)
    gold_moves = list(step.get("moves") or [])
    candidates, gold_successor = executable_event_candidates(
        state=state,
        gold_moves=gold_moves,
        max_candidates=max_candidates,
    )

    key = f"{row.get('id')}::event-{event_index:03d}"
    rng_seed = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16)
    rng = random.Random(rng_seed)
    rng.shuffle(candidates)
    if len(candidates) > len(LABELS):
        candidates = candidates[: len(LABELS)]

    options = []
    correct_label = ""
    for label, candidate in zip(LABELS, candidates):
        descriptor = event_descriptor(state, candidate["moves"])
        options.append(
            {
                "label": label,
                "descriptor": descriptor,
                "moves": candidate["moves"],
                "successor": candidate["successor"],
                "is_gold": bool(candidate["is_gold"]),
            }
        )
        if candidate["is_gold"]:
            correct_label = label
    if not correct_label:
        raise AssertionError("gold event lost during candidate shuffle")

    history = []
    for previous in steps[:event_index]:
        previous_state = state_with_imports(previous)
        previous_moves = list(previous.get("moves") or [])
        if any(item.get("mode") == "BE_DELTA" for item in previous_moves):
            history.append("previous event: executor-verified legacy BE-delta transition")
        else:
            history.append(event_descriptor(previous_state, previous_moves))

    target = str(row.get("target_smiles") or plan.get("target_smiles") or "")
    return {
        "key": key,
        "id": str(row.get("id") or ""),
        "event_index": event_index,
        "event_depth": event_index + 1,
        "trajectory_events": len(steps),
        "target_product": unmap_state(target) if target else "",
        "current_state": unmap_state(state),
        "current_state_mapped_audit": state,
        "history": history,
        "candidate_count": len(options),
        "correct_label": correct_label,
        "gold_successor_mapped_audit": gold_successor,
        "options": options,
        "claim_boundary": (
            "Pure-Qwen F-oracle gold-state multiple-choice diagnostic; candidates "
            "are executor-grounded and gold-derived hard negatives, not a deployable "
            "full action enumerator and not benchmark endpoint accuracy."
        ),
    }


def render_grounded_event_prompt(task: Mapping[str, Any]) -> tuple[str, str]:
    system = (
        "You are evaluating an elementary inverse electron-flow decision. "
        "Every option is grounded to concrete sites in the current molecular graph "
        "and is formally executable by the chemistry executor. Choose the single "
        "event that best continues the retrosynthetic electron-flow trajectory. "
        "Do not write SMARTS, atom-map numbers, or a mechanism. Return only the "
        "option label."
    )
    history = task.get("history") or []
    history_text = "none" if not history else "\n\n".join(
        f"committed event {index + 1}:\n{value}" for index, value in enumerate(history)
    )
    options = "\n\n".join(
        f"[{item['label']}]\n{item['descriptor']}" for item in task.get("options") or []
    )
    user = (
        f"ORIGINAL TARGET PRODUCT:\n{task.get('target_product') or ''}\n\n"
        f"CURRENT EXECUTOR STATE:\n{task.get('current_state') or ''}\n\n"
        f"COMMITTED ELECTRON-FLOW HISTORY:\n{history_text}\n\n"
        f"EXECUTOR-GROUNDED CANDIDATE EVENTS:\n{options}\n\n"
        "Choose the best next event. Answer with one label only."
    )
    return system, user
