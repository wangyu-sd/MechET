"""Deterministic, model-free gates for the rapid A7 rescue protocol."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

from .forward_expert import ElectronMove, enumerate_containers, verify_electron_step


STRATA = ("short", "medium", "long")


def mechanism_length_stratum(n_events: int) -> str:
    if n_events <= 0:
        raise ValueError("a trajectory must contain at least one mechanism event")
    if n_events <= 2:
        return "short"
    if n_events <= 4:
        return "medium"
    return "long"


def stable_sample_key(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def stratified_sample(
    rows: Iterable[Mapping[str, Any]], *, size: int, seed: int
) -> list[dict[str, Any]]:
    """Select an approximately balanced deterministic sample by event length."""

    if size <= 0:
        raise ValueError("sample size must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        identifier = str(row.get("id") or "")
        if not identifier or identifier in seen:
            raise ValueError(f"missing or duplicate stable ID: {identifier!r}")
        seen.add(identifier)
        steps = ((row.get("metadata") or {}).get("trace_plan") or {}).get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"{identifier}: missing nonempty trace_plan.steps")
        grouped[mechanism_length_stratum(len(steps))].append(row)
    if len(seen) < size:
        raise ValueError(f"requested {size} rows from only {len(seen)}")
    for name in STRATA:
        grouped[name].sort(key=lambda row: stable_sample_key(str(row["id"]), seed))

    base, remainder = divmod(size, len(STRATA))
    requested = {
        name: base + (1 if index < remainder else 0)
        for index, name in enumerate(STRATA)
    }
    selected: list[dict[str, Any]] = []
    deficits = 0
    for name in STRATA:
        take = min(requested[name], len(grouped[name]))
        selected.extend(grouped[name][:take])
        grouped[name] = grouped[name][take:]
        deficits += requested[name] - take
    if deficits:
        remainder_rows = sorted(
            (row for name in STRATA for row in grouped[name]),
            key=lambda row: stable_sample_key(str(row["id"]), seed),
        )
        selected.extend(remainder_rows[:deficits])
    if len(selected) != size:
        raise AssertionError(f"selected result has {len(selected)} rows, expected {size}")
    return sorted(selected, key=lambda row: str(row["id"]))


def canonical_mapped_state(smiles: str) -> str:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError("invalid mapped state SMILES")
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
        raise ValueError("state atoms require unique positive maps")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_event(moves: Sequence[Mapping[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for raw in moves:
        if raw.get("mode") == "BE_DELTA":
            normalized.append(
                {
                    "mode": "BE_DELTA",
                    "bond_deltas": sorted(
                        (
                            {
                                "atoms": sorted(int(value) for value in item["atoms"]),
                                "delta": int(item["delta"]),
                            }
                            for item in raw.get("bond_deltas") or []
                        ),
                        key=lambda item: (item["atoms"], item["delta"]),
                    ),
                    "charge_actions": sorted(
                        (
                            {
                                "atom_map": int(item["atom_map"]),
                                "q0": int(item["q0"]),
                                "q1": int(item["q1"]),
                            }
                            for item in raw.get("charge_actions") or []
                        ),
                        key=lambda item: (item["atom_map"], item["q0"], item["q1"]),
                    ),
                }
            )
            continue
        move = ElectronMove.parse(raw)
        normalized.append(
            {
                "source": {
                    "kind": move.source.kind,
                    "atoms": list(move.source.atoms),
                },
                "sink": {"kind": move.sink.kind, "atoms": list(move.sink.atoms)},
                "electrons": 2,
            }
        )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _lp_electrons(atom: Chem.Atom) -> int:
    table = Chem.GetPeriodicTable()
    return int(
        table.GetNOuterElecs(atom.GetAtomicNum())
        - atom.GetFormalCharge()
        - sum(round(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds())
        - atom.GetTotalNumHs()
    )


def executor_candidate_containers(smiles: str):
    """Enumerate containers in the same Kekule representation as execution.

    The historical inspect inventory is intentionally left unchanged because a
    silent change would alter old checkpoints. This rescue-only inventory also
    includes homolytic containers accepted by ``verify_electron_step``.
    """

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("invalid mapped state SMILES")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms = list(mol.GetAtoms())
    from .forward_expert import ElectronContainer

    sources = []
    sinks = []
    for atom in atoms:
        atom_map = atom.GetAtomMapNum()
        sinks.append(ElectronContainer("ATOM", (atom_map,)))
        if _lp_electrons(atom) >= 2:
            sources.append(ElectronContainer("LP", (atom_map,)))
    for index, left in enumerate(atoms):
        for right in atoms[index + 1 :]:
            pair = (left.GetAtomMapNum(), right.GetAtomMapNum())
            bond = mol.GetBondBetweenAtoms(left.GetIdx(), right.GetIdx())
            sinks.append(ElectronContainer("BOND", pair))
            if bond is not None:
                sources.append(ElectronContainer("BOND", pair))
                sinks.append(ElectronContainer("RADICAL_PAIR", pair))
            elif _lp_electrons(left) >= 1 and _lp_electrons(right) >= 1:
                sources.append(ElectronContainer("RADICAL_PAIR", pair))
    return tuple(sorted(set(sources))), tuple(sorted(set(sinks)))


def assistant_events(row: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    events: list[list[dict[str, Any]]] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") != "apply_coupled_electron_moves":
                continue
            arguments = function.get("arguments") or {}
            moves = arguments.get("moves")
            if not isinstance(moves, list) or not moves:
                raise ValueError(f"{row.get('id')}: malformed coupled event")
            events.append([dict(item) for item in moves])
    return events


def assistant_event_targets(
    row: Mapping[str, Any],
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Return message indices and gold events without exposing later messages."""

    output: list[tuple[int, list[dict[str, Any]]]] = []
    for message_index, message in enumerate(row.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") != "apply_coupled_electron_moves":
                continue
            moves = (function.get("arguments") or {}).get("moves")
            if not isinstance(moves, list) or not moves:
                raise ValueError(f"{row.get('id')}: malformed coupled event")
            output.append((message_index, [dict(item) for item in moves]))
    return output


def local_prediction_metrics(
    *,
    state_before: str,
    state_after: str,
    gold_moves: Sequence[Mapping[str, Any]],
    predicted_name: str,
    predicted_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one generated next-tool decision from an authoritative gold state."""

    predicted_moves = predicted_arguments.get("moves")
    well_formed = (
        predicted_name == "apply_coupled_electron_moves"
        and isinstance(predicted_moves, list)
        and bool(predicted_moves)
        and all(isinstance(item, dict) for item in predicted_moves)
    )
    if not well_formed:
        return {
            "well_formed_event": False,
            "event_exact": False,
            "formal_execute": False,
            "successor_exact": False,
            "source_site_exact": False,
            "sink_site_exact": False,
            "source_type_exact": False,
            "sink_type_exact": False,
            "execution_error": "NOT_ONE_COUPLED_EVENT",
        }
    try:
        exact = canonical_event(predicted_moves) == canonical_event(gold_moves)
        predicted_parsed = [
            ElectronMove.parse(item)
            for item in predicted_moves
            if item.get("mode") != "BE_DELTA"
        ]
        gold_parsed = [
            ElectronMove.parse(item)
            for item in gold_moves
            if item.get("mode") != "BE_DELTA"
        ]
        predicted_sources = sorted(
            (move.source.kind, move.source.atoms) for move in predicted_parsed
        )
        gold_sources = sorted(
            (move.source.kind, move.source.atoms) for move in gold_parsed
        )
        predicted_sinks = sorted(
            (move.sink.kind, move.sink.atoms) for move in predicted_parsed
        )
        gold_sinks = sorted((move.sink.kind, move.sink.atoms) for move in gold_parsed)
        execution = verify_electron_step(state_before, predicted_moves)
        formal = bool(execution.get("ok"))
        successor = bool(
            formal
            and canonical_mapped_state(str(execution["state_smiles"]))
            == canonical_mapped_state(state_after)
        )
        return {
            "well_formed_event": True,
            "event_exact": exact,
            "formal_execute": formal,
            "successor_exact": successor,
            "source_site_exact": [atoms for _, atoms in predicted_sources]
            == [atoms for _, atoms in gold_sources],
            "sink_site_exact": [atoms for _, atoms in predicted_sinks]
            == [atoms for _, atoms in gold_sinks],
            "source_type_exact": [kind for kind, _ in predicted_sources]
            == [kind for kind, _ in gold_sources],
            "sink_type_exact": [kind for kind, _ in predicted_sinks]
            == [kind for kind, _ in gold_sinks],
            "execution_error": (
                ""
                if formal
                else str(
                    execution.get("message")
                    or execution.get("code")
                    or "EXECUTION_FAILED"
                )
            ),
        }
    except Exception as exc:
        return {
            "well_formed_event": False,
            "event_exact": False,
            "formal_execute": False,
            "successor_exact": False,
            "source_site_exact": False,
            "sink_site_exact": False,
            "source_type_exact": False,
            "sink_type_exact": False,
            "execution_error": str(exc),
        }


def audit_gold_event(step: Mapping[str, Any]) -> dict[str, Any]:
    before = str(step.get("state_before") or "")
    after = str(step.get("state_after") or "")
    moves = list(step.get("moves") or [])
    if not before or not after or not moves:
        raise ValueError("gold step requires before, after, and nonempty moves")
    result = verify_electron_step(before, moves)
    replay_ok = bool(result.get("ok"))
    endpoint_equal = bool(
        replay_ok
        and canonical_mapped_state(str(result["state_smiles"]))
        == canonical_mapped_state(after)
    )
    visible_sources, visible_sinks = enumerate_containers(before)
    candidate_sources, candidate_sinks = executor_candidate_containers(before)
    visible_source_set, visible_sink_set = set(visible_sources), set(visible_sinks)
    candidate_source_set, candidate_sink_set = set(candidate_sources), set(candidate_sinks)
    normal_moves = [item for item in moves if item.get("mode") != "BE_DELTA"]
    parsed = [ElectronMove.parse(item) for item in normal_moves]
    visible_source_covered = sum(move.source in visible_source_set for move in parsed)
    visible_sink_covered = sum(move.sink in visible_sink_set for move in parsed)
    candidate_source_covered = sum(move.source in candidate_source_set for move in parsed)
    candidate_sink_covered = sum(move.sink in candidate_sink_set for move in parsed)
    be_delta = len(moves) - len(parsed)
    return {
        "replay_ok": replay_ok,
        "successor_exact": endpoint_equal,
        "n_moves": len(moves),
        "n_standard_moves": len(parsed),
        "n_be_delta_moves": be_delta,
        "visible_source_containers_covered": visible_source_covered,
        "visible_sink_containers_covered": visible_sink_covered,
        "candidate_source_containers_covered": candidate_source_covered,
        "candidate_sink_containers_covered": candidate_sink_covered,
        "visible_inventory_covered": bool(
            visible_source_covered == len(parsed)
            and visible_sink_covered == len(parsed)
            and not be_delta
        ),
        "candidate_inventory_covered": bool(
            candidate_source_covered == len(parsed)
            and candidate_sink_covered == len(parsed)
        ),
        "gold_legal": bool(
            replay_ok
            and endpoint_equal
            and candidate_source_covered == len(parsed)
            and candidate_sink_covered == len(parsed)
        ),
    }


def audit_gold_row(row: Mapping[str, Any]) -> dict[str, Any]:
    identifier = str(row.get("id") or "")
    steps = list(((row.get("metadata") or {}).get("trace_plan") or {}).get("steps") or [])
    declared = [list(step.get("moves") or []) for step in steps]
    supervised = assistant_events(row)
    aligned = len(declared) == len(supervised) and all(
        canonical_event(left) == canonical_event(right)
        for left, right in zip(declared, supervised)
    )
    events = [audit_gold_event(step) for step in steps]
    return {
        "id": identifier,
        "stratum": mechanism_length_stratum(len(steps)),
        "n_events": len(steps),
        "n_moves": sum(item["n_moves"] for item in events),
        "supervision_aligned": aligned,
        "gold_legal": aligned and all(item["gold_legal"] for item in events),
        "events": events,
    }
