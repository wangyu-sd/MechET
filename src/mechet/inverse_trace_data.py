"""Convert globally mapped forward paths into trace-owned inverse supervision."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .agent_env import AgentEnvConfig
from .endpoints import split_precursor_endpoints
from .forward_expert import ElectronContainer, ElectronMove
from .proof_program import _canonical_mapped, sides_equal
from .tool_schemas import trace_tool_schemas
from .trace_agent_env import TraceOwnedAgentEnv


SYSTEM_PROMPT = """You are MechET, a trace-owned inverse electron-flow agent.
Reconstruct the precursor only through explicit environment tool calls. The
final proof and precursor must be produced by finish_trace."""


def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles), params)
    if mol is None:
        raise ValueError("TARGET_COMPONENT_PARSE_FAILED")
    return mol


def _fragments(smiles: str) -> list[str]:
    return [
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        for fragment in Chem.GetMolFrags(
            _mol(smiles), asMols=True, sanitizeFrags=True
        )
    ]


def _component_key(smiles: str) -> str:
    """Map/stereo-insensitive constitutional key for target identification."""
    mol = Chem.RemoveHs(_mol(smiles))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _specified_tetrahedral_maps(smiles: str) -> set[int]:
    return {
        atom.GetAtomMapNum()
        for atom in _mol(smiles).GetAtoms()
        if atom.GetAtomMapNum()
        and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    }


def _reaction_center_maps(forward_steps: Sequence[Mapping[str, Any]]) -> set[int]:
    atom_maps: set[int] = set()
    for step in forward_steps:
        for value in step.get("moves") or []:
            move = ElectronMove.parse(value)
            atom_maps.update(move.source.atoms)
            atom_maps.update(move.sink.atoms)
    return atom_maps


def underdetermined_stereo_maps(
    initial_state: str,
    final_state: str,
    forward_steps: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Find reacting tetrahedral centers whose product no longer specifies stereo.

    Electron-pair moves encode connectivity and charge, not attack-face geometry.
    A precursor chiral tag lost at a reacting center therefore cannot be inferred
    during inverse replay and must not be used as an exact endpoint label.
    """
    initial = _specified_tetrahedral_maps(initial_state)
    final = _specified_tetrahedral_maps(final_state)
    reacting = _reaction_center_maps(forward_steps)
    return tuple(sorted((initial - final) & reacting))


def clear_atom_stereo(smiles: str, atom_maps: Sequence[int]) -> str:
    selected = set(atom_maps)
    if not selected:
        return _canonical_mapped(smiles)
    mol = _mol(smiles)
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in selected:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return _canonical_mapped(mol)


def select_mapped_target(
    final_state: str, product_reference: str
) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    """Select mapped product fragments and return the remaining root imports."""
    final_fragments = sorted(_fragments(final_state))
    reference_fragments = _fragments(product_reference)
    available: dict[str, list[str]] = defaultdict(list)
    for fragment in final_fragments:
        available[_component_key(fragment)].append(fragment)
    required = Counter(_component_key(fragment) for fragment in reference_fragments)
    selected: list[str] = []
    for key, count in sorted(required.items()):
        candidates = available.get(key, [])
        if len(candidates) < count:
            raise ValueError(
                f"STRUCTURAL_TARGET_NOT_FOUND: key={key} required={count} "
                f"available={len(candidates)}"
            )
        selected.extend(candidates[:count])
        del candidates[:count]
    imports = tuple(
        sorted(fragment for values in available.values() for fragment in values)
    )
    target = _canonical_mapped(".".join(selected))
    if not sides_equal(
        _canonical_mapped(".".join((target, *imports))),
        final_state,
        ignore_maps=False,
    ):
        raise ValueError("TARGET_IMPORT_PARTITION_MISMATCH")
    return target, imports, {
        "reference_fragments": len(reference_fragments),
        "target_fragments": len(selected),
        "root_imports": len(imports),
        "component_match_ignores_maps": True,
        "component_match_ignores_stereo": True,
    }


def invert_move(value: Mapping[str, Any]) -> dict[str, Any]:
    move = ElectronMove.parse(value)
    if move.source.kind == "LP" and move.sink.kind == "BOND":
        inverse = ElectronMove(
            ElectronContainer("BOND", move.sink.atoms),
            ElectronContainer("ATOM", move.source.atoms),
        )
    elif move.source.kind == "BOND" and move.sink.kind in {"ATOM", "LP"}:
        inverse = ElectronMove(
            ElectronContainer("LP", move.sink.atoms),
            ElectronContainer("BOND", move.source.atoms),
        )
    elif move.source.kind == "BOND" and move.sink.kind == "BOND":
        inverse = ElectronMove(
            ElectronContainer("BOND", move.sink.atoms),
            ElectronContainer("BOND", move.source.atoms),
        )
    else:
        raise ValueError(f"INVERSE_MOVE_UNSUPPORTED: {move.id}")
    # Tool arguments must follow TRACE_TOOLS exactly; cached IDs are useful in
    # internal artifacts but are forbidden by additionalProperties=false.
    return {
        "source": {
            "kind": inverse.source.kind,
            "atoms": list(inverse.source.atoms),
        },
        "sink": {
            "kind": inverse.sink.kind,
            "atoms": list(inverse.sink.atoms),
        },
        "electrons": 2,
    }


def invert_moves(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [invert_move(value) for value in reversed(values)]


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tool_result(
    call_id: str, name: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(dict(result), ensure_ascii=False),
    }


def build_inverse_tool_sft_row(
    forward_trace: Mapping[str, Any],
    *,
    product_reference: str,
) -> dict[str, Any]:
    reaction_id = str(forward_trace.get("id") or "")
    forward_steps = list(forward_trace.get("steps") or [])
    if not reaction_id or not forward_steps:
        raise ValueError("FORWARD_TRACE_INCOMPLETE")
    final_state = _canonical_mapped(str(forward_trace["final_state"]))
    stereo_maps = underdetermined_stereo_maps(
        str(forward_trace["initial_state"]), final_state, forward_steps
    )
    initial_state = clear_atom_stereo(
        str(forward_trace["initial_state"]), stereo_maps
    )
    target, initial_imports, target_metadata = select_mapped_target(
        final_state, product_reference
    )
    required_calls = len(initial_imports) + len(forward_steps) + 1
    env = TraceOwnedAgentEnv(
        config=AgentEnvConfig(max_tool_calls=max(12, required_calls + 2))
    )
    observation = json.loads(
        env.reset(target_smiles=target, expected_precursor=initial_state)
    )
    observation["instructions"] = [
        instruction
        for instruction in observation.get("instructions") or []
        if "inspect_state" not in str(instruction)
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"TARGET: {target}\n"
                "Reproduce the executable inverse electron-flow trace and call "
                "finish_trace. The precursor must come only from the environment.\n\n"
                "INITIAL ENVIRONMENT OBSERVATION:\n"
                + json.dumps(observation, ensure_ascii=False)
            ),
        },
    ]
    call_index = 0

    for fragment in initial_imports:
        call_id = f"call_{call_index:03d}"
        call_index += 1
        arguments = {"fragment_smiles": fragment}
        messages.append(_tool_call(call_id, "import_fragment", arguments))
        result = json.loads(env.import_fragment(**arguments))
        if not result.get("ok"):
            raise ValueError(f"INVERSE_ROOT_IMPORT_FAILED: {result}")
        messages.append(_tool_result(call_id, "import_fragment", result))

    inverse_steps: list[dict[str, Any]] = []
    for inverse_index, forward_step in enumerate(reversed(forward_steps)):
        moves = invert_moves(forward_step.get("moves") or [])
        expected_after = clear_atom_stereo(
            str(forward_step["state_smiles"]), stereo_maps
        )
        call_id = f"call_{call_index:03d}"
        call_index += 1
        arguments = {"moves": moves}
        messages.append(
            _tool_call(call_id, "apply_coupled_electron_moves", arguments)
        )
        result = json.loads(
            env.apply_coupled_electron_moves(
                json.dumps(moves, ensure_ascii=False)
            )
        )
        if not result.get("ok"):
            raise ValueError(f"INVERSE_MOVE_REPLAY_FAILED: {result}")
        if not sides_equal(
            str(result.get("state_smiles") or ""),
            expected_after,
            ignore_maps=False,
        ):
            raise ValueError(
                f"INVERSE_MOVE_STATE_MISMATCH: step={inverse_index}"
            )
        messages.append(_tool_result(call_id, "apply_coupled_electron_moves", result))
        inverse_steps.append(
            {
                "step_index": inverse_index,
                "forward_step_index": int(forward_step.get("step_index", 0)),
                "state_before": str(result["trace_step"]["state_before"]),
                "state_after": str(result["state_smiles"]),
                "imports": list(result["trace_step"].get("imports") or []),
                "moves": moves,
            }
        )

    call_id = f"call_{call_index:03d}"
    messages.append(_tool_call(call_id, "finish_trace", {}))
    terminal = json.loads(env.finish_trace())
    if not terminal.get("ok") or not terminal.get("endpoint_exact"):
        raise ValueError(f"INVERSE_FINISH_TRACE_FAILED: {terminal}")
    messages.append(_tool_result(call_id, "finish_trace", terminal))
    messages.append(
        {
            "role": "assistant",
            "content": (
                "The environment-owned inverse trace compiled and executed "
                "successfully; the precursor is taken only from finish_trace."
            ),
        }
    )

    endpoints = split_precursor_endpoints(initial_state, target)
    source_digest = hashlib.sha256(
        json.dumps(
            dict(forward_trace), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    plan = {
        "target_smiles": target,
        "expected_precursor": initial_state,
        "initial_imports": list(initial_imports),
        "steps": inverse_steps,
    }
    return {
        "id": f"mech-uspto31k-inverse:{reaction_id}",
        "source_id": reaction_id,
        "artifact_type": "supervision",
        "messages": messages,
        "tools": trace_tool_schemas(textbook=False),
        "target_smiles": target,
        "expected_precursor": endpoints.full,
        **endpoints.to_dict(),
        "metadata": {
            "source_dataset": "mech_uspto_31k",
            "source_split": str(forward_trace.get("split") or ""),
            "source_forward_trace_sha256": source_digest,
            "source_product_reference": product_reference,
            "direction": "inverse",
            "trace_condition": "trace_no_knowledge",
            "trace_plan": plan,
            "n_trace_steps": len(inverse_steps),
            "n_trace_moves": sum(len(step["moves"]) for step in inverse_steps),
            "n_trace_imports": len(initial_imports),
            "target_selection": target_metadata,
            "stereo_normalization": {
                "mode": "clear_underdetermined_reaction_center_tetrahedral_tags_v1",
                "atom_maps": list(stereo_maps),
                "count": len(stereo_maps),
            },
            "compiled_proof": terminal.get("compiled_proof"),
            "trace_digest": terminal.get("trace_digest"),
            "move_sequence_digest": terminal.get("move_sequence_digest"),
            "executor_replayed": True,
            "endpoint_source": "environment_owned_trace",
        },
    }
