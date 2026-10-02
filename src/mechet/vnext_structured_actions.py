"""Per-state JSON action contracts for the vNext structured-decoding ablation.

This constrains syntax and *visible handle names* only.  It does not declare
any electron move chemically valid; the deterministic executor still decides.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .electron_pointer import PointerObservation


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


def structured_action_schema(
    observation: PointerObservation | None, *, inventory_handles: bool
) -> dict[str, Any]:
    """Return a vLLM/XGrammar-compatible JSON schema for one tool decision."""
    if inventory_handles and observation is None:
        raise ValueError("inventory-constrained schema requires current state")
    atoms = list(observation.atom_names) if observation is not None else []
    bonds = list(observation.bonds) if observation is not None else []
    names = {index: name for index, name in enumerate(atoms)}
    source_phrases = [f"a lone pair on atom {name}" for name in atoms]
    source_phrases += [
        f"the bond between atoms {names[i]} and {names[j]}" for i, j in bonds
    ]
    sink_phrases = [f"atom {name}" for name in atoms]
    sink_phrases += [
        f"the bond to form between atoms {names[i]} and {names[j]}"
        for i in range(len(atoms)) for j in range(i + 1, len(atoms))
    ]
    sink_phrases += [
        f"the bond between atoms {names[i]} and {names[j]}" for i, j in bonds
    ]
    source_field = {"type": "string", "enum": source_phrases} if inventory_handles else {"type": "string"}
    sink_field = {"type": "string", "enum": sink_phrases} if inventory_handles else {"type": "string"}
    atom_field = {"type": "string", "enum": atoms} if inventory_handles else {"type": "string"}
    move = _object(
        {"source": source_field, "destination": sink_field, "instruction": {"type": "string"}},
        ["source", "destination", "instruction"],
    )
    bond_delta = _object(
        {"atoms": {"type": "array", "items": atom_field},
         "delta": {"type": "integer", "minimum": -3, "maximum": 3},
         "instruction": {"type": "string"}},
        ["atoms", "delta", "instruction"],
    )
    charge_delta = _object(
        {"atom": atom_field, "from": {"type": "integer"},
         "to": {"type": "integer"}, "instruction": {"type": "string"}},
        ["atom", "from", "to", "instruction"],
    )
    event = _object(
        {"direction": {"type": "string", "const": "retrosynthetic"},
         "electron_flow": {"type": "array", "items": move},
         "bond_order_changes": {"type": "array", "items": bond_delta},
         "charge_changes": {"type": "array", "items": charge_delta}},
        ["direction", "electron_flow", "bond_order_changes", "charge_changes"],
    )
    fragment = _object(
        {"smiles": {"type": "string", "minLength": 1},
         "count": {"type": "integer", "minimum": 1},
         "purpose": {"type": "string", "enum": ["electron_participant", "endpoint_context"]}},
        ["smiles", "count", "purpose"],
    )
    import_args = _object(
        {"fragments": {"type": "array", "items": fragment}},
        ["fragments"],
    )
    variants = [
        _object({"name": {"const": "import_fragments"}, "arguments": import_args}, ["name", "arguments"]),
        _object({"name": {"const": "apply_electron_flow"}, "arguments": event}, ["name", "arguments"]),
        _object({"name": {"const": "finish_trace"}, "arguments": _object({}, [])}, ["name", "arguments"]),
    ]
    return {"oneOf": variants}


def parse_structured_action(text: str) -> tuple[str, dict[str, Any]]:
    """Read one plain JSON envelope; no surrounding prose or second action."""
    value = json.loads(text.strip())
    if not isinstance(value, Mapping) or set(value) != {"name", "arguments"}:
        raise ValueError("structured output must contain only name and arguments")
    name = value["name"]
    args = value["arguments"]
    if name not in {"import_fragments", "apply_electron_flow", "finish_trace"} or not isinstance(args, dict):
        raise ValueError("invalid structured action envelope")
    # vLLM 0.8.5 rejects minItems/maxItems before xgrammar compilation. Keep
    # those cardinality constraints here instead of silently weakening the
    # action contract for its generated JSON.
    if name == "import_fragments":
        fragments = args.get("fragments")
        if not isinstance(fragments, list) or not fragments:
            raise ValueError("import_fragments requires at least one fragment")
    elif name == "apply_electron_flow":
        changes = args.get("bond_order_changes")
        if not isinstance(changes, list):
            raise ValueError("bond_order_changes must be an array")
        for change in changes:
            if not isinstance(change, dict) or not isinstance(change.get("atoms"), list) or len(change["atoms"]) != 2:
                raise ValueError("bond_order_changes atoms must contain exactly two atoms")
    return str(name), args
