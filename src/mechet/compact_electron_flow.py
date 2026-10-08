"""Opt-in compact notation for the existing inverse electron-flow executor.

FLOW: B(A03,A04)>A04;LP(A02)>B(A02,A03)
DELTA: DELTA|B(A01,A02):+1;Q(A03):0>+1

This is a lossless serialization layer, not a new chemical transformation rule.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

ATOM = r"A[0-9]+"
_PAIR = rf"({ATOM}),({ATOM})"
_BOND = re.compile(rf"B\({_PAIR}\)")
_LP = re.compile(rf"LP\(({ATOM})\)")
_RADICAL = re.compile(rf"RP\({_PAIR}\)")
_ATOM = re.compile(rf"{ATOM}")
_BOND_DELTA = re.compile(rf"B\({_PAIR}\):([+-][0-9]+)")
_CHARGE_DELTA = re.compile(rf"Q\(({ATOM})\):([+-]?[0-9]+)>([+-]?[0-9]+)")
_NL_LP = re.compile(rf"a lone pair on atom ({ATOM})")
_NL_ATOM = re.compile(rf"atom ({ATOM})")
_NL_BOND = re.compile(rf"the bond (?:between|to form between) atoms ({ATOM}) and ({ATOM})")
_NL_RADICAL = re.compile(rf"a radical pair on atoms ({ATOM}) and ({ATOM})")


def _compact_container(text: str, *, source: bool) -> str:
    if source:
        match = _NL_LP.fullmatch(text)
        if match:
            return f"LP({match[1]})"
    match = _NL_BOND.fullmatch(text)
    if match:
        return f"B({match[1]},{match[2]})"
    match = _NL_RADICAL.fullmatch(text)
    if match:
        return f"RP({match[1]},{match[2]})"
    if not source:
        match = _NL_ATOM.fullmatch(text)
        if match:
            return match[1]
    raise ValueError(f"COMPACT_UNSUPPORTED_CONTAINER:{text}")


def compact_from_natural_arguments(arguments: Mapping[str, Any]) -> dict[str, str]:
    """Read v2 public source/sink fields, without consulting private atom maps."""
    if arguments.get("direction") != "retrosynthetic":
        raise ValueError("COMPACT_DIRECTION_INVALID")
    arrows = list(arguments.get("electron_flow") or [])
    bonds = list(arguments.get("bond_order_changes") or [])
    charges = list(arguments.get("charge_changes") or [])
    if arrows and (bonds or charges):
        raise ValueError("COMPACT_MIXED_EVENT")
    if arrows:
        return {"flow": ";".join(
            _compact_container(str(a["source"]), source=True) + ">"
            + _compact_container(str(a["destination"]), source=False)
            for a in arrows
        )}
    if not (bonds or charges):
        raise ValueError("COMPACT_EMPTY_EVENT")
    parts = []
    for item in bonds:
        atoms = list(item["atoms"])
        if len(atoms) != 2 or not all(_ATOM.fullmatch(str(a)) for a in atoms):
            raise ValueError("COMPACT_DELTA_ATOMS_INVALID")
        value = int(item["delta"])
        if not value:
            raise ValueError("COMPACT_ZERO_BOND_DELTA")
        parts.append(f"B({atoms[0]},{atoms[1]}):{value:+d}")
    for item in charges:
        atom = str(item["atom"])
        if not _ATOM.fullmatch(atom):
            raise ValueError("COMPACT_DELTA_ATOM_INVALID")
        parts.append(f"Q({atom}):{int(item['from'])}>{int(item['to']):+d}")
    return {"flow": "DELTA|" + ";".join(parts)}


def _natural_container(token: str, *, source: bool) -> str:
    if source:
        match = _LP.fullmatch(token)
        if match:
            return f"a lone pair on atom {match[1]}"
    match = _BOND.fullmatch(token)
    if match:
        kind = "the bond between atoms" if source else "the bond to form between atoms"
        return f"{kind} {match[1]} and {match[2]}"
    match = _RADICAL.fullmatch(token)
    if match:
        return f"a radical pair on atoms {match[1]} and {match[2]}"
    if not source and _ATOM.fullmatch(token):
        return f"atom {token}"
    raise ValueError(f"COMPACT_INVALID_CONTAINER:{token}")


def natural_from_compact_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Return the original v2 arguments for the authoritative interpreter."""
    if set(arguments) != {"flow"}:
        raise ValueError("COMPACT_ARGUMENTS_MUST_CONTAIN_ONLY_FLOW")
    flow = arguments["flow"]
    if not isinstance(flow, str) or not flow or len(flow) > 8192:
        raise ValueError("COMPACT_FLOW_LENGTH_INVALID")
    if any(ch.isspace() for ch in flow):
        raise ValueError("COMPACT_FLOW_WHITESPACE_INVALID")
    delta = flow.startswith("DELTA|")
    pieces = (flow[6:] if delta else flow).split(";")
    if not 1 <= len(pieces) <= 128 or any(not p for p in pieces):
        raise ValueError("COMPACT_FLOW_PARTS_INVALID")
    if delta:
        bonds, charges = [], []
        for p in pieces:
            b = _BOND_DELTA.fullmatch(p)
            if b:
                bonds.append({"atoms": [b[1], b[2]], "delta": int(b[3])})
                continue
            c = _CHARGE_DELTA.fullmatch(p)
            if c:
                charges.append({"atom": c[1], "from": int(c[2]), "to": int(c[3])})
                continue
            raise ValueError(f"COMPACT_INVALID_DELTA:{p}")
        return {"direction": "retrosynthetic", "electron_flow": [],
                "bond_order_changes": bonds, "charge_changes": charges}
    arrows = []
    for index, p in enumerate(pieces, 1):
        if p.count(">") != 1:
            raise ValueError(f"COMPACT_INVALID_MOVE:{p}")
        src, dst = p.split(">")
        source = _natural_container(src, source=True)
        destination = _natural_container(dst, source=False)
        arrows.append({
            "source": source, "destination": destination,
            "instruction": (
                f"Move {index}: transfer the electron pair from {source} "
                f"to {destination}."
            ),
        })
    return {"direction": "retrosynthetic", "electron_flow": arrows,
            "bond_order_changes": [], "charge_changes": []}


def render_compact_event_arguments(
    mapped_state: str, moves: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """Render using the existing v2 atom-alias assignment."""
    from .natural_language_electron_flow import render_event_arguments
    return compact_from_natural_arguments(render_event_arguments(mapped_state, moves))
