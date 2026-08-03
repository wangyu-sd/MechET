"""Canonical JSON schemas for MechET tool-calling supervision and inference."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _function(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required or []),
                "additionalProperties": False,
            },
        },
    }


TRACE_TOOLS: tuple[dict[str, Any], ...] = (
    _function(
        "inspect_state",
        "Inspect the current mapped molecular state and enumerate legal electron sources and sinks.",
        {},
    ),
    _function(
        "import_fragment",
        "Import one mapped fragment that is absent from the product but required by the next inverse transition.",
        {"fragment_smiles": {"type": "string", "description": "Atom-mapped fragment SMILES with unique positive map numbers."}},
        ["fragment_smiles"],
    ),
    _function(
        "apply_electron_move",
        "Apply one explicit two-electron source-to-sink move to the current state.",
        {
            "source_kind": {"type": "string", "enum": ["LP", "BOND"]},
            "source_atoms": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
            "sink_kind": {"type": "string", "enum": ["ATOM", "LP", "BOND"]},
            "sink_atoms": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
        },
        ["source_kind", "source_atoms", "sink_kind", "sink_atoms"],
    ),
    _function(
        "apply_coupled_electron_moves",
        "Apply a non-empty list of coupled two-electron moves atomically as one elementary event.",
        {
            "moves": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["LP", "BOND"]},
                                "atoms": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
                            },
                            "required": ["kind", "atoms"],
                            "additionalProperties": False,
                        },
                        "sink": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["ATOM", "LP", "BOND"]},
                                "atoms": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
                            },
                            "required": ["kind", "atoms"],
                            "additionalProperties": False,
                        },
                        "electrons": {"type": "integer", "enum": [2]},
                    },
                    "required": ["source", "sink"],
                    "additionalProperties": False,
                },
            }
        },
        ["moves"],
    ),
    _function(
        "finish_trace",
        "Compile the committed environment-owned trace, execute it, and derive the only admissible precursor endpoint.",
        {},
    ),
    _function(
        "abstain",
        "Terminate the episode without an unsupported precursor when available evidence is insufficient.",
        {"reason": {"type": "string", "description": "Concise reason for abstaining."}},
        ["reason"],
    ),
)

TEXTBOOK_TOOL = _function(
    "retrieve_textbook_guidance",
    "Retrieve bounded provenance-aware textbook evidence for the current molecular state. The evidence is soft and unrewarded.",
    {
        "query": {"type": "string", "description": "Optional chemistry query derived from information available at inference time."},
        "top_k": {"type": "integer", "minimum": 1},
        "max_characters": {"type": "integer", "minimum": 1},
    },
)

ANCHOR_TOOL = _function(
    "retrieve_primitives",
    "Retrieve structured mechanistic knowledge anchors with role bindings, warnings, and candidate moves. Matches are soft evidence.",
    {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1},
    },
)

LEGACY_SUBMIT_PROOF_TOOL = _function(
    "submit_proof",
    "Submit a complete MECH_PROOF v1 program in the legacy complete-proof baseline.",
    {"proof": {"type": "string"}},
    ["proof"],
)


def trace_tool_schemas(*, textbook: bool = False, anchors: bool = False, legacy_submit_proof: bool = False) -> list[dict[str, Any]]:
    """Return a fresh canonical tool-schema list for one experimental condition."""

    tools = [deepcopy(item) for item in TRACE_TOOLS]
    if textbook:
        tools.insert(1, deepcopy(TEXTBOOK_TOOL))
    if anchors:
        insert_at = 2 if textbook else 1
        tools.insert(insert_at, deepcopy(ANCHOR_TOOL))
    if legacy_submit_proof:
        tools.append(deepcopy(LEGACY_SUBMIT_PROOF_TOOL))
    return tools


def tool_names(tools: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str((item.get("function") or {}).get("name") or "") for item in tools)
