"""MechET SFT chat formatting (MECH_ET v3 only)."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

MECH_ET_SYSTEM_PROMPT = (
    "You are MechET for electron-transfer mechanism-graph retrosynthesis. "
    "Given only a mapped main product SMILES, first perceive electronic endpoints and the reaction center, "
    "name the inverse electron-transfer signature, then reconstruct the full reverse FlowER mechanism graph "
    "(chain, tree, or DAG) with per-edge bond-electron deltas (BE_DELTA). "
    "BE_DELTA uses FlowER units (single bond = 1): BOND i j d, LP i d, optional CHARGE i q0 q1. "
    "STATE lines use strip-H mapped SMILES with SHARED spectators. "
    "Output exactly this format and nothing else:\n"
    "<mechanism>\n"
    "MECH_ET v3\n"
    "DIRECTION RETRO\n"
    'TARGET_SMILES "<mapped main product>"\n'
    "PERCEIVE\n"
    "  ENDPOINT <label> maps=<id,...>\n"
    "  CENTER <id-id>\n"
    "ET_SIGNATURE <name>\n"
    "ET_DEMAND <name>\n"
    "N_STATES n\n"
    "N_EDGES m\n"
    'SHARED "<optional spectator SMILES>"\n'
    'STATE s0 "<active mapped state SMILES>"\n'
    "...\n"
    "TARGET_STATE s0\n"
    "PRECURSOR_STATE sk\n"
    "RETRO_EDGE s0 s1\n"
    "  BE_DELTA\n"
    "    BOND i j d\n"
    "    LP i d\n"
    "...\n"
    "</mechanism>\n"
    "<answer>\n"
    "<initial reactant SMILES joined by '.'>\n"
    "</answer>"
)

MECHANISM_OPEN = "<mechanism>"
MECHANISM_CLOSE = "</mechanism>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def convert_record_to_qwen_sft(
    row: dict[str, Any],
    *,
    record_id: str | None = None,
    system_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = build_mech_et_messages(row, system_prompt=system_prompt)
    validate_messages(messages)
    out_id = str(record_id or row.get("id") or row.get("sample_id") or _stable_id(row))
    merged = copy.deepcopy(row.get("metadata") or {})
    if metadata:
        merged.update(copy.deepcopy(metadata))
    return {
        "id": out_id,
        "messages": messages,
        "task_type": "mech_et_cot_retro",
        "metadata": {
            **merged,
            "qwen_sft_format": "chat_messages_v1",
            "assistant_only_loss": True,
        },
    }


def build_mech_et_messages(row: dict[str, Any], *, system_prompt: str | None = None) -> list[dict[str, str]]:
    product = _require_first(row, "product_smiles", "target_smiles", "target", "product", "main_product")
    mechanism = _require_first(row, "mechanism", "mechanism_et", "mechanism_graph", "program", "target_program")
    precursors = _require_first(
        row,
        "reactants",
        "precursors",
        "precursor_smiles",
        "gt_reactants",
        "initial_reactants",
    )
    user = (
        f"TARGET: {product}\n"
        "Perceive electronic endpoints, name the inverse electron-transfer signature, "
        "predict the full reverse mechanism graph with BE_DELTA on each edge, then the initial precursors."
    )
    return [
        {"role": "system", "content": system_prompt or MECH_ET_SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": format_mech_et_assistant(_ensure_str(mechanism), _format_precursors(precursors)),
        },
    ]


def format_mech_et_assistant(mechanism_body: str, precursors: str | list[str]) -> str:
    mechanism = _ensure_str(mechanism_body).strip()
    answer = _format_precursors(precursors).strip()
    if not mechanism:
        raise ValueError("mechanism is empty")
    if not answer:
        raise ValueError("precursor answer is empty")
    return (
        f"{MECHANISM_OPEN}\n{mechanism}\n{MECHANISM_CLOSE}\n"
        f"{ANSWER_OPEN}\n{answer}\n{ANSWER_CLOSE}"
    )


def parse_mech_cot_output(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    mechanism = _extract_tagged_block(raw, MECHANISM_OPEN, MECHANISM_CLOSE)
    answer = _extract_tagged_block(raw, ANSWER_OPEN, ANSWER_CLOSE)
    format_ok = mechanism is not None and answer is not None
    return {
        "format_ok": format_ok,
        "mechanism": mechanism or "",
        "answer": (answer or "").strip(),
        "raw": raw,
    }


def parse_mech_et_output(text: str) -> dict[str, Any]:
    from mechet.mech_et import parse_mech_et_body, verify_mech_et

    parsed = parse_mech_cot_output(text)
    body = str(parsed.get("mechanism") or "")
    graph = parse_mech_et_body(body)
    verified = verify_mech_et(mechanism_body=body, answer=str(parsed.get("answer") or ""))
    return {
        **parsed,
        "graph": graph,
        "graph_ok": bool(graph.get("ok")),
        "reachability_ok": bool(verified.get("reachability_ok")),
        "be_delta_exact": bool(verified.get("be_delta_exact")),
        "electron_conserved": bool(verified.get("electron_conserved")),
        "n_states": int(verified.get("n_states") or 0),
        "n_edges": int(verified.get("n_edges") or 0),
        "diagnostics": list(verified.get("diagnostics") or []),
    }


def validate_messages(messages: list[dict[str, str]]) -> None:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("messages must end with an assistant turn")
    if not any(m.get("role") == "user" for m in messages):
        raise ValueError("messages must include a user turn")


def _require_first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    raise ValueError(f"missing required field; tried: {','.join(keys)}")


def _ensure_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_precursors(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ".".join(_ensure_str(item) for item in value)
    return _ensure_str(value)


def _extract_tagged_block(text: str, open_tag: str, close_tag: str) -> str | None:
    lower = text.lower()
    open_l = open_tag.lower()
    close_l = close_tag.lower()
    start = lower.find(open_l)
    if start < 0:
        return None
    content_start = start + len(open_tag)
    end = lower.find(close_l, content_start)
    if end < 0:
        return None
    return text[content_start:end].strip()


def _stable_id(row: dict[str, Any]) -> str:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
