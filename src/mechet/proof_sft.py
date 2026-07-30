"""SFT formatting for proof-carrying retrosynthesis."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from mechet.proof_program import (
    compile_mech_et_body,
    execute_proof,
    format_proof_output,
    sides_equal,
)

MECH_PROOF_SYSTEM_PROMPT = (
    "You are MechET-Proof for proof-carrying retrosynthesis. "
    "Given only a mapped target-product SMILES, emit one executable MECH_PROOF v1 program. "
    "Do not emit intermediate STATE SMILES and do not emit an answer. The executor derives the precursor. "
    "ROOT imports declare mapped species present before electron-flow actions. Each EDGE contains sparse "
    "BOND, LP, and CHARGE operations. All atom maps must be unique and every edge must conserve electrons. "
    "Output exactly:\n"
    "<proof>\n"
    "MECH_PROOF v1\n"
    'TARGET_SMILES "<mapped product>"\n'
    "ROOT <state-id>\n"
    '  IMPORT "<mapped species>"\n'
    "PRECURSOR_STATE <state-id>\n"
    "EDGE <source-id> <destination-id>\n"
    '  IMPORT "<optional newly introduced mapped species>"\n'
    "  BOND <map-i> <map-j> <signed-delta>\n"
    "  LP <map-i> <signed-delta>\n"
    "  CHARGE <map-i> <old-charge> <new-charge>\n"
    "</proof>"
)


def _stable_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _extract_tagged(text: str, open_tag: str, close_tag: str) -> str:
    raw = (text or "").strip()
    lower = raw.lower()
    start = lower.find(open_tag.lower())
    if start < 0:
        return ""
    start += len(open_tag)
    end = lower.find(close_tag.lower(), start)
    if end < 0:
        return ""
    return raw[start:end].strip()


def _extract_existing(row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row.get("messages") or []
    user = next(
        (
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        ),
        "",
    )
    assistant = next(
        (
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "assistant"
        ),
        "",
    )
    product = user.split("\n", 1)[0].replace("TARGET:", "").strip()
    mechanism = _extract_tagged(assistant, "<mechanism>", "</mechanism>")
    answer = _extract_tagged(assistant, "<answer>", "</answer>")
    if not product or not mechanism or not answer:
        raise ValueError(
            "row does not contain TARGET, <mechanism>, and <answer>"
        )
    return product, mechanism, answer


def convert_mech_et_row_to_proof_sft(
    row: dict[str, Any],
) -> dict[str, Any]:
    product, mechanism, original_answer = _extract_existing(row)
    program = compile_mech_et_body(mechanism)
    executed = execute_proof(program)
    if not executed.ok:
        raise ValueError(f"compiled proof failed: {executed.diagnostics}")
    if not sides_equal(executed.precursor_smiles, original_answer):
        raise ValueError(
            "executor-derived precursor does not match original answer"
        )

    metadata = copy.deepcopy(row.get("metadata") or {})
    metadata.update(
        {
            "source_task_type": row.get("task_type")
            or metadata.get("task_type"),
            "task_type": "mech_proof_retro",
            "proof_version": "MECH_PROOF v1",
            "answer_channel": "executor_derived",
            "derived_precursor": executed.precursor_smiles,
            "assistant_only_loss": True,
            "qwen_sft_format": "chat_messages_v1",
        }
    )
    user = (
        f"TARGET: {product}\n"
        "Synthesize an executable inverse electron-flow proof. "
        "The precursor will be obtained only by executing the proof."
    )
    return {
        "id": str(row.get("id") or _stable_id(row)).replace(
            "flower_mech_et",
            "flower_mech_proof",
        ),
        "messages": [
            {
                "role": "system",
                "content": MECH_PROOF_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": format_proof_output(program),
            },
        ],
        "task_type": "mech_proof_retro",
        "metadata": metadata,
    }
