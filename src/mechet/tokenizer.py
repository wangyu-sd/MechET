"""Tokenizer audit for MechET MECH_ET structural keywords.

MechET does not use the ORBIT coding-agent opcode IR; we only check that
MECH_ET v3 keywords stay reasonably atomic under the base tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

from .model import resolve_qwen_model_path

# Structural tokens that appear in MECH_ET v3 CoT (not special-token registry).
MECH_ET_KEYWORDS = (
    "MECH_ET",
    "DIRECTION",
    "RETRO",
    "TARGET_SMILES",
    "PERCEIVE",
    "ENDPOINT",
    "CENTER",
    "ET_SIGNATURE",
    "ET_DEMAND",
    "N_STATES",
    "N_EDGES",
    "SHARED",
    "STATE",
    "TARGET_STATE",
    "PRECURSOR_STATE",
    "RETRO_EDGE",
    "BE_DELTA",
    "BOND",
    "LP",
    "CHARGE",
    "<mechanism>",
    "</mechanism>",
    "<answer>",
    "</answer>",
)


def audit_tokenizer(model_path: str | None = None) -> dict:
    model_path = model_path or resolve_qwen_model_path()
    report: dict = {
        "model_path": model_path,
        "status": "not_executed",
        "mech_et_keywords": list(MECH_ET_KEYWORDS),
        "keyword_split_examples": {},
        "multi_piece_keywords": [],
    }
    if not model_path:
        report["reason"] = "QWEN_MODEL_PATH unset and no local checkpoint found"
        return report
    try:
        from transformers import AutoTokenizer
    except ImportError:
        report["reason"] = "transformers not installed in active Python"
        return report

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    multi_piece: list[str] = []
    for kw in MECH_ET_KEYWORDS:
        pieces = tokenizer.tokenize(kw)
        report["keyword_split_examples"][kw] = pieces
        if len(pieces) > 1:
            multi_piece.append(kw)
    report["multi_piece_keywords"] = multi_piece
    report["status"] = "completed"
    report["vocab_size"] = len(tokenizer)
    return report


def save_tokenizer_audit(out_path: Path, model_path: str | None = None) -> dict:
    report = audit_tokenizer(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
