"""Tokenizer audit and special-token handling for ORBIT language."""

from __future__ import annotations

import json
from pathlib import Path

from orbit_coding_agent.ir.validator import OPERATIONS

from .model import resolve_qwen_model_path

STABLE_OPCODE_TOKENS = sorted(OPERATIONS)


def audit_tokenizer(model_path: str | None = None) -> dict:
    model_path = model_path or resolve_qwen_model_path()
    report = {
        "model_path": model_path,
        "status": "not_executed",
        "stable_opcode_tokens": STABLE_OPCODE_TOKENS,
        "added_special_tokens": [],
        "opcode_split_examples": {},
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
    for op in STABLE_OPCODE_TOKENS[:5]:
        report["opcode_split_examples"][op] = tokenizer.tokenize(op)
    existing = set(tokenizer.additional_special_tokens or [])
    to_add = [op for op in STABLE_OPCODE_TOKENS if op not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add[:8]})
        report["added_special_tokens"] = to_add[:8]
    report["status"] = "completed"
    report["vocab_size"] = len(tokenizer)
    return report


def save_tokenizer_audit(out_path: Path, model_path: str | None = None) -> dict:
    report = audit_tokenizer(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
