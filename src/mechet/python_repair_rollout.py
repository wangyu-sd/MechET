"""Gold-free execution feedback for full Python electron-program repair.

Programs are parsed as a restricted AST, never passed to Python eval/exec.
Repairs replace the whole STEPS list and replay from the original product.
"""
from __future__ import annotations

import json
from typing import Any

from .python_program import execute_python_program
from .python_template_slots import parse_template_slots
from .proof_program import ProofProgramError


def execution_feedback(target: str, program: str, *, terminated: bool) -> dict[str, Any]:
    """Intentionally accepts no reference precursor, gold trace, or reward."""
    if not terminated:
        return {"ok": False, "diagnostics": [{"code": "GENERATION_TRUNCATED",
                "message": "The STEPS output did not terminate within the generation budget."}]}
    try:
        parsed = parse_template_slots(program, target_smiles=target)
        result = execute_python_program(parsed, target_smiles=target)
        return {"ok": bool(result.ok), "diagnostics": list(result.diagnostics)}
    except (ProofProgramError, ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "diagnostics": [{"code": "STEPS_PARSE_FAILED", "message": str(exc)}]}


def repair_message(feedback: dict[str, Any]) -> str:
    if feedback["ok"]:
        raise ValueError("Executable candidates must not receive gold-dependent repair requests")
    # Fixed whitelist: even an accidentally enriched reward record cannot leak labels.
    errors = [{"code": str(d.get("code", "")), "message": str(d.get("message", ""))[:1200]}
              for d in feedback["diagnostics"][:2]]
    return ("The executor rejected your STEPS list. Diagnostics (step indices are zero-based):\n"
            + json.dumps(errors, ensure_ascii=False)
            + "\nReturn a corrected COMPLETE STEPS list, replayed from the original TARGET. "
            "Preserve valid actions when possible. Output only the list, with no prose or Markdown. "
            "The executor, not your answer text, determines the precursor.")


def continuation_tokens(tokenizer: Any, message: str, previous_ids: list[int]) -> list[int]:
    """Qwen ChatML environment continuation; never retokenize sampled actions.

    All returned tokens are observed context, not policy actions. The original
    assistant stop token (when present) is retained in the sampled action span.
    """
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if end_id is None or start_id is None or end_id == tokenizer.unk_token_id:
        raise ValueError("Repair v1 requires a Qwen ChatML tokenizer")
    closure = "" if previous_ids and previous_ids[-1] == end_id else "<|im_end|>"
    text = (closure + "\n<|im_start|>user\n" + message + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return tokenizer.encode(text, add_special_tokens=False)


def stop_token_ids(tokenizer: Any, generation_config: Any) -> list[int]:
    values = generation_config.eos_token_id
    result = set(values if isinstance(values, (list, tuple)) else [values])
    result.add(tokenizer.eos_token_id)
    result.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return sorted(int(i) for i in result if i is not None and i != tokenizer.unk_token_id)


def sampled_span(ids: list[int], stops: list[int]) -> tuple[list[int], bool]:
    for i, token in enumerate(ids):
        if token in stops:
            return ids[:i + 1], True
    return ids, False
