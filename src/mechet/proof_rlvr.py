"""Group scoring for accuracy- and hypothesis-oriented proof RLVR."""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from mechet.iclr_rewards import compute_core_proof_reward
from mechet.proof_equivalence import canonical_partial_order_signature


def score_proof_group(
    row: dict[str, Any],
    completions: Sequence[str],
    *,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score a prompt group without rewarding invalid-string diversity."""
    cfg = dict(config or {})
    mode = str(cfg.get("mode", "accuracy"))
    scored = [compute_core_proof_reward(row, text, config=cfg) for text in completions]
    digests: list[str] = []
    for text, item in zip(completions, scored):
        digest = ""
        if item.get("execute_ok"):
            try:
                digest = canonical_partial_order_signature(text).digest()
            except Exception:
                digest = ""
        digests.append(digest)
    counts = Counter(value for value in digests if value)
    seen: set[str] = set()
    class_bonus = float(cfg.get("new_class_bonus", 1.0))
    duplicate_penalty = float(cfg.get("duplicate_class_penalty", 0.5))
    for item, digest in zip(scored, digests):
        item["equivalence_digest"] = digest
        item["base_rlvr_total"] = float(item.get("rlvr_total") or 0.0)
        if mode != "hypothesis" or not item.get("execute_ok") or not digest:
            item["diversity_adjustment"] = 0.0
            continue
        adjustment = 0.0
        if digest not in seen:
            adjustment += class_bonus
            seen.add(digest)
        if counts[digest] > 1:
            adjustment -= duplicate_penalty * (counts[digest] - 1) / counts[digest]
        item["diversity_adjustment"] = adjustment
        item["rlvr_total"] = float(item["rlvr_total"]) + adjustment
    return scored


def group_diagnostics(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    executable = [item for item in scored if item.get("execute_ok")]
    classes = {
        str(item.get("equivalence_digest") or "")
        for item in executable
        if item.get("equivalence_digest")
    }
    return {
        "n_rollouts": len(scored),
        "execute_rate": len(executable) / max(len(scored), 1),
        "endpoint_core_exact_rate": sum(bool(item.get("endpoint_core_exact")) for item in scored) / max(len(scored), 1),
        "n_executable_classes": len(classes),
        "class_diversity": len(classes) / max(len(executable), 1),
    }
