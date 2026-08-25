#!/usr/bin/env python3
"""Append explicitly labelled upstream-corrupt endpoint rows without filtering."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SYSTEM = (
    "You are MechET. This frozen upstream record has no atom-conserving "
    "reaction-level mechanism. Predict only its recorded precursor endpoint."
)


def tagged(text: str, name: str) -> str:
    match = re.search(
        rf"<{name}>\s*(.*?)\s*</{name}>", text, flags=re.DOTALL
    )
    if not match:
        raise ValueError(f"missing <{name}> block")
    return match.group(1).strip()


def fallback_row(row: dict) -> dict:
    metadata = dict(row.get("metadata") or {})
    assistant = next(
        str(message.get("content") or "")
        for message in reversed(row.get("messages") or [])
        if message.get("role") == "assistant"
    )
    user = next(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
    )
    target = user.split("\n", 1)[0].replace("TARGET:", "").strip()
    precursor = tagged(assistant, "answer")
    trajectory_id = str(metadata.get("trajectory_id") or "")
    if trajectory_id not in {"RC", "PC", "PM", "RS"}:
        raise ValueError(f"unexpected fallback trajectory: {trajectory_id}")
    return {
        "id": str(row.get("id") or "").replace(
            "flower_mech_et", "flower_endpoint_fallback"
        ),
        "artifact_type": "supervision",
        "target_smiles": target,
        "expected_precursor": precursor,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"TARGET: {target}\nReturn the frozen precursor endpoint.",
            },
            {"role": "assistant", "content": f"<answer>{precursor}</answer>"},
        ],
        "tools": [],
        "metadata": {
            "source": "FlowER flower_retro frozen endpoint",
            "source_trajectory_id": trajectory_id,
            "upstream_endpoint_fallback": True,
            "endpoint_source": "upstream_frozen_endpoint_fallback",
            "executor_replayed": False,
            "corpus_used": False,
            "failure_reason": "upstream endpoints are not atom-conserving",
            "reaction_filtering": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mech-et", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=4)
    args = parser.parse_args()

    selected = []
    with args.mech_et.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("metadata") or {}).get("upstream_endpoint_fallback"):
                selected.append(fallback_row(row))
    if len(selected) != args.expected:
        raise ValueError(f"fallback count mismatch: {len(selected)} != {args.expected}")
    with args.trace.open("a", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"trace": str(args.trace), "appended": len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
