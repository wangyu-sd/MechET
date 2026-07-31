#!/usr/bin/env python3
"""Create atom-map-permuted proof rows for invariance training and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.map_invariance import record_map_permutation, remap_proof_text, remap_smiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            user_message = next(message for message in row["messages"] if message.get("role") == "user")
            product = str(user_message["content"]).split("\n", 1)[0].replace("TARGET:", "", 1).strip()
            assistant = next((str(message.get("content") or "") for message in row["messages"] if message.get("role") == "assistant"), "")
            metadata0 = row.get("metadata") or {}
            precursor0 = str(metadata0.get("derived_precursor") or metadata0.get("initial_reactants") or "")
            for copy_index in range(args.copies):
                permutation_seed = args.seed + row_index * args.copies + copy_index
                mapping = record_map_permutation(
                    product=product,
                    proof=assistant,
                    precursor=precursor0,
                    seed=permutation_seed,
                )
                variant = json.loads(json.dumps(row))
                variant["id"] = f"{row.get('id')}__map{copy_index}"
                for message in variant["messages"]:
                    content = str(message.get("content") or "")
                    if message.get("role") == "user" and content.startswith("TARGET:"):
                        first, *rest = content.split("\n")
                        message["content"] = "TARGET: " + remap_smiles(first.replace("TARGET:", "", 1).strip(), mapping)
                        if rest:
                            message["content"] += "\n" + "\n".join(rest)
                    elif message.get("role") == "assistant":
                        message["content"] = remap_proof_text(content, mapping)
                metadata = variant.get("metadata") or {}
                for key in ("derived_precursor", "initial_reactants", "core_precursor"):
                    if metadata.get(key):
                        metadata[key] = remap_smiles(str(metadata[key]), mapping)
                metadata["map_permutation_seed"] = permutation_seed
                variant["metadata"] = metadata
                handle.write(json.dumps(variant, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
