#!/usr/bin/env python3
"""Build verified equivalent MECH_PROOF variants for SFT augmentation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_curriculum import equivalence_metadata, proof_text_from_row
from mechet.proof_variants import build_equivalent_variants


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants-per-row", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = skipped = variants_written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            proof = proof_text_from_row(row)
            if not proof:
                skipped += 1
                continue
            try:
                variants = build_equivalent_variants(
                    proof,
                    n_variants=args.variants_per_row,
                    seed=args.seed + row_index,
                )
                eq = equivalence_metadata(proof)
            except Exception:
                skipped += 1
                continue
            accepted += 1
            for variant_index, variant in enumerate(variants):
                output = dict(row)
                output["id"] = f"{row.get('id', row_index)}:eq{variant_index}"
                messages = [dict(message) for message in row.get("messages") or []]
                replaced = False
                for index in range(len(messages) - 1, -1, -1):
                    if messages[index].get("role") == "assistant":
                        messages[index]["content"] = variant
                        replaced = True
                        break
                if not replaced:
                    messages.append({"role": "assistant", "content": variant})
                output["messages"] = messages
                output["task_type"] = "mech_proof_retro"
                output["metadata"] = {
                    **dict(row.get("metadata") or {}),
                    **eq,
                    "equivalence_source_id": str(row.get("id", row_index)),
                    "equivalence_variant_index": variant_index,
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                variants_written += 1
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "n_input": len(rows),
        "accepted_sources": accepted,
        "skipped_sources": skipped,
        "variants_written": variants_written,
        "variants_per_row_requested": args.variants_per_row,
        "seed": args.seed,
        "acceptance": "execute_ok_and_partial_order_equivalent",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
