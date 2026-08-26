#!/usr/bin/env python3
"""Build trace-owned inverse Tool-SFT from stitched mech-USPTO forward paths."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.inverse_trace_data import build_inverse_tool_sft_row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def error_code(exc: Exception) -> str:
    text = str(exc)
    match = re.match(r"([A-Z][A-Z0-9_]+)(?::|\b)", text)
    return match.group(1) if match else "UNCLASSIFIED_INVERSE_CONVERSION_FAILURE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--observation-mode",
        choices=(
            "action_delta",
            "compact_full_state",
            "reaction_center_delta",
            "full_state",
        ),
        default="action_delta",
    )
    args = parser.parse_args()

    raw = pd.read_parquet(
        args.raw_parquet, columns=["rxn_idx", "rxn_prod_min"]
    )
    product_references = {
        str(int(reaction_id)): str(group.iloc[0]["rxn_prod_min"])
        for reaction_id, group in raw.groupby("rxn_idx")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quarantine.parent.mkdir(parents=True, exist_ok=True)
    read = written = 0
    errors: Counter[str] = Counter()
    steps: Counter[int] = Counter()
    imports: Counter[int] = Counter()
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as good, args.quarantine.open("w", encoding="utf-8") as bad:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            if args.limit and read >= args.limit:
                break
            read += 1
            row = json.loads(line)
            reaction_id = str(row.get("id") or "")
            try:
                if reaction_id not in product_references:
                    raise ValueError("PRODUCT_REFERENCE_MISSING")
                value = build_inverse_tool_sft_row(
                    row,
                    product_reference=product_references[reaction_id],
                    observation_mode=args.observation_mode,
                )
                good.write(json.dumps(value, ensure_ascii=False) + "\n")
                written += 1
                metadata = value["metadata"]
                steps[int(metadata["n_trace_steps"])] += 1
                imports[int(metadata["n_trace_imports"])] += 1
            except Exception as exc:
                code = error_code(exc)
                errors[code] += 1
                bad.write(
                    json.dumps(
                        {
                            "source_row": index,
                            "id": reaction_id,
                            "error_code": code,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report = {
        "scientific_contract": "mech_uspto_inverse_trace_owned_v2",
        "source": "mech_uspto_31k",
        "condition": "trace_no_knowledge",
        "observation_mode": f"{args.observation_mode}_v1",
        "intermediate_state_model_visible": args.observation_mode != "action_delta",
        "read": read,
        "written": written,
        "quarantined": read - written,
        "conversion_rate": written / max(read, 1),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "raw_parquet": str(args.raw_parquet),
        "raw_parquet_sha256": sha256(args.raw_parquet),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "quarantine": str(args.quarantine),
        "quarantine_sha256": sha256(args.quarantine),
        "failure_codes": dict(sorted(errors.items())),
        "accepted_trace_steps": dict(sorted(steps.items())),
        "accepted_root_imports": dict(sorted(imports.items())),
        "target_source": "rxn_prod_min matched into globally mapped final state",
        "endpoint_source": "finish_trace",
        "corpus_used": False,
        "stereo_policy": (
            "clear only reacting tetrahedral tags absent from the final state"
        ),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
