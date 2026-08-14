#!/usr/bin/env python3
"""Build replay-verified globally mapped traces from standardized step rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.trace_stitching import stitch_steps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--complete-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--max-matches", type=int, default=4096)
    args = parser.parse_args()

    complete = {
        str(json.loads(line)["reaction_id"])
        for line in args.complete_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    headers: dict[str, dict] = {}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reaction_id = str(row["id"])
            if reaction_id not in complete:
                continue
            grouped[reaction_id].extend(row.get("steps") or [])
            headers.setdefault(
                reaction_id,
                {
                    "id": reaction_id,
                    "source": row.get("source"),
                    "split": row.get("split"),
                },
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quarantine.parent.mkdir(parents=True, exist_ok=True)
    written = quarantined = ambiguous_links = total_links = 0
    max_isomorphism_matches = 1
    with args.output.open("w", encoding="utf-8") as good, args.quarantine.open(
        "w", encoding="utf-8"
    ) as bad:
        for reaction_id in sorted(complete, key=int):
            try:
                steps, metadata = stitch_steps(
                    grouped[reaction_id], max_matches=args.max_matches
                )
                payload = {
                    **headers[reaction_id],
                    "initial_state": steps[0]["state_smiles"],
                    "final_state": steps[-1]["target_product"],
                    "steps": steps,
                    "metadata": {
                        "atom_map_scope": "global_trace",
                        "stitching": metadata,
                    },
                }
                good.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
                ambiguous_links += int(metadata["ambiguous_links"])
                total_links += int(metadata["links"])
                max_isomorphism_matches = max(
                    max_isomorphism_matches,
                    int(metadata["max_isomorphism_matches"]),
                )
            except Exception as exc:
                bad.write(
                    json.dumps(
                        {"reaction_id": reaction_id, "error": str(exc)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                quarantined += 1

    report = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "complete_ids": str(args.complete_ids),
        "complete_ids_sha256": _sha256(args.complete_ids),
        "expected": len(complete),
        "written": written,
        "quarantined": quarantined,
        "total_links": total_links,
        "ambiguous_links": ambiguous_links,
        "max_isomorphism_matches": max_isomorphism_matches,
        "max_matches": args.max_matches,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "quarantine": str(args.quarantine),
        "quarantine_sha256": _sha256(args.quarantine),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
