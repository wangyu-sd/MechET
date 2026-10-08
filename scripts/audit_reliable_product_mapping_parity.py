#!/usr/bin/env python3
"""Audit whether frozen gold decisions replay after product-only private remapping."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.chemical_runtime import require_endpoint_process_rdkit
from scripts.earho_v2_protocol import replay_reference
from scripts.run_natural_language_value_search import (
    policy_prompt, product_only_private_state, read_selected, visible,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    source: Path, decisions: Path, *, n: int, seed: int,
    compact_history: bool = True, max_imports: int = 64,
) -> dict:
    if max_imports < 1:
        raise ValueError("max_imports must be positive")
    rdkit_version = require_endpoint_process_rdkit()
    selected = read_selected(source, n, seed)
    if len(selected) != n or len({str(row["source_id"]) for row in selected}) != n:
        raise ValueError("source does not contain the requested unique reaction denominator")
    wanted = {str(row["source_id"]) for row in selected}
    by_source = defaultdict(list)
    with decisions.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row["source_id"])
            if source_id in wanted:
                by_source[source_id].append(row)
    if set(by_source) != wanted:
        raise ValueError("gold decisions do not cover the selected source reactions")

    counts: Counter[str] = Counter()
    failures = []
    for index, row in enumerate(selected, 1):
        source_id = str(row["source_id"])
        gold = by_source[source_id]
        gold.sort(key=lambda item: int(item["metadata"]["decision_index"]))
        canonical = product_only_private_state(str(row["target_smiles"]))
        root_prompt = policy_prompt(
            visible(canonical), canonical, include_inventory=True,
            actions=[], compact_history=compact_history,
        )
        if root_prompt != gold[0]["messages"][1]["content"]:
            counts["root_prompt_mismatch"] += 1
            failures.append({"source_id": source_id, "kind": "root_prompt_mismatch"})
            continue
        counts["root_prompt_exact"] += 1
        try:
            replay_reference(
                row, gold, compact_history=compact_history,
                max_imports=max_imports,
            )
            counts["original_private_map_replay_ok"] += 1
        except Exception as exc:
            counts["original_private_map_replay_failed"] += 1
            failures.append({
                "source_id": source_id, "kind": "original_private_map_replay_failed",
                "error": f"{type(exc).__name__}:{exc}",
            })
            continue
        remapped = dict(row, target_smiles=canonical)
        try:
            replay_reference(
                remapped, gold, compact_history=compact_history,
                max_imports=max_imports,
            )
            counts["product_only_remap_replay_ok"] += 1
        except Exception as exc:
            counts["product_only_remap_replay_failed"] += 1
            failures.append({
                "source_id": source_id, "kind": "product_only_remap_replay_failed",
                "error": f"{type(exc).__name__}:{exc}",
            })
        if index % 100 == 0:
            print(f"[mapping-parity] {index}/{n} remap_failed={counts['product_only_remap_replay_failed']}", flush=True)
    return {
        "artifact_type": "reliable_mechet_product_only_private_mapping_audit_v1",
        "source": str(source), "source_sha256": sha256(source),
        "decisions": str(decisions), "decisions_sha256": sha256(decisions),
        "n_reactions": n, "seed": seed, "counts": dict(counts),
        "rdkit_version": rdkit_version,
        "max_imports": max_imports,
        "observation_contract": (
            "compressed_history" if compact_history else "state_only"
        ),
        "failures": failures,
        "interpretation": (
            "The policy-visible root prompt can be identical while executor replay "
            "changes under product-only private remapping. This is an address/Kekule "
            "representation diagnostic, not model accuracy or chemical adjudication."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--state-only", action="store_true")
    parser.add_argument("--max-imports", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.source, args.decisions, n=args.n, seed=args.seed,
        compact_history=not args.state_only, max_imports=args.max_imports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "n_reactions": report["n_reactions"], "counts": report["counts"],
        "output": str(args.output),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
