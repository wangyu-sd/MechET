#!/usr/bin/env python3
"""Best-first search over an offline pool of executable proof expansions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_program import sides_equal
from mechet.proof_routes import best_first_route_search, verify_route


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_building_blocks(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("{"):
            row = json.loads(text)
            text = str(row.get("smiles") or row.get("molecule") or "")
        if text:
            values.append(text)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--building-blocks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-routes", type=int, default=10)
    parser.add_argument("--branch-limit", type=int, default=32)
    args = parser.parse_args()

    targets = load_jsonl(args.targets)
    pool = load_jsonl(args.candidate_pool)
    building_blocks = load_building_blocks(args.building_blocks)

    def is_building_block(molecule: str) -> bool:
        return any(sides_equal(molecule, item, ignore_maps=True) for item in building_blocks)

    def expand(molecule: str):
        for row in pool:
            product = str(row.get("product") or row.get("target") or "")
            if product and sides_equal(product, molecule, ignore_maps=True):
                return list(row.get("candidates") or row.get("hypotheses") or [])
        return []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in targets:
            target = str(row.get("target") or row.get("product") or "")
            routes, stats = best_first_route_search(
                target,
                expand=expand,
                is_building_block=is_building_block,
                max_nodes=args.max_nodes,
                max_depth=args.max_depth,
                max_routes=args.max_routes,
                branch_limit=args.branch_limit,
            )
            payload = {
                "id": row.get("id"),
                "target": target,
                "stats": stats,
                "routes": [
                    {
                        **route.to_dict(),
                        "verification": verify_route(
                            target,
                            route.steps,
                            is_building_block=is_building_block,
                        ).to_dict(),
                    }
                    for route in routes
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
