#!/usr/bin/env python3
"""Evaluate solved and fully verified proof-carrying synthesis routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.routes.read_text(encoding="utf-8").splitlines() if line.strip()]
    solved = fully_verified = 0
    total_routes = total_steps = search_nodes = invalid_expansions = 0
    route_lengths = []
    cases = []
    for row in rows:
        routes = list(row.get("routes") or [])
        solved += int(bool(routes))
        target_verified = any(bool((route.get("verification") or {}).get("ok")) for route in routes)
        fully_verified += int(target_verified)
        total_routes += len(routes)
        search_nodes += int((row.get("stats") or {}).get("nodes_popped") or 0)
        invalid_expansions += int((row.get("stats") or {}).get("invalid_expansions") or 0)
        for route in routes:
            length = len(route.get("steps") or [])
            route_lengths.append(length)
            total_steps += length
        cases.append({
            "id": row.get("id"),
            "target": row.get("target"),
            "n_routes": len(routes),
            "fully_verified": target_verified,
            "search_stats": row.get("stats") or {},
        })
    n = len(rows)
    report = {
        "overall": {
            "n_targets": n,
            "solved_target_rate": solved / max(n, 1),
            "fully_verified_route_rate": fully_verified / max(n, 1),
            "n_routes": total_routes,
            "mean_route_length": sum(route_lengths) / max(len(route_lengths), 1),
            "mean_search_nodes": search_nodes / max(n, 1),
            "invalid_expansions_per_target": invalid_expansions / max(n, 1),
            "total_verified_steps_reported": total_steps,
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
