#!/usr/bin/env python3
"""Evaluate formal closure of proof-carrying catalytic-cycle proposals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.catalytic_cycle import CatalyticCycle, verify_catalytic_cycle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.cycles.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = []
    for row in rows:
        cycle = CatalyticCycle.from_dict(row)
        result = verify_catalytic_cycle(cycle)
        cases.append({
            "cycle_id": cycle.cycle_id,
            "verification": result.to_dict(),
        })
    n = len(cases)
    report = {
        "overall": {
            "n_cycles": n,
            "formal_cycle_pass_rate": sum(item["verification"]["ok"] for item in cases) / max(n, 1),
            "proof_execution_rate": sum(item["verification"]["proof_execution_ok"] for item in cases) / max(n, 1),
            "catalyst_regeneration_rate": sum(item["verification"]["catalyst_regenerated"] for item in cases) / max(n, 1),
            "oxidation_closure_rate": sum(item["verification"]["oxidation_state_closed"] for item in cases) / max(n, 1),
        },
        "cases": cases,
        "scope": (
            "formal proof/catalyst/ledger checks only; no energetic, kinetic, "
            "spin-state, condition, or experimental validation"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
