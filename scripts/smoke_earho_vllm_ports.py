#!/usr/bin/env python3
"""Exercise eight simultaneous K=2 EARHO collectors without an optimizer step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_earho_v2 import load_earho_config, prepare, validate_contract
from scripts.run_natural_language_anchor_branch_rl import run_workers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_earho_config(args.config)
    validate_contract(cfg)
    output = Path(cfg["output_dir"])
    prepare(cfg, output)
    _, summary = run_workers(
        cfg,
        output / "round00/source.jsonl",
        Path(cfg["initial_adapter_path"]),
        output / "port_smoke_rollouts",
        frontier=int(cfg["curriculum"]["initial_frontier"]),
        round_index=0,
        evaluation=False,
    )
    print(json.dumps({"stage": "port-smoke-complete", "summary": {
        key: value for key, value in summary.items() if key != "group_summaries"
    }}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
