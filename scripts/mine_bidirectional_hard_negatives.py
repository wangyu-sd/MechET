#!/usr/bin/env python3
"""Mine inverse-actor proposals that fool the compact forward expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.adversarial import MiningConfig, load_jsonl, mine_forward_hard_negatives
from mechet.forward_expert import ForwardElectronExpert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-target-score", type=float, default=0.65)
    parser.add_argument("--maximum-selectivity-margin", type=float)
    parser.add_argument("--include-endpoint-exact", action="store_true")
    args = parser.parse_args()
    model = ForwardElectronExpert.load(args.checkpoint, device=args.device)
    negatives = mine_forward_hard_negatives(
        model,
        load_jsonl(args.predictions),
        config=MiningConfig(
            minimum_target_score=args.minimum_target_score,
            maximum_selectivity_margin=args.maximum_selectivity_margin,
            require_endpoint_error=not args.include_endpoint_exact,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in negatives:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"written": len(negatives), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
