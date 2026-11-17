#!/usr/bin/env python3
"""Prepare TRAIN-only self-search CE targets; no model, teacher or training call."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'receipt-manifest', 'training-targets', 'output-dir', 'source-actor-sha']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--eos-token-id', type=int, required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--anchors')
    parser.add_argument('--sft-anchor-source-sha')
    parser.add_argument('--actor-anchor-source-sha')
    parser.add_argument('--sft-weight', type=float, default=0.0)
    parser.add_argument('--actor-weight', type=float, default=0.0)
    args = parser.parse_args(argv)
    # Standalone loading avoids unrelated eager Torch imports in mechet.__init__.
    path = Path(__file__).resolve().parents[1]/'src/mechet/route_self_distill.py'
    spec = importlib.util.spec_from_file_location('standalone_route_self_distill', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.prepare_dataset(args.input, args.receipt_manifest, args.training_targets,
                                    args.output_dir, expected_actor_sha=args.source_actor_sha,
                                    eos_token_id=args.eos_token_id, temperature=args.temperature,
                                    anchors_path=args.anchors, sft_source_sha=args.sft_anchor_source_sha,
                                    actor_source_sha=args.actor_anchor_source_sha,
                                    sft_weight=args.sft_weight, actor_weight=args.actor_weight)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
