#!/usr/bin/env python3
"""Product-only closed-loop evaluation for the graph electron policy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch

from mechet.direct_graph_rl_env import DirectGraphElectronEnv
from mechet.graph_electron_policy import GraphElectronPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.max_steps < 1:
        raise SystemExit("limit and max-steps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    model = GraphElectronPolicy(
        hidden_dim=int(config.get("hidden_dim", 192)),
        num_layers=int(config.get("layers", 6)),
        dropout=0.0,
    )
    model.freeze_enumerated_action_heads()
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    accepted = attempted = exact = finished = generation_failures = 0
    action_families: Counter[str] = Counter()
    rejection_codes: Counter[str] = Counter()
    started = time.time()
    with args.data.open() as handle:
        for index, line in enumerate(handle):
            if index >= args.limit:
                break
            source = json.loads(line)
            target = str(source["target_smiles"])
            expected = str(source["expected_precursor"])
            env = DirectGraphElectronEnv(max_steps=args.max_steps)
            observation = env.reset(target=target, expected_precursor=expected)
            events = []
            endpoint_exact = False
            while True:
                sampled = model.sample_action(
                    observation.current,
                    observation.target,
                    trajectory=observation.history,
                    greedy=not args.sample,
                    temperature=args.temperature,
                )
                if not sampled.get("ok"):
                    generation_failures += 1
                action = sampled.get("action") or {
                    "kind": str(sampled.get("family") or "")
                }
                action_families[str(action.get("kind") or "UNKNOWN")] += 1
                transition = env.step(action)
                attempted += 1
                accepted += int(transition.accepted)
                if not transition.accepted:
                    rejection_codes[str(transition.result.get("code") or "UNKNOWN")] += 1
                endpoint_exact = endpoint_exact or bool(
                    transition.result.get("endpoint_exact")
                )
                events.append(
                    {
                        "action": action,
                        "sample": {
                            key: value
                            for key, value in sampled.items()
                            if key not in {"action", "program"}
                        },
                        "accepted": transition.accepted,
                        "result": dict(transition.result),
                    }
                )
                observation = transition.next_observation
                if transition.done:
                    finished += int(str(action.get("kind") or "") == "FINISH")
                    break
            exact += int(endpoint_exact)
            rows.append(
                {
                    "id": source.get("id"),
                    "events": events,
                    "endpoint_exact": endpoint_exact,
                    "steps": len(events),
                }
            )
            if (index + 1) % args.log_every == 0:
                print(
                    f"[graph-eval] reactions={index + 1}/{args.limit} "
                    f"endpoint_exact={exact}/{index + 1} "
                    f"executor_accept={accepted}/{attempted} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    denominator = len(rows)
    report = {
        "status": "completed",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "product_only_policy_input": True,
        "reactions": denominator,
        "endpoint_exact": exact,
        "endpoint_exact_rate": exact / max(1, denominator),
        "finished": finished,
        "executor_attempts": attempted,
        "executor_accepted": accepted,
        "executor_accept_rate": accepted / max(1, attempted),
        "generation_failures": generation_failures,
        "action_families": dict(action_families),
        "rejection_codes": dict(rejection_codes),
        "seed": args.seed,
        "wall_seconds": time.time() - started,
        "rows": rows,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
