#!/usr/bin/env python3
"""Fine-tune the compact forward expert on actor-mined explicit negatives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from mechet.forward_expert import ForwardElectronExpert


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positive-data", type=Path, required=True)
    parser.add_argument("--negative-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    positives = load_jsonl(args.positive_data)
    negatives = [
        row
        for row in load_jsonl(args.negative_data)
        if float(row.get("label", 0.0)) <= 0.0
    ]
    if not positives or not negatives:
        raise ValueError("positive and explicit negative datasets are both required")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "positives": len(positives),
                    "negatives": len(negatives),
                    "negative_ratio": args.negative_ratio,
                },
                indent=2,
            )
        )
        return 0
    model = ForwardElectronExpert.load(args.checkpoint, device=args.device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        n_negative = min(
            len(negatives), max(1, int(len(positives) * args.negative_ratio))
        )
        epoch_rows = [(row, 1.0) for row in positives] + [
            (row, 0.0) for row in rng.sample(negatives, n_negative)
        ]
        rng.shuffle(epoch_rows)
        model.train()
        losses = []
        for row, label in epoch_rows:
            reactants = str(
                row.get("reactants") or row.get("state_smiles") or ""
            )
            product = str(
                row.get("products") or row.get("target_product") or ""
            )
            if not reactants or not product:
                continue
            logit = model.reaction_score(
                reactants, product, conditions=row.get("conditions")
            )
            target = torch.full_like(logit, label)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with (args.output / "hard_negative_log.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "loss": sum(losses) / max(len(losses), 1),
                        "n_rows": len(losses),
                    }
                )
                + "\n"
            )
    model.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
