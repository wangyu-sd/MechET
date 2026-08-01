#!/usr/bin/env python3
"""Fine-tune the forward expert on independently verified hard negatives."""
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


ACCEPTED_LABEL_SOURCES = {
    "expert_review",
    "experiment",
    "known_competing_product",
    "independent_calibrated_ensemble",
}


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_verified_negative(row: dict) -> bool:
    metadata = row.get("metadata") or {}
    source = str(row.get("label_source") or metadata.get("label_source") or "")
    return (
        float(row.get("label", 1.0)) <= 0.0
        and bool(row.get("training_eligible"))
        and str(row.get("audit_status") or "") == "verified_negative"
        and source in ACCEPTED_LABEL_SOURCES
    )


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
    raw_negatives = load_jsonl(args.negative_data)
    negatives = [row for row in raw_negatives if is_verified_negative(row)]
    if not positives:
        raise ValueError("positive dataset is empty")
    if not negatives:
        raise ValueError(
            "no independently verified negatives: unreviewed actor/verifier "
            "disagreements are not training labels"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "positives": len(positives),
                    "raw_negative_candidates": len(raw_negatives),
                    "verified_negatives": len(negatives),
                    "negative_ratio": args.negative_ratio,
                    "accepted_label_sources": sorted(ACCEPTED_LABEL_SOURCES),
                },
                indent=2,
            )
        )
        return 0

    model = ForwardElectronExpert.load(args.checkpoint, device=args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
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
            reactants = str(row.get("reactants") or row.get("state_smiles") or "")
            product = str(row.get("products") or row.get("target_product") or "")
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
    model.save(
        args.output,
        metadata={
            "hard_negative_label_sources": sorted(ACCEPTED_LABEL_SOURCES),
            "unreviewed_disagreements_used": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
