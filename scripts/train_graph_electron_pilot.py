#!/usr/bin/env python3
"""Train a small graph electron policy on real frozen FlowER trace decisions.

This is a feasibility runner, not a headline benchmark.  It deliberately keeps
the reaction denominator and source hashes visible in ``report.json`` and does
not rewrite/filter the source artifact.  Full-scale training should batch graph
states and use this runner's exact action/data contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch

from mechet.forward_expert import ElectronMove
from mechet.graph_electron_policy import (
    ACTION_FAMILIES,
    GraphElectronPolicy,
    strip_atom_maps,
)
from mechet.graph_fragment_actions import (
    classify_imports,
    decompose_reactive_fragment,
)
from mechet.transactional_event_space import MoveInventory


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            yield json.loads(line)


def fragment_bank(path: Path, row_limit: int, bank_size: int) -> tuple[list[str], Counter[str]]:
    counts: Counter[str] = Counter()
    for row in rows(path, row_limit):
        plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
        for item in classify_imports(plan):
            if item.kind == "IMPORT_ENV":
                counts[strip_atom_maps(item.fragment)] += 1
    return [item for item, _ in counts.most_common(bank_size)], counts


def append_fragment(state: str, mapped_fragment: str) -> str:
    return f"{state}.{mapped_fragment}" if state else mapped_fragment


def build_decisions(
    path: Path,
    *,
    reaction_limit: int,
    decision_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    decisions: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row in rows(path, reaction_limit):
        plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
        target = str(plan.get("target_smiles") or "")
        current = target
        import_supervision = iter(classify_imports(plan))

        def add_import(fragment: str) -> None:
            nonlocal current
            item = next(import_supervision)
            if item.fragment != str(fragment):
                raise ValueError("chronological import supervision drift")
            decision: dict[str, Any] = {
                "reaction_id": row.get("id"),
                "kind": item.kind,
                "current": current,
                "target": target,
                "fragment": str(fragment),
            }
            if item.kind == "IMPORT_REACTIVE":
                decision["program"] = decompose_reactive_fragment(
                    str(fragment),
                    participating_maps=item.participating_maps,
                    role=str(item.role),
                )
            decisions.append(decision)
            stats[item.kind] += 1
            current = append_fragment(current, str(fragment))

        for fragment in plan.get("initial_imports") or ():
            add_import(str(fragment))
            if len(decisions) >= decision_limit:
                return decisions, dict(stats)
        for step in plan.get("steps") or ():
            for fragment in step.get("imports") or ():
                add_import(str(fragment))
                if len(decisions) >= decision_limit:
                    return decisions, dict(stats)
            moves = list(step.get("moves") or ())
            if len(moves) == 1 and moves[0].get("mode") == "BE_DELTA":
                kind = "BE_DELTA"
            else:
                kind = "FLOW"
                # Fail loudly on candidate-space gaps; do not silently filter.
                inventory = MoveInventory.from_state(str(step["state_before"]))
                for value in moves:
                    move = ElectronMove.parse(value)
                    if move.source not in inventory.sources or move.sink not in inventory.compatible_sinks(move.source):
                        raise ValueError(
                            f"{row.get('id')} step {step.get('step_index')} outside legal inventory: {move}"
                        )
            decisions.append(
                {
                    "reaction_id": row.get("id"),
                    "kind": kind,
                    # ``state_before`` is recorded before this step's imports.
                    # ``current`` includes them and is therefore the only valid
                    # observation for moves that reference newly allocated maps
                    # (notably explicit-H BE_DELTA events).
                    "current": current,
                    "target": target,
                    "moves": moves,
                }
            )
            stats[kind] += 1
            current = str(step["state_after"])
            if len(decisions) >= decision_limit:
                return decisions, dict(stats)
        decisions.append(
            {
                "reaction_id": row.get("id"),
                "kind": "FINISH",
                "current": current,
                "target": target,
            }
        )
        stats["FINISH"] += 1
        if len(decisions) >= decision_limit:
            return decisions, dict(stats)
    return decisions, dict(stats)


def import_candidates(
    gold: str,
    bank: list[str],
    *,
    negatives: int,
    rng: random.Random,
) -> tuple[list[str], int]:
    key = strip_atom_maps(gold)
    pool = [item for item in bank if item != key]
    sampled = rng.sample(pool, min(negatives, len(pool)))
    candidates = sampled + [key]
    rng.shuffle(candidates)
    return candidates, candidates.index(key)


def decision_loss(
    model: GraphElectronPolicy,
    decision: dict[str, Any],
    *,
    bank: list[str],
    negatives: int,
    rng: random.Random,
) -> torch.Tensor:
    kind = decision["kind"]
    if kind == "FLOW":
        return model.flow_nll(decision["current"], decision["target"], decision["moves"])[0]
    if kind == "BE_DELTA":
        return model.be_delta_nll(
            decision["current"], decision["target"], decision["moves"][0]
        )[0]
    if kind == "IMPORT_ENV":
        candidates, gold = import_candidates(
            decision["fragment"], bank, negatives=negatives, rng=rng
        )
        return model.import_nll(
            decision["current"], decision["target"], candidates, gold
        )
    if kind == "IMPORT_REACTIVE":
        return model.reactive_fragment_nll(
            decision["current"], decision["target"], decision["program"]
        )[0]
    if kind == "FINISH":
        return model.finish_nll(decision["current"], decision["target"])
    raise ValueError(f"unknown decision kind: {kind}")


@torch.no_grad()
def evaluate_decisions(
    model: GraphElectronPolicy,
    decisions: list[dict[str, Any]],
    *,
    bank: list[str],
    negatives: int,
    seed: int,
    limit: int,
) -> dict[str, Any]:
    model.eval()
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    for decision in decisions[:limit]:
        context = model.encode_context(decision["current"], decision["target"])
        family = ACTION_FAMILIES[int(model.family_logits(context).argmax())]
        counts["family_total"] += 1
        counts["family_correct"] += int(family == decision["kind"])
        if decision["kind"] == "IMPORT_ENV":
            candidates, gold = import_candidates(
                decision["fragment"], bank, negatives=negatives, rng=rng
            )
            scores, _ = model.import_logits(context, candidates)
            counts["import_total"] += 1
            counts["import_correct"] += int(int(scores.argmax()) == gold)
        elif decision["kind"] == "FLOW":
            inventory = MoveInventory.from_state(decision["current"])
            history = torch.zeros_like(context.vector)
            for value in decision["moves"]:
                move = ElectronMove.parse(value)
                source_scores, source_embeddings = model.source_logits(
                    context, inventory.sources, history
                )
                source_index = inventory.sources.index(move.source)
                counts["source_total"] += 1
                counts["source_correct"] += int(int(source_scores.argmax()) == source_index)
                sinks = inventory.compatible_sinks(move.source)
                sink_scores, sink_embeddings = model.sink_logits(
                    context, source_embeddings[source_index], sinks, history
                )
                sink_index = sinks.index(move.sink)
                counts["sink_total"] += 1
                counts["sink_correct"] += int(int(sink_scores.argmax()) == sink_index)
                action = torch.tanh(source_embeddings[source_index] + sink_embeddings[sink_index])
                history = model.event_cell(action[None, :], history[None, :]).squeeze(0)

    def ratio(correct: str, total: str) -> float | None:
        return counts[correct] / counts[total] if counts[total] else None

    return {
        "decisions": min(limit, len(decisions)),
        "counts": dict(counts),
        "family_top1": ratio("family_correct", "family_total"),
        "import_sampled_top1": ratio("import_correct", "import_total"),
        "source_top1_teacher_forced": ratio("source_correct", "source_total"),
        "sink_top1_teacher_forced": ratio("sink_correct", "sink_total"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reaction-limit", type=int, default=256)
    parser.add_argument("--decision-limit", type=int, default=512)
    parser.add_argument("--bank-rows", type=int, default=50000)
    parser.add_argument("--bank-size", type=int, default=5000)
    parser.add_argument("--import-negatives", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-decisions", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--heartbeat", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(args.manifest.read_text())
    frozen = source_manifest["splits"]["train"]
    actual_sha = sha256_file(args.train)
    if int(frozen["rows"]) != 257167 or actual_sha != frozen["sha256"]:
        raise SystemExit(
            "source contract mismatch: expected strict FlowER train 257167 and pinned SHA"
        )
    print(
        f"[graph-electron] source verified rows={frozen['rows']} sha256={actual_sha}",
        flush=True,
    )
    started = time.time()
    bank, counts = fragment_bank(args.train, args.bank_rows, args.bank_size)
    print(
        f"[graph-electron] fragment bank={len(bank)} scanned_reactions={args.bank_rows} "
        f"observations={sum(counts.values())}",
        flush=True,
    )
    decisions, decision_counts = build_decisions(
        args.train,
        reaction_limit=args.reaction_limit,
        decision_limit=args.decision_limit,
    )
    print(
        f"[graph-electron] decisions={len(decisions)} counts={decision_counts}",
        flush=True,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphElectronPolicy(
        hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=0.1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    history: list[dict[str, float]] = []
    initial_eval = evaluate_decisions(
        model,
        decisions,
        bank=bank,
        negatives=args.import_negatives,
        seed=args.seed + 1000,
        limit=args.eval_decisions,
    )
    print(f"[graph-electron] initial_eval={initial_eval}", flush=True)
    global_step = 0
    last_heartbeat = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(decisions)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for offset, decision_index in enumerate(order, start=1):
            loss = decision_loss(
                model,
                decisions[decision_index],
                bank=bank,
                negatives=args.import_negatives,
                rng=rng,
            )
            (loss / args.accumulate).backward()
            running += float(loss.detach())
            if offset % args.accumulate == 0 or offset == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            now = time.time()
            if now - last_heartbeat >= args.heartbeat:
                print(
                    f"[graph-electron] heartbeat epoch={epoch + 1}/{args.epochs} "
                    f"decision={offset}/{len(order)} mean_loss={running / offset:.5f}",
                    flush=True,
                )
                last_heartbeat = now
        mean_loss = running / max(1, len(order))
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss})
        print(
            f"[graph-electron] epoch={epoch + 1}/{args.epochs} mean_loss={mean_loss:.6f}",
            flush=True,
        )
        torch.save(
            {"model": model.state_dict(), "config": {"hidden_dim": args.hidden_dim, "layers": args.layers}},
            args.output / f"checkpoint-epoch{epoch + 1}.pt",
        )
    report = {
        "status": "completed_pilot",
        "claim_boundary": "feasibility/overfit pilot; not full-test accuracy",
        "source": {
            "train": str(args.train),
            "manifest": str(args.manifest),
            "rows": int(frozen["rows"]),
            "sha256": actual_sha,
            "official_denominators": source_manifest["official_reaction_denominators"],
            "strict_universe": source_manifest["splits"],
        },
        "pilot": {
            "reaction_limit": args.reaction_limit,
            "decision_limit": args.decision_limit,
            "decisions": len(decisions),
            "decision_counts": decision_counts,
            "fragment_bank_rows": args.bank_rows,
            "fragment_bank_size": len(bank),
            "import_negatives": args.import_negatives,
        },
        "model": {
            "name": "GraphElectronPolicy",
            "action_families": list(ACTION_FAMILIES),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device),
        },
        "history": history,
        "initial_eval": initial_eval,
        "final_eval": evaluate_decisions(
            model,
            decisions,
            bank=bank,
            negatives=args.import_negatives,
            seed=args.seed + 1000,
            limit=args.eval_decisions,
        ),
        "wall_seconds": time.time() - started,
    }
    (args.output / "fragment_bank.json").write_text(json.dumps(bank, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[graph-electron] completed report={args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
