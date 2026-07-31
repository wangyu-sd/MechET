#!/usr/bin/env python3
"""Inference, electron-flow generation and evaluation for the forward expert."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from rdkit import Chem

from mechet.forward_expert import (
    ElectronMove,
    ForwardElectronExpert,
    score_reaction,
    verify_electron_step,
)


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def same_molecule(left: str, right: str) -> bool:
    left_mol = Chem.MolFromSmiles(left)
    right_mol = Chem.MolFromSmiles(right)
    if left_mol is None or right_mol is None:
        return False
    for mol in (left_mol, right_mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(
        left_mol,
        canonical=True,
        isomericSmiles=True,
    ) == Chem.MolToSmiles(
        right_mol,
        canonical=True,
        isomericSmiles=True,
    )


def gold_move_statistics(
    model: ForwardElectronExpert,
    state: str,
    moves: list[dict[str, Any]],
    conditions: Any = None,
) -> list[dict[str, Any]]:
    ranked = model.rank_moves(state, top_k=100000, conditions=conditions)
    index = {
        ElectronMove.parse(
            {"source": item["source"], "sink": item["sink"]}
        ).id: rank
        for rank, item in enumerate(ranked, 1)
    }
    output = []
    for move in moves:
        parsed = ElectronMove.parse(move)
        output.append(
            {
                "move": parsed.to_dict(),
                "rank": index.get(parsed.id),
                "top1": index.get(parsed.id) == 1,
            }
        )
    return output


def candidate_steps(
    model,
    state: str,
    *,
    proposal_pool: int,
    max_candidates: int,
    coupled: bool,
    conditions: Any = None,
):
    ranked = model.rank_moves(
        state,
        top_k=proposal_pool,
        conditions=conditions,
    )
    proposals = []
    for item in ranked:
        move = ElectronMove.parse(item)
        result = verify_electron_step(state, [move])
        if result["ok"]:
            proposals.append(
                {
                    "moves": [move.to_dict()],
                    "state_smiles": result["state_smiles"],
                    "logprob": item["logprob"],
                    "formal": result,
                }
            )
    if coupled:
        for left, right in combinations(ranked, 2):
            moves = [ElectronMove.parse(left), ElectronMove.parse(right)]
            touched_left = set(moves[0].source.atoms + moves[0].sink.atoms)
            touched_right = set(moves[1].source.atoms + moves[1].sink.atoms)
            if not (touched_left & touched_right):
                continue
            result = verify_electron_step(state, moves)
            if result["ok"]:
                proposals.append(
                    {
                        "moves": [move.to_dict() for move in moves],
                        "state_smiles": result["state_smiles"],
                        "logprob": left["logprob"] + right["logprob"],
                        "formal": result,
                    }
                )
    best = {}
    for item in proposals:
        previous = best.get(item["state_smiles"])
        if previous is None or item["logprob"] > previous["logprob"]:
            best[item["state_smiles"]] = item
    return sorted(
        best.values(),
        key=lambda item: item["logprob"],
        reverse=True,
    )[:max_candidates]


def infer(args) -> None:
    model = ForwardElectronExpert.load(args.checkpoint, device=args.device)
    output = []
    for row in read_jsonl(args.input):
        reactants = row.get("reactants") or row.get("state_smiles")
        target = row.get("products") or row.get("target_product")
        competitors = list(row.get("competitor_products") or [])
        if args.auto_competitors:
            generated = candidate_steps(
                model,
                reactants,
                proposal_pool=args.proposal_pool,
                max_candidates=args.auto_competitors * 3,
                coupled=True,
                conditions=row.get("conditions"),
            )
            for item in generated:
                value = item["state_smiles"]
                if same_molecule(value, target) or any(
                    same_molecule(value, old) for old in competitors
                ):
                    continue
                competitors.append(value)
                if len(competitors) >= args.auto_competitors:
                    break
        evidence = score_reaction(
            model,
            reactants,
            target,
            competitors,
            conditions=row.get("conditions"),
        )
        moves = list(row.get("moves") or [])
        formal = (
            verify_electron_step(row.get("state_smiles") or reactants, moves)
            if moves
            else {"ok": None, "code": "NO_GOLD_MOVES"}
        )
        output.append(
            {
                "id": row.get("id"),
                "reactants": reactants,
                "target_product": target,
                "competitor_products": competitors,
                "formal": formal,
                "forward_evidence": evidence.to_dict(),
                "gold_move_statistics": (
                    gold_move_statistics(
                        model,
                        row.get("state_smiles") or reactants,
                        moves,
                        row.get("conditions"),
                    )
                    if moves
                    else []
                ),
                "label": row.get("label", 1),
                "formal_label": row.get("formal_label"),
            }
        )
    write_jsonl(args.output, output)


def generate(args) -> None:
    model = ForwardElectronExpert.load(args.checkpoint, device=args.device)
    outputs = []
    for row in read_jsonl(args.input):
        initial = row.get("reactants") or row.get("state_smiles")
        target = row.get("products") or row.get("target_product") or args.target
        beams = [
            {
                "state_smiles": initial,
                "moves": [],
                "score": 0.0,
                "states": [initial],
            }
        ]
        solved = []
        for depth in range(args.max_steps):
            children = []
            for beam in beams:
                if target and same_molecule(beam["state_smiles"], target):
                    solved.append(beam)
                    continue
                proposals = candidate_steps(
                    model,
                    beam["state_smiles"],
                    proposal_pool=args.proposal_pool,
                    max_candidates=args.branch_limit,
                    coupled=not args.single_arrow_only,
                    conditions=row.get("conditions"),
                )
                for proposal in proposals:
                    if proposal["state_smiles"] in beam["states"]:
                        continue
                    forward_bonus = (
                        float(
                            torch.sigmoid(
                                model.reaction_score(
                                    initial,
                                    proposal["state_smiles"],
                                    conditions=row.get("conditions"),
                                )
                            ).detach()
                        )
                        if args.use_product_score
                        else 0.0
                    )
                    child = {
                        "state_smiles": proposal["state_smiles"],
                        "moves": beam["moves"] + [{"depth": depth, **proposal}],
                        "score": (
                            beam["score"]
                            + proposal["logprob"]
                            + args.product_score_weight * forward_bonus
                        ),
                        "states": beam["states"] + [proposal["state_smiles"]],
                    }
                    if target and same_molecule(child["state_smiles"], target):
                        solved.append(child)
                    else:
                        children.append(child)
            beams = sorted(
                children,
                key=lambda item: item["score"],
                reverse=True,
            )[: args.beam_size]
            if solved and args.stop_when_solved:
                break
            if not beams:
                break
        ranked = sorted(
            solved or beams,
            key=lambda item: item["score"],
            reverse=True,
        )[: args.num_return_sequences]
        outputs.append(
            {
                "id": row.get("id"),
                "initial_state": initial,
                "target_product": target,
                "solved": bool(solved),
                "paths": ranked,
            }
        )
    write_jsonl(args.output, outputs)


def expected_calibration_error(probabilities, labels, bins=10):
    if not probabilities:
        return 0.0
    total = len(probabilities)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if low <= probability < high
            or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(value for value, _ in members) / len(members)
        accuracy = sum(value for _, value in members) / len(members)
        result += len(members) / total * abs(confidence - accuracy)
    return result


def evaluate(args) -> None:
    rows = read_jsonl(args.predictions)
    formal_known = [
        row for row in rows if row.get("formal", {}).get("ok") is not None
    ]
    formal_pass = sum(bool(row["formal"]["ok"]) for row in formal_known)
    gold_moves = [
        item
        for row in rows
        for item in row.get("gold_move_statistics") or []
    ]
    target_top1 = [
        int((row.get("forward_evidence") or {}).get("target_rank") == 1)
        for row in rows
    ]
    selectivity = [
        int(
            (row.get("forward_evidence") or {}).get(
                "selectivity_margin",
                -math.inf,
            )
            > args.selectivity_threshold
        )
        for row in rows
        if (row.get("forward_evidence") or {}).get("selectivity_margin")
        is not None
    ]
    probabilities = [
        float((row.get("forward_evidence") or {}).get("target_score") or 0.0)
        for row in rows
    ]
    labels = [int(row.get("label", 1)) for row in rows]
    formal_labeled = [
        row
        for row in rows
        if row.get("formal_label") is not None
        and row.get("formal", {}).get("ok") is not None
    ]
    false_accept = sum(
        bool(row["formal"]["ok"]) and not bool(row["formal_label"])
        for row in formal_labeled
    )
    false_reject = sum(
        not bool(row["formal"]["ok"]) and bool(row["formal_label"])
        for row in formal_labeled
    )
    summary = {
        "n_reactions": len(rows),
        "formal_pass_rate": formal_pass / max(len(formal_known), 1),
        "move_top1": sum(bool(item.get("top1")) for item in gold_moves)
        / max(len(gold_moves), 1),
        "move_mrr": sum(
            1.0 / item["rank"] for item in gold_moves if item.get("rank")
        )
        / max(len(gold_moves), 1),
        "target_top1_rate": sum(target_top1) / max(len(target_top1), 1),
        "selectivity_pair_support_rate": sum(selectivity)
        / max(len(selectivity), 1),
        "brier": sum(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        )
        / max(len(labels), 1),
        "ece": expected_calibration_error(
            probabilities,
            labels,
            args.calibration_bins,
        ),
        "formal_false_acceptance_rate": false_accept
        / max(len(formal_labeled), 1),
        "formal_false_rejection_rate": false_reject
        / max(len(formal_labeled), 1),
        "selectivity_threshold": args.selectivity_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def evaluate_generation(args) -> None:
    rows = read_jsonl(args.predictions)
    paths = [path for row in rows for path in row.get("paths") or []]
    solved_rows = [row for row in rows if row.get("solved")]
    solved_paths = [
        path
        for row in rows
        for path in row.get("paths") or []
        if row.get("target_product")
        and same_molecule(path.get("state_smiles", ""), row["target_product"])
    ]
    endpoints = {
        path.get("state_smiles", "")
        for path in paths
        if path.get("state_smiles")
    }
    summary = {
        "n_targets": len(rows),
        "solve_rate": len(solved_rows) / max(len(rows), 1),
        "n_returned_paths": len(paths),
        "target_exact_path_rate": len(solved_paths) / max(len(paths), 1),
        "mean_path_steps": sum(len(path.get("moves") or []) for path in paths)
        / max(len(paths), 1),
        "unique_endpoints": len(endpoints),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--input", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    infer_parser = sub.add_parser(
        "infer",
        parents=[common],
        help="score labeled reactions and electron moves",
    )
    infer_parser.add_argument(
        "--auto-competitors",
        type=int,
        default=0,
        help="add formally reachable alternatives when explicit competitors are absent",
    )
    infer_parser.add_argument("--proposal-pool", type=int, default=24)
    infer_parser.set_defaults(func=infer)

    generate_parser = sub.add_parser(
        "generate",
        parents=[common],
        help="beam search over formally executable electron-flow steps",
    )
    generate_parser.add_argument("--target", default="")
    generate_parser.add_argument("--beam-size", type=int, default=8)
    generate_parser.add_argument("--branch-limit", type=int, default=12)
    generate_parser.add_argument("--proposal-pool", type=int, default=24)
    generate_parser.add_argument("--max-steps", type=int, default=4)
    generate_parser.add_argument("--num-return-sequences", type=int, default=5)
    generate_parser.add_argument("--single-arrow-only", action="store_true")
    generate_parser.add_argument("--stop-when-solved", action="store_true")
    generate_parser.add_argument("--use-product-score", action="store_true")
    generate_parser.add_argument("--product-score-weight", type=float, default=0.5)
    generate_parser.set_defaults(func=generate)

    eval_parser = sub.add_parser(
        "eval",
        help="aggregate verifier, move, selectivity and calibration metrics",
    )
    eval_parser.add_argument("--predictions", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--selectivity-threshold", type=float, default=0.1)
    eval_parser.add_argument("--calibration-bins", type=int, default=10)
    eval_parser.set_defaults(func=evaluate)

    gen_eval = sub.add_parser(
        "eval-generation",
        help="evaluate generated forward electron-flow paths",
    )
    gen_eval.add_argument("--predictions", type=Path, required=True)
    gen_eval.add_argument("--output", type=Path, required=True)
    gen_eval.set_defaults(func=evaluate_generation)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
