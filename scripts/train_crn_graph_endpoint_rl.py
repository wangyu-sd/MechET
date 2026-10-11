#!/usr/bin/env python3
"""Bounded train-only endpoint RL for CRN-JEPA FLOW; Qwen stays frozen.

Each training reaction supplies a verified reference prefix. Choose a late
electron-flow frontier, sample a stochastic slate from the graph policy's
bounded proposal distribution, execute alternatives and continue with frozen
Qwen + graph until terminal. Exact STRUCTURAL precursor recovery is reward 1;
reference-successor match supplies weaker local support. If the slate has no
supported option, do ONLY reference replay SFT, never all-negative GRPO.
This is a group policy-gradient surrogate over a truncated slate, not PPO.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.crn_graph_bridge import file_sha256, graph_event_coordinates
from mechet.graph_endpoint_rl import (
    VerifiedBranch, graph_policy_gradient_loss, success_gated_advantages,
)
from mechet.endpoints import (
    reference_structural_precursor, split_precursor_endpoints, structural_exact,
)
from mechet.in_place_grounded_flow import mapped_atom_numbers, mapped_state_signature
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.build_natural_language_event_sft import convert_row
from scripts.run_crn_qwen_hybrid import HybridRuntime, parser, prepare_args
from scripts.run_natural_language_value_search import (
    Action, Node, execute, product_only_private_state, read_selected, visible,
)


def verified_reference_frontiers(row: dict):
    """Replay genuine training-reference decisions, never expose them to Qwen."""
    root = str(row["target_smiles"])
    if not mapped_atom_numbers(root):
        raise ValueError("train reference must have a privately mapped product")
    node = Node(
        target=visible(root), state=root,
        next_map=max(mapped_atom_numbers(root)) + 1,
        visited={visible(root)},
    )
    frontiers = []
    for decision in convert_row(row):
        function = decision["messages"][2]["tool_calls"][0]["function"]
        name = function["name"]
        arguments = dict(function["arguments"])
        moves = None
        if name == "apply_electron_flow":
            compiled = compile_event_arguments(node.state, arguments)
            if not any(move.get("mode") == "BE_DELTA" for move in compiled):
                moves = tuple(compiled)
        action = Action(name=name, arguments=arguments, raw="<train_only_gold>",
                        logprob=0.0, tokens=1)
        child, error = execute(node, action, max_imports=32)
        if child is None:
            raise ValueError(f"reference replay failed: {error}")
        if moves is not None:
            frontiers.append((node, child.state, moves))
        node = child
    if not node.terminal:
        raise ValueError("reference never reached terminal")
    return frontiers, root


def sample_distinct_slate(candidates, count: int, rng: random.Random, temperature: float):
    """Gumbel weighted sampling without replacement within model top-32 slate."""
    if count <= 0 or temperature <= 0:
        raise ValueError("invalid slate sampling parameters")
    def priority(candidate):
        u = min(1.0 - 1e-9, max(1e-9, rng.random()))
        return candidate.logprob / temperature - math.log(-math.log(u))
    return sorted(candidates, key=priority, reverse=True)[:count]


def continue_to_terminal(runtime: HybridRuntime, node: Node, args, max_steps: int):
    """A gold-free rollout; never reads the expected precursor."""
    for _ in range(max_steps):
        if node.terminal:
            return node
        proposed = runtime.proposals(
            node, candidates=1, max_new_tokens=args.max_import_tokens,
            compact_history=args.compact_history,
        )
        if not proposed:
            return None
        child = None
        for action in proposed:
            child, _ = execute(
                node, action, max_imports=args.max_imports,
                reject_target_retained_finish=True,
            )
            if child is not None:
                break
        if child is None:
            return None
        node = child
    return node if node.terminal else None


def main():
    import torch
    p = parser()
    p.description = __doc__
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--rl-updates", type=int, default=64)
    p.add_argument("--slate", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--continuation-horizon", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--reference-weight", type=float, default=0.1)
    args = prepare_args(p.parse_args())
    sys.path.insert(0, str((args.crn_root / "src").resolve()))
    from crn_jepa.grounded_actions import GroundedMove, MechETGroundedInventory
    from crn_jepa.models.grounded_graph import build_grounded_graph
    from crn_jepa.models.pr18_full_corpus_objectives import teacher_forced_from_tokens
    if args.data.name != "train.jsonl" or "test" in str(args.data).lower():
        raise ValueError("endpoint RL only accepts an authenticated train.jsonl")
    observed_sha = file_sha256(args.data)
    if observed_sha != args.source_sha256:
        raise ValueError("training source SHA-256 mismatch")
    if args.rl_updates < 1 or args.slate < 2 or args.slate > 32:
        raise ValueError("invalid RL budget")
    if args.continuation_horizon < 1 or args.continuation_horizon > 40:
        raise ValueError("invalid RL continuation horizon")
    if args.graph_rl_head:
        raise ValueError("v1 RL must initialize from the frozen CRN-JEPA checkpoint")
    if not 0.0 < args.reference_weight <= 1.0:
        raise ValueError("v1 requires positive verified replay weight")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("RL output directory must be initially empty")

    rng = random.Random(args.seed)
    rows = read_selected(args.data, args.sample_reactions, args.seed)
    if len(rows) != args.sample_reactions:
        raise ValueError("insufficient authenticated train rows")
    runtime = HybridRuntime(args, 0)
    # Freeze graph backbone and all JEPA targets. Only electronic source/sink/
    # commit heads and existing direction-conditioned policy layers are updated.
    world = runtime.graph.model
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    for name, parameter in world.policy.named_parameters():
        if not name.startswith("graph_encoder."):
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [p for p in world.policy.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    expected = {}
    stats = dict(selected=len(rows), replay_failures=0, no_flow=0,
                 effective_rl_groups=0, reference_only_groups=0,
                 reference_steps=0, candidate_count=0,
                 executable_candidates=0, endpoint_success_candidates=0,
                 reference_successor_candidates=0, updated=0)
    audit = []
    for row in rows[:args.rl_updates]:
        row_id = str(row["source_id"])
        try:
            frontiers, root = verified_reference_frontiers(row)
        except (ValueError, KeyError) as exc:
            stats["replay_failures"] += 1
            audit.append({"source_id": row_id, "status": "replay_failure",
                          "error": str(exc)})
            continue
        if not frontiers:
            stats["no_flow"] += 1
            continue
        # Curriculum: favor last three executable FLOW decisions. This
        # increases exact endpoint signal without changing inference prompts.
        depth_from_end = rng.randrange(min(3, len(frontiers)))
        anchor, reference_state, reference_moves = frontiers[-1 - depth_from_end]
        try:
            # CRN-JEPA cannot score BE_DELTA in this v1 action decoder.
            reference_actions = tuple(GroundedMove.from_raw(m) for m in reference_moves)
            event_depth, remaining_budget = graph_event_coordinates(
                anchor.actions, args.graph_initial_budget,
            )
            slate = runtime.graph.candidates(
                anchor.state, limit=32,
                event_depth=event_depth, remaining_budget=remaining_budget,
                beam_size=args.graph_beam,
                source_width=args.graph_source_width,
                sink_width=args.graph_sink_width,
                move_width=args.graph_move_width,
            )
            candidates = sample_distinct_slate(
                slate, args.slate, rng, args.temperature
            )
            if not candidates:
                stats["no_flow"] += 1
                continue
            reference_endpoint = reference_structural_precursor(dict(row))
            outcomes = []
            for candidate in candidates:
                action = Action(
                    name="apply_electron_flow",
                    arguments=candidate.arguments,
                    raw="<crn_jepa_sampled_flow>",
                    logprob=candidate.logprob, tokens=max(1, len(candidate.moves)),
                )
                child, _ = execute(anchor, action, max_imports=32)
                accepted = child is not None
                equivalent = accepted and (
                    mapped_state_signature(child.state)
                    == mapped_state_signature(reference_state)
                )
                terminal = continue_to_terminal(
                    runtime, child, args, args.continuation_horizon
                ) if accepted else None
                exact = bool(
                    terminal is not None and reference_endpoint and
                    structural_exact(
                        split_precursor_endpoints(terminal.state, root).structural,
                        reference_endpoint,
                    )
                )
                outcomes.append(VerifiedBranch(
                    accepted=accepted, exact_terminal=exact,
                    reference_successor=bool(equivalent),
                ))
                stats["candidate_count"] += 1
                stats["executable_candidates"] += int(accepted)
                stats["endpoint_success_candidates"] += int(exact)
                stats["reference_successor_candidates"] += int(equivalent)

            advantages = success_gated_advantages(outcomes)
            world.policy.train()
            world.policy.graph_encoder.eval()
            graph = build_grounded_graph(anchor.state, None)
            inventory = MechETGroundedInventory(anchor.state)
            nodes, context, prefix = world.policy.encode_directional_observation(
                graph, event_depth=event_depth, remaining_budget=remaining_budget,
                direction="retrosynthetic_inverse",
            )
            nlls = [
                teacher_forced_from_tokens(
                    world.policy, graph, inventory,
                    tuple(GroundedMove.from_raw(m) for m in candidate.moves),
                    nodes, context, prefix,
                )
                for candidate in candidates
            ]
            reference_nll = teacher_forced_from_tokens(
                world.policy, graph, inventory,
                reference_actions, nodes, context, prefix,
            )
            loss = graph_policy_gradient_loss(
                nlls, advantages, reference_nll,
                reference_replay_weight=args.reference_weight,
            )
            if any(abs(a) > 1e-9 for a in advantages):
                stats["effective_rl_groups"] += 1
            else:
                stats["reference_only_groups"] += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in world.policy.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            stats["updated"] += 1
            stats["reference_steps"] += 1
            audit.append({
                "source_id": row_id, "status": "updated",
                "depth_from_reference_end": depth_from_end,
                "graph_event_depth": event_depth,
                "graph_remaining_budget": remaining_budget,
                "candidate_rewards": [r.reward for r in outcomes],
                "advantages": list(advantages),
                "loss": float(loss.detach().item()),
            })
        except (ValueError, RuntimeError, KeyError) as exc:
            audit.append({"source_id": row_id, "status": "skipped",
                          "error": f"{type(exc).__name__}:{exc}"})
            continue
        finally:
            world.policy.eval()
    if not stats["updated"]:
        raise RuntimeError("no optimizer updates; no unverified checkpoint saved")
    head = {
        key: value.detach().cpu().clone()
        for key, value in world.policy.state_dict().items()
        if not key.startswith("graph_encoder.")
    }
    checkpoint_path = args.output / "graph_endpoint_rl_head.pt"
    torch.save({
        "policy_head": head, "base_checkpoint_sha256": runtime.graph.sha256,
        "train_source_sha256": observed_sha, "qwen_frozen": True,
        "rl_algorithm": "success_gated_truncated_slate_policy_gradient_v1",
    }, checkpoint_path)
    with (args.output / "training_groups.jsonl").open("w") as sink:
        for item in audit:
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {
        "training_only": True,
        "graph_pretrained_sha256": runtime.graph.sha256,
        "graph_rl_head_sha256": file_sha256(checkpoint_path),
        "frozen_qwen_model": args.model,
        "frozen_qwen_revision": args.model_revision,
        "frozen_qwen_adapter_sha256":
            file_sha256(args.policy_adapter / "adapter_model.safetensors"),
        "source_sha256": observed_sha,
        "stats": stats,
        "effective_rl_updates": stats["effective_rl_groups"],
        "note": "If effective_rl_updates=0, checkpoint was reference-replay SFT only, NOT endpoint RL evidence.",
        "locked_test_accessed": False,
    }
    (args.output / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
