#!/usr/bin/env python3
"""Graph-EARHO: execution-anchored, receding-horizon graph-only optimization.

Not a generic last-three-gold-FLOW group-RL smoke. The frozen Qwen actor
first rolls out from a product-only mapping to locate its earliest executed
divergence. A graph-only update is causally attributable only if both
reference and chosen action are FLOW at that first divergence. If Qwen fails
first, a separately counted reference-conditional graph rehearsal is allowed
to retain local FLOW competence, but is not misreported as an on-policy graph
frontier. The curriculum extends its verified correction horizon as support
grows, and the optimizer pools actions by their EXECUTED successor molecules.
All rewards are training-only. The executor is unchanged.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import sys
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.crn_graph_bridge import file_sha256
from mechet.endpoints import reference_structural_precursor, split_precursor_endpoints, structural_exact
from mechet.graph_earho import (
    AdaptiveHorizon, ExecutedAlternative, Transition,
    VerifiedOutcome, first_executed_divergence,
    pool_executed_successors, select_receding_flow_anchor,
    Divergence,
)
from mechet.graph_endpoint_rl import (
    grounded_event_log_probability, successor_class_policy_loss,
    truncated_slate_reference_kl,
)
from mechet.in_place_grounded_flow import mapped_atom_numbers
from mechet.natural_language_anchor_branch_rl import contains_unchanged_target
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.build_natural_language_event_sft import convert_row
from scripts.pr83_resume import prepare_training_resume
from scripts.run_crn_qwen_hybrid import HybridRuntime, parser, prepare_args
from scripts.run_natural_language_value_search import (
    Action, Node, execute, product_only_private_state, read_selected, visible,
)


def replay_train_trace(row):
    """Reference replay is private training information; never a model prompt."""
    mapped_target = str(row["target_smiles"])
    if not mapped_atom_numbers(mapped_target):
        raise ValueError("train-only reference product lacks atom maps")
    initial = Node(
        target=visible(mapped_target), state=mapped_target,
        next_map=max(mapped_atom_numbers(mapped_target)) + 1,
        visited={visible(mapped_target)},
    )
    nodes = [initial]
    reference = []
    graph_moves = {}
    decisions = convert_row(row)
    for index, decision in enumerate(decisions):
        fn = decision["messages"][2]["tool_calls"][0]["function"]
        name, arguments = str(fn["name"]), dict(fn["arguments"])
        label = name
        if name == "apply_electron_flow":
            moves = tuple(compile_event_arguments(nodes[-1].state, arguments))
            if len(moves) == 1 and moves[0].get("mode") == "BE_DELTA":
                label = "BE_DELTA"  # no existing CRN source/sink head for this family
            else:
                graph_moves[index] = moves
        action = Action(name, arguments, "<train_reference>", 0.0, 1)
        child, error = execute(nodes[-1], action, max_imports=32)
        if child is None:
            raise ValueError(f"reference {index} rejected: {error}")
        reference.append(Transition(
            name=label, accepted=True,
            successor_key=visible(child.state), terminal=child.terminal,
        ))
        nodes.append(child)
    if not nodes[-1].terminal:
        raise ValueError("reference trace has no verified final terminal")
    if not graph_moves:
        raise ValueError("reference has no CRN-compatible FLOW decision")
    return {
        "nodes": nodes, "reference": reference, "graph_moves": graph_moves,
        "original_target": mapped_target,
    }


def observe_policy_divergence(runtime, trace, args):
    """Return first chemical divergence AND the actual autonomous pre-error Node.

    A replay-verified state is useful as a private training label, never a
    substitute for the model-owned state at its first error. This retains
    Qwen-imported chemical context, action history and private atom maps.
    """
    product = visible(trace["original_target"])
    state = product_only_private_state(product)
    node = Node(
        target=product, state=state,
        next_map=max(mapped_atom_numbers(state)) + 1,
        visited={product},
    )
    predictions = []
    for expected in trace["reference"]:
        before = node
        proposed = runtime.proposals(
            node, candidates=1, max_new_tokens=args.max_import_tokens,
            compact_history=args.compact_history,
        )
        action = proposed[0] if proposed else None
        if action is None:
            predictions.append(
                Transition("NO_ACTION", False, None, error="NO_MODEL_PROPOSAL")
            )
            return first_executed_divergence(
                trace["reference"][:len(predictions)], predictions
            ), before
        child, error = execute(
            node, action, max_imports=args.max_imports,
            reject_target_retained_finish=True,
        )
        trial = Transition(
            name=action.name, accepted=child is not None,
            successor_key=visible(child.state) if child else None,
            terminal=bool(child and child.terminal), error=error,
        )
        predictions.append(trial)
        if (
            child is None or trial.successor_key != expected.successor_key
            or trial.terminal != expected.terminal
        ):
            return first_executed_divergence(
                trace["reference"][:len(predictions)], predictions
            ), before
        node = child
    return first_executed_divergence(trace["reference"], predictions), None


def align_reference_moves(reference_mapped_state, actor_mapped_state, moves):
    """Remap train-only reference electron containers onto the model's maps.

    Private map integers carry no semantics. We require exact map-free
    chemical identity and a UNIQUE grounded action after atom-isomorphism
    matching. Symmetry-ambiguous gold pointers are excluded from the
    supervised fallback (not from the sampled actor's endpoint rewards).
    """
    from rdkit import Chem
    from crn_jepa.grounded_actions import GroundedMove, unordered_event_key

    if visible(reference_mapped_state) != visible(actor_mapped_state):
        return None
    reference = Chem.MolFromSmiles(reference_mapped_state)
    current = Chem.MolFromSmiles(actor_mapped_state)
    if reference is None or current is None:
        return None
    old_atom_maps = tuple(atom.GetAtomMapNum() for atom in reference.GetAtoms())
    new_atom_maps = tuple(atom.GetAtomMapNum() for atom in current.GetAtoms())
    if (any(m <= 0 for m in old_atom_maps + new_atom_maps)
            or len(old_atom_maps) != len(new_atom_maps)):
        return None
    for atom in reference.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom in current.GetAtoms():
        atom.SetAtomMapNum(0)
    matches = current.GetSubstructMatches(
        reference, uniquify=False, useChirality=True, maxMatches=33,
    )
    if not matches or len(matches) >= 33:
        return None
    variants = {}
    for match in matches:
        remap = {
            old_atom_maps[i]: new_atom_maps[match[i]]
            for i in range(len(old_atom_maps))
        }
        transformed = []
        for raw in moves:
            move = GroundedMove.from_raw(raw)
            def container(value):
                return {
                    "kind": value.kind,
                    "atoms": [remap[a] for a in value.atom_maps],
                }
            transformed.append(
                GroundedMove.from_raw({
                    "source": container(move.source),
                    "sink": container(move.sink),
                    "electrons": move.electrons,
                })
            )
        signature = unordered_event_key(transformed)
        variants[signature] = tuple(transformed)
        if len(variants) > 1:
            return None  # don't choose one arbitrary symmetric gold atom
    return next(iter(variants.values()))


def sample_slate(candidates, k: int, rng, temperature: float):
    """Distinct Gumbel top-K from the graph's bounded ranked event slate."""
    if not 1 <= k <= len(candidates) or temperature <= 0:
        raise ValueError("insufficient candidates or invalid temperature")
    keys = []
    for index, item in enumerate(candidates):
        u = min(1.0 - 1e-9, max(1e-9, rng.random()))
        score = item.logprob / temperature - math.log(-math.log(u))
        keys.append((score, index))
    keys.sort(reverse=True)
    return [candidates[i] for _, i in keys[:k]]


@dataclass(frozen=True)
class ContinuationResult:
    terminals: tuple[Node, ...]
    censored: bool
    attempted: int


def bounded_continuation(runtime, start, args, *, decisions: int):
    """Gold-free, TYPE-COMPARABLE bounded breadth-first search.

    A parent may propose several *graph* FLOW actions ranked by the same
    graph policy, but IMPORT and FINISH have only the frozen Qwen choice.
    Across different parents or Qwen/Graph action families we NEVER compare
    raw Qwen token-NLL against graph move-NLL: breadth-first order and
    per-parent candidate rank determine a reproducible equal-budget beam.

    Censored means the finite search failed to exhaust plausible branches.
    Lack of endpoint evidence after censoring must not become negative RL.
    """
    width = args.rollout_beam_width
    if start.terminal:
        return ContinuationResult((start,), False, 0)
    beam = [start]
    terminals = []
    censored = decisions <= 0
    attempts = 0
    for _ in range(max(0, decisions)):
        next_states = {}
        incomplete = False
        for parent in beam:
            if len(parent.actions) >= args.max_decisions:
                incomplete = True
                continue
            proposals = runtime.proposals(
                parent, candidates=1,
                max_new_tokens=args.max_import_tokens,
                compact_history=args.compact_history,
            )
            if not proposals:
                incomplete = True
                continue
            for action in proposals:
                attempts += 1
                child, _ = execute(
                    parent, action, max_imports=args.max_imports,
                    reject_target_retained_finish=True,
                )
                if child is None:
                    incomplete = True
                    continue
                if child.terminal:
                    terminals.append(child)
                else:
                    key = (
                        visible(child.state),
                        tuple(record["name"] for record in child.actions)
                        if args.compact_history else (),
                    )
                    # First admissible successor in model-ranked per-parent
                    # order wins. Don't rank heterogeneous neural scores.
                    next_states.setdefault(key, child)
        if len(next_states) > width:
            incomplete = True
        censored = censored or incomplete
        beam = list(next_states.values())[:width]
        if not beam:
            break
    if beam:
        censored = True
    return ContinuationResult(tuple(terminals), censored, attempts)


def graph_head_parameters(world):
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    for name, parameter in world.policy.named_parameters():
        if not name.startswith("graph_encoder."):
            parameter.requires_grad_(True)
    return [p for p in world.policy.parameters() if p.requires_grad]


def main() -> int:
    import torch
    cli = parser()
    cli.description = __doc__
    cli.add_argument("--source-sha256", required=True)
    cli.add_argument("--rl-updates", type=int, default=64)
    cli.add_argument("--slate", type=int, default=8)
    cli.add_argument("--temperature", type=float, default=1.0)
    cli.add_argument("--continuation-horizon", type=int, default=8,
                     help="max adaptive correction horizon; initial horizon is two")
    cli.add_argument("--rollout-beam-width", type=int, default=2)
    cli.add_argument("--curriculum-round", type=int, default=16)
    cli.add_argument("--min-effective-groups", type=int, default=4)
    cli.add_argument("--productive-pass-rate", type=float, default=0.1)
    cli.add_argument("--full-episode-fraction", type=float, default=0.25)
    cli.add_argument("--learning-rate", type=float, default=5e-6)
    cli.add_argument("--reference-weight", type=float, default=0.1)
    cli.add_argument("--slate-kl-weight", type=float, default=0.02,
                     help="frozen CRN policy KL on identical sampled event slate")
    cli.add_argument("--save-every", type=int, default=4)
    args = prepare_args(cli.parse_args())
    # A genuine bounded FLOW beam requires more than one same-policy option.
    # K=1 held-out evaluation is separate and retains its own declared budget.
    args.graph_flow_topk = max(args.graph_flow_topk, args.rollout_beam_width)
    sys.path.insert(0, str((args.crn_root / "src").resolve()))
    from crn_jepa.grounded_actions import GroundedMove, MechETGroundedInventory
    from crn_jepa.models.grounded_graph import build_grounded_graph
    from crn_jepa.models.pr18_full_corpus_objectives import teacher_forced_from_tokens
    # External model modules are imported only after the selected checkout is
    # explicitly in sys.path; the default package is not installed in MechET.
    if (args.data.name != "train.jsonl" or "test" in str(args.data).lower()
        or file_sha256(args.data) != args.source_sha256):
        raise ValueError("Graph-EARHO only accepts SHA-authenticated train data")
    if not (1 <= args.rl_updates <= args.sample_reactions
            and 2 <= args.slate <= 32 and 1 <= args.rollout_beam_width <= 4
            and 2 <= args.continuation_horizon <= 40
            and 1 <= args.curriculum_round and args.save_every >= 1
            and 0 < args.reference_weight <= 1.0
            and 0 <= args.slate_kl_weight <= 1.0
            and args.temperature > 0):
        raise ValueError("invalid bounded Graph-EARHO training contract")
    if args.graph_rl_head:
        raise ValueError("Graph-EARHO starts from the untouched pretrained policy")
    rows = read_selected(args.data, args.sample_reactions, args.seed)
    if len(rows) != args.sample_reactions:
        raise ValueError("train source has too few selected reaction rows")
    code_paths = (
        "scripts/train_crn_graph_earho.py", "src/mechet/graph_earho.py",
        "src/mechet/graph_endpoint_rl.py", "src/mechet/crn_graph_bridge.py",
        "scripts/pr83_resume.py", "scripts/run_crn_qwen_hybrid.py",
        "scripts/run_natural_language_value_search.py",
    )
    manifest = {
        "schema_version": 2, "algorithm": "graph_earho_executed_successor_v2",
        "train_source_sha256": args.source_sha256,
        "train_source_path": str(args.data.resolve()),
        "selected_ids": [str(row["id"]) for row in rows],
        "qwen_model": args.model, "qwen_revision": args.model_revision,
        "qwen_adapter_sha256": file_sha256(
            args.policy_adapter / "adapter_model.safetensors"
        ),
        "crn_pretrained_sha256": file_sha256(args.graph_checkpoint),
        "settings": {
            "sample_reactions": args.sample_reactions,
            "seed": args.seed, "rl_updates": args.rl_updates,
            "slate": args.slate, "temperature": args.temperature,
            "continuation_horizon": args.continuation_horizon,
            "rollout_beam_width": args.rollout_beam_width,
            "graph_flow_topk": args.graph_flow_topk,
            "graph_beam": args.graph_beam,
            "graph_source_width": args.graph_source_width,
            "graph_sink_width": args.graph_sink_width,
            "graph_move_width": args.graph_move_width,
            "curriculum_round": args.curriculum_round,
            "min_effective_groups": args.min_effective_groups,
            "productive_pass_rate": args.productive_pass_rate,
            "full_episode_fraction": args.full_episode_fraction,
            "learning_rate": args.learning_rate,
            "reference_weight": args.reference_weight,
            "slate_kl_weight": args.slate_kl_weight,
            "save_every": args.save_every,
            "compact_history": args.compact_history,
            "max_context": args.max_context, "max_decisions": args.max_decisions,
            "max_imports": args.max_imports,
            "max_import_tokens": args.max_import_tokens,
        },
        "code_sha256": {
            path: file_sha256(ROOT / path) for path in code_paths
        },
    }
    if manifest["crn_pretrained_sha256"] != args.graph_sha256:
        raise ValueError("CRN checkpoint does not match requested SHA")
    saved_file = prepare_training_resume(
        args.output, manifest=manifest, resume=args.resume,
    )
    rng = random.Random(args.seed)
    runtime = HybridRuntime(args, 0)
    world = runtime.graph.model
    # Frozen reference MUST be copied BEFORE restoring a resumed adapted actor.
    # This is the original CRN policy, not a moving/training-time teacher.
    frozen_policy = copy.deepcopy(world.policy).eval()
    for frozen_param in frozen_policy.parameters():
        frozen_param.requires_grad_(False)
    parameters = graph_head_parameters(world)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    horizon = AdaptiveHorizon(
        current=2, maximum=args.continuation_horizon,
        min_effective_groups=args.min_effective_groups,
        min_productive_fraction=args.productive_pass_rate,
        full_episode_fraction=args.full_episode_fraction,
    )
    stats = {
        "selected": len(rows), "processed": 0, "replay_failed": 0,
        "graph_correctable_frontiers": 0, "controller_blocked_frontiers": 0,
        "no_divergence": 0, "reference_conditional_frontiers": 0,
        "full_product_anchored": 0, "full_product_requested_but_not_eligible": 0,
        "insufficient_slate": 0, "sampled_actions": 0,
        "unique_executed_successors": 0, "pooled_duplicate_actions": 0,
        "executor_accepted": 0, "endpoint_positive_classes": 0,
        "reference_successor_only_classes": 0,
        "effective_rl_groups": 0, "conditional_effective_groups": 0,
        "on_policy_endpoint_positive_classes": 0,
        "conditional_endpoint_positive_classes": 0,
        "reference_only_groups": 0,
        "optimizer_updates": 0, "exception_skips": 0,
        "curriculum_promotions": 0,
        "censored_successor_classes": 0, "wrong_terminal_classes": 0,
        "replay_label_unavailable": 0, "replay_only_updates": 0,
        "slate_kl_total": 0.0, "full_horizon_branches": 0,
    }
    audit = []
    round_status = {"total": 0, "effective": 0, "productive": 0}
    cursor = 0
    checkpoint = args.output / "latest_train_state.pt"
    if saved_file:
        state = torch.load(saved_file, map_location="cpu", weights_only=False)
        if state.get("manifest") != manifest:
            raise ValueError("resumed Graph-EARHO lineage mismatch")
        cursor = int(state["next_index"])
        if not 0 <= cursor <= args.rl_updates:
            raise ValueError("invalid saved cursor")
        head = dict(state["policy_head"])
        current = world.policy.state_dict()
        if not head or any(
            key.startswith("graph_encoder.") or key not in current
            or current[key].shape != value.shape
            for key, value in head.items()
        ):
            raise ValueError("saved policy head cannot be loaded")
        current.update(head)
        world.policy.load_state_dict(current, strict=True)
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng_state"])
        stats = dict(state["stats"])
        audit = list(state["audit"])
        horizon = AdaptiveHorizon(**state["horizon"])
        round_status = dict(state["round_status"])

    def save(next_index):
        state = {
            "manifest": manifest, "next_index": next_index,
            "optimizer": optimizer.state_dict(),
            "policy_head": {
                key: weight.detach().cpu().clone()
                for key, weight in world.policy.state_dict().items()
                if not key.startswith("graph_encoder.")
            },
            "rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "stats": stats, "audit": audit,
            "horizon": {
                "current": horizon.current, "maximum": horizon.maximum,
                "min_effective_groups": horizon.min_effective_groups,
                "min_productive_fraction": horizon.min_productive_fraction,
                "full_episode_fraction": horizon.full_episode_fraction,
            },
            "round_status": round_status,
        }
        tmp = checkpoint.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, checkpoint)

    if saved_file:
        saved = torch.load(saved_file, map_location="cpu", weights_only=False)
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    else:
        save(0)

    for index in range(cursor, args.rl_updates):
        row = rows[index]
        reaction_id = str(row["source_id"])
        state_was_valid = False
        abort = False
        try:
            trace = replay_train_trace(row)
            divergence, actor_anchor = observe_policy_divergence(runtime, trace, args)
            if divergence.index is None:
                stats["no_divergence"] += 1
                audit.append({
                    "reaction_id": reaction_id, "status": "reference_aligned",
                    "reason": divergence.reason,
                })
                continue
            # The graph policy cannot repair a wrong Qwen route/import/finish.
            # Do NOT replace this error with a reference state and call it RL.
            if not divergence.graph_correctable or actor_anchor is None:
                stats["controller_blocked_frontiers"] += 1
                audit.append({
                    "reaction_id": reaction_id, "status": "controller_blocked",
                    "first_index": divergence.index,
                    "reason": divergence.reason,
                })
                continue
            stats["graph_correctable_frontiers"] += 1
            gold_names = [item.name for item in trace["reference"]]
            full_requested = rng.random() < horizon.full_episode_fraction
            choice = select_receding_flow_anchor(
                divergence=divergence, reference_names=gold_names,
                horizon=horizon.current, full_product=full_requested,
            )
            if choice is None or choice.decision_index not in trace["graph_moves"]:
                stats["insufficient_slate"] += 1
                continue
            # CRITICAL: the actor's REAL pre-error state and accepted prefix,
            # not the reference state near the end of the gold trajectory.
            anchor = actor_anchor
            gold_state = trace["nodes"][choice.decision_index]
            successor_ref = trace["nodes"][choice.decision_index + 1]
            if visible(anchor.state) != visible(gold_state.state):
                raise ValueError("actor/reference prefix diverged before selected frontier")
            gold_moves = align_reference_moves(
                gold_state.state, anchor.state,
                trace["graph_moves"][choice.decision_index],
            )
            if gold_moves is None:
                stats["replay_label_unavailable"] += 1
            if choice.full_product:
                stats["full_product_anchored"] += 1
            # The full-budget branch is a REAL autonomous prefix followed
            # by bounded independent continuation, not a gold-prefixed reset.
            if full_requested:
                stats["full_horizon_branches"] += 1
            slate = runtime.graph.candidates(
                anchor.state, limit=32,
                beam_size=args.graph_beam,
                source_width=args.graph_source_width,
                sink_width=args.graph_sink_width,
                move_width=args.graph_move_width,
            )
            if len(slate) < args.slate:
                stats["insufficient_slate"] += 1
                audit.append({
                    "reaction_id": reaction_id, "status": "insufficient_slate",
                    "found": len(slate), "needed": args.slate,
                })
                continue
            candidates = sample_slate(slate, args.slate, rng, args.temperature)
            alternatives = []
            child_by_index = {}
            for k, candidate in enumerate(candidates):
                a = Action(
                    name="apply_electron_flow", arguments=candidate.arguments,
                    raw="<graph_earho_sample>", logprob=candidate.logprob,
                    tokens=max(1, len(candidate.moves)),
                )
                child, error = execute(
                    anchor, a, max_imports=args.max_imports,
                    reject_target_retained_finish=True,
                )
                child_by_index[k] = child
                stats["sampled_actions"] += 1
                stats["executor_accepted"] += int(child is not None)
                alternatives.append(ExecutedAlternative(
                    candidate_index=k, accepted=child is not None,
                    successor_key=visible(child.state) if child else None,
                    logprob=float(candidate.logprob), error=str(error),
                ))
            groups = pool_executed_successors(alternatives)
            stats["unique_executed_successors"] += sum(g.accepted for g in groups)
            stats["pooled_duplicate_actions"] += len(candidates) - len(groups)
            endpoint = reference_structural_precursor(dict(row))
            outcomes = []
            group_rows = []
            for group in groups:
                child = child_by_index[group.representative_index]
                reference_hit = bool(
                    child is not None
                    and visible(child.state) == visible(successor_ref.state)
                )
                # H bounds forward exploration FROM the actual first error.
                # A controlled subset gets the full remaining decision budget.
                remaining = max(0, min(
                    args.max_decisions - len(anchor.actions) - 1,
                    args.max_decisions if full_requested else horizon.current - 1,
                ))
                continuation = (
                    bounded_continuation(runtime, child, args, decisions=remaining)
                    if child is not None else ContinuationResult((), False, 0)
                )
                endings = continuation.terminals
                endpoint_hit = any(
                    end.terminal
                    and not contains_unchanged_target(end.state, anchor.target)
                    and bool(endpoint)
                    and structural_exact(
                        split_precursor_endpoints(
                            end.state, trace["original_target"]
                        ).structural,
                        endpoint,
                    )
                    for end in endings
                )
                # A nonterminating/cutoff search is CENSORED, not a failure
                # of the initial accepted electron-flow action.
                no_op = bool(
                    endings and not endpoint_hit and any(
                        contains_unchanged_target(e.state, anchor.target)
                        for e in endings
                    )
                )
                wrong_terminal = bool(
                    endings and not endpoint_hit and not continuation.censored
                    and not no_op and not reference_hit
                )
                outcome = VerifiedOutcome(
                    executed=child is not None,
                    endpoint_exact=bool(endpoint_hit),
                    reference_successor=reference_hit,
                    wrong_terminal=wrong_terminal,
                    target_retained_noop=no_op and not reference_hit,
                    exhausted=bool(child is not None and continuation.censored),
                )
                outcomes.append(outcome)
                stats["censored_successor_classes"] += int(outcome.unresolved)
                stats["wrong_terminal_classes"] += int(wrong_terminal)
                group_rows.append({
                    "representative": group.representative_index,
                    "members": list(group.member_indices),
                    "successor_sha256": (
                        __import__("hashlib").sha256(
                            group.key.encode("utf-8")
                        ).hexdigest() if group.accepted else None
                    ),
                    "executed": bool(child),
                    "endpoint_exact": endpoint_hit,
                    "reference_successor": reference_hit,
                    "wrong_terminal": wrong_terminal,
                    "censored": outcome.unresolved,
                    "continued_terminals": len(endings),
                    "continuation_calls": continuation.attempted,
                })
            if not any(o.positive for o in outcomes):
                stats["reference_only_groups"] += 1
            adv = successor_group_advantages(outcomes)
            supported_contrast = any(abs(v) > 1e-12 for v in adv)
            endpoint_positive = sum(o.endpoint_exact for o in outcomes)
            if supported_contrast and train_from_policy:
                stats["effective_rl_groups"] += 1
            elif supported_contrast:
                stats["conditional_effective_groups"] += 1
            else:
                stats["reference_only_groups"] += int(any(o.positive for o in outcomes))
            stats["endpoint_positive_classes"] += endpoint_positive
            stats["reference_successor_only_classes"] += sum(
                o.reference_successor and not o.endpoint_exact for o in outcomes
            )
            if train_from_policy:
                stats["on_policy_endpoint_positive_classes"] += endpoint_positive
                # Receding-horizon promotion is anchored to real
                # product-induced GRAPH divergences only. Gold-reset
                # conditional replay after a Qwen-first error may train
                # local graph competence, but must NEVER fake promotion.
                round_status["total"] += 1
                round_status["effective"] += int(supported_contrast)
                round_status["productive"] += int(any(o.positive for o in outcomes))
            else:
                stats["conditional_endpoint_positive_classes"] += endpoint_positive
            # Single graph forward shared across all grouped action alternatives.
            world.policy.train()
            world.policy.graph_encoder.eval()
            graph = build_grounded_graph(anchor.state, None)
            inventory = MechETGroundedInventory(anchor.state)
            nodes, context, prefix = world.policy.encode_directional_observation(
                graph, event_depth=0, remaining_budget=1,
                direction="retrosynthetic_inverse",
            )
            logps = [
                grounded_event_log_probability(
                    world.policy, graph, inventory,
                    tuple(GroundedMove.from_raw(v) for v in candidate.moves),
                    nodes, context, prefix,
                )
                for candidate in candidates
            ]
            reference_nll = (
                teacher_forced_from_tokens(
                    world.policy, graph, inventory, gold_moves,
                    nodes, context, prefix,
                ) if gold_moves is not None else None
            )
            # No verifiable endpoint/local branch contrast AND no map-safe
            # reference label => no supported update, never a fabricated RL step.
            if not supported_contrast and reference_nll is None:
                audit.append({
                    "reaction_id": reaction_id,
                    "status": "no_supported_gradient",
                    "reason": "all-negative/censored and no safe gold map alignment",
                })
                continue
            loss = successor_class_policy_loss(
                logps, [group.member_indices for group in groups],
                adv, reference_nll, reference_weight=args.reference_weight,
            )
            slate_kl = loss.new_zeros(())
            if args.slate_kl_weight:
                with torch.no_grad():
                    old_nodes, old_context, old_prefix = (
                        frozen_policy.encode_directional_observation(
                            graph, event_depth=0, remaining_budget=1,
                            direction="retrosynthetic_inverse",
                        )
                    )
                    original_logps = [
                        grounded_event_log_probability(
                            frozen_policy, graph, inventory,
                            tuple(GroundedMove.from_raw(v) for v in candidate.moves),
                            old_nodes, old_context, old_prefix,
                        )
                        for candidate in candidates
                    ]
                slate_kl = truncated_slate_reference_kl(logps, original_logps)
                loss = loss + args.slate_kl_weight * slate_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            stats["slate_kl_total"] += float(slate_kl.detach().item())
            stats["optimizer_updates"] += 1
            stats["replay_only_updates"] += int(not supported_contrast)
            audit.append({
                "reaction_id": reaction_id, "status": "updated",
                "graph_correctable_first_divergence": train_from_policy,
                "original_divergence": divergence.reason,
                "earliest_divergence": choice.earliest_divergence,
                "anchor_index": choice.decision_index,
                "horizon": horizon.current, "full_product": choice.full_product,
                "groups": group_rows,
                "group_advantages": list(adv),
                "effective_rl_group": supported_contrast,
                "safe_reference_alignment": gold_moves is not None,
                "slate_kl": float(slate_kl.detach().item()),
                "full_budget_branch": full_requested,
                "loss": float(loss.detach().item()),
            })
        except (RuntimeError, ValueError, KeyError, FloatingPointError) as exc:
            if "out of memory" in str(exc).lower():
                abort = True
                raise
            stats["exception_skips"] += 1
            audit.append({
                "reaction_id": reaction_id, "status": "skipped",
                "error": f"{type(exc).__name__}:{exc}",
            })
        finally:
            world.policy.eval()
            stats["processed"] += int(not abort)
            if not abort and (index + 1) % args.curriculum_round == 0:
                horizon, promoted = horizon.advance(
                    total_groups=round_status["total"],
                    effective_groups=round_status["effective"],
                    productive_groups=round_status["productive"],
                )
                stats["curriculum_promotions"] += int(promoted)
                audit.append({
                    "status": "curriculum_round",
                    "after_root": index + 1,
                    "horizon": horizon.current,
                    "promoted": promoted,
                    "round_groups": dict(round_status),
                })
                round_status = {"total": 0, "effective": 0, "productive": 0}
            if not abort and (index + 1) % args.save_every == 0:
                save(index + 1)
    save(args.rl_updates)
    if stats["optimizer_updates"] == 0:
        raise RuntimeError(
            "No gradient updates: inspect audited controller failures and "
            "chemical executability before changing scientific settings"
        )
    head_path = args.output / "graph_earho_policy_head.pt"
    tmp = head_path.with_suffix(".pt.tmp")
    torch.save({
        "policy_head": {
            key: value.detach().cpu().clone()
            for key, value in world.policy.state_dict().items()
            if not key.startswith("graph_encoder.")
        },
        "base_checkpoint_sha256": runtime.graph.sha256,
        "qwen_frozen": True,
        "algorithm": manifest["algorithm"],
        "source_sha256": args.source_sha256,
    }, tmp)
    os.replace(tmp, head_path)
    with (args.output / "training_groups.jsonl").open("w") as stream:
        for item in audit:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    summary = {
        "algorithm": manifest["algorithm"],
        "source_sha256": args.source_sha256,
        "qwen_adapter_sha256": manifest["qwen_adapter_sha256"],
        "graph_pretrained_sha256": runtime.graph.sha256,
        "graph_earho_head_sha256": file_sha256(head_path),
        "stats": stats,
        "promoted_horizon": horizon.current,
        "resumed_root_index": cursor,
        "has_on_policy_endpoint_reward_support": (
            stats["on_policy_endpoint_positive_classes"] > 0
        ),
        "has_graph_successor_class_gradient": stats["effective_rl_groups"] > 0,
        "note": (
            "Neither flag establishes a heldout endpoint gain; conditional "
            "reference replay groups do not count as on-policy EARHO."
        ),
        "warning": (
            "Endpoint RL success is not established by local training group "
            "evidence. Match product-start validation against frozen graph."
        ),
        "locked_test_accessed": False,
    }
    (args.output / "train_report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
