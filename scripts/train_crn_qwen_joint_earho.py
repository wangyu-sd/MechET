#!/usr/bin/env python3
"""Joint Qwen IMPORT/controller and CRN FLOW conditional endpoint optimization.

Model-owned frontiers, fixed pre-return proposal support, IID replacement
draws, matched K1 follower, binary terminal endpoint returns, verified replay
and frozen-original conditional KL. No value critic, no GT proposal insertion.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.crn_graph_bridge import CALL_BEGIN, file_sha256, graph_event_coordinates, tool_prefix, parse_qwen_import
from mechet.endpoints import reference_structural_precursor, split_precursor_endpoints, structural_exact
from mechet.in_place_grounded_flow import mapped_atom_numbers
from mechet.joint_endpoint_policy import (
    JointProposal, freeze_joint_support, qwen_continuation_log_probability,
    joint_action_log_probability, conditional_score_gradients, import_generation_budget, delegated_route_tokens,
)
from mechet.graph_endpoint_rl import grounded_event_log_probability, slate_log_probabilities, iid_slate_indices
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS, convert_row
from scripts.eval_natural_language_event_local import prediction_call
from scripts.pr83_resume import AuditJournal, prepare_training_resume
from scripts.run_crn_qwen_hybrid import HybridRuntime, prepare_args
from scripts.run_natural_language_value_search import (
    Node, Action, execute, visible, product_only_private_state, policy_prompt, read_selected,
)
from scripts.train_crn_graph_earho import (
    training_parser, graph_head_parameters, replay_train_trace,
    align_reference_moves, executed_successor_key,
)

ALGORITHM = 'joint_import_flow_endpoint_v1'
SOURCE_SHA = 'edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75'
ADAPTER_SHA = 'fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd'


def select_joint_frontier(episode, reference=None):
    """Any first executed divergence is eligible, including controller errors."""
    if not episode:
        return None, None
    if reference is None:
        return episode[0][0], 0
    for index, (node, attempt) in enumerate(episode):
        if index >= len(reference):
            return node, index
        expected = reference[index]
        if (not attempt['accepted'] or attempt['successor'] != expected.successor_key
                or attempt['terminal'] != expected.terminal):
            return node, index
    return None, None


def endpoint_return(node, row, origin):
    if node is None or not node.terminal:
        return 0.
    endpoint = reference_structural_precursor(dict(row))
    if endpoint:
        return float(structural_exact(split_precursor_endpoints(node.state, origin).structural, endpoint))
    return float(visible(node.state) == visible(str(row.get('full_precursor_state') or row['expected_precursor'])))


def prompt_for(runtime, node):
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=policy_prompt(
        node.target, node.state, include_inventory=True, actions=node.actions,
        compact_history=runtime.args.compact_history))]
    return render_qwen_sft_tool_prefix(runtime.tokenizer, messages, tools=TOOLS)


def input_tokens(runtime, node):
    return tuple(runtime.tokenizer.encode(prompt_for(runtime, node), add_special_tokens=False))


def root_node(row):
    product = visible(str(row['target_smiles']))
    state = product_only_private_state(product)
    return Node(product, state, max(mapped_atom_numbers(state))+1, visited={product})


def follow(runtime, start, args, decisions):
    """Exactly one greedy episode. Failures stop, never resample or insert GT."""
    node = start
    episode, records = [], []
    for _ in range(max(0, decisions)):
        if node.terminal or len(node.actions) >= args.max_decisions:
            break
        runtime.raw_generations.clear()
        runtime.graph_observations.clear()
        proposals = runtime.proposals(node, candidates=1,
            max_new_tokens=args.max_import_tokens, compact_history=args.compact_history)
        child, error = (execute(node, proposals[0], max_imports=args.max_imports,
            reject_target_retained_finish=True) if proposals else
            (None, runtime.last_proposal_error or 'NO_ACTION'))
        attempt = dict(accepted=child is not None, successor=visible(child.state) if child else None,
                       terminal=bool(child and child.terminal))
        episode.append((node, attempt))
        records.append(dict(state_before=node.state, next_map=node.next_map,
            action=asdict(proposals[0]) if proposals else None, error=error,
            state_after=child.state if child else None,
            model_generations=list(runtime.raw_generations),
            graph_observations=list(runtime.graph_observations), **attempt))
        if child is None:
            return None, episode, records
        node = child
    return node, episode, records


def collect_joint_support(runtime, node, args, rng):
    """Freeze model-only raw token paths before rewards, including bad imports."""
    import torch
    prompt_ids = input_tokens(runtime, node)
    proposals = []
    # Canonical name prefixes define explicit delegated route events. Forced
    # prefixes are scored under Qwen; they are not assigned probability one.
    flow_route = delegated_route_tokens(runtime.tokenizer, 'apply_electron_flow')
    finish_route = delegated_route_tokens(runtime.tokenizer, 'finish_trace')
    import_route = tuple(runtime.tokenizer.encode(tool_prefix('import_fragments'), add_special_tokens=False))
    generation_budget = import_generation_budget(len(prompt_ids), len(import_route),
        max_context=args.max_context, max_tokens=args.max_import_tokens)
    runtime.raw_generations.clear()
    for index in range(args.import_proposals):
        ids = torch.tensor([prompt_ids+import_route], dtype=torch.long, device=runtime.device)
        with torch.inference_mode():
            out = runtime.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=index > 0, **({'temperature':1., 'top_p':1., 'top_k':0} if index > 0 else {}),
                max_new_tokens=generation_budget, pad_token_id=runtime.tokenizer.pad_token_id,
                eos_token_id=runtime.tokenizer.eos_token_id)
        suffix = tuple(out[0, ids.shape[1]:].tolist())
        raw = runtime.tokenizer.decode(import_route+suffix, skip_special_tokens=False)
        arguments, error = {}, ''
        try:
            arguments = parse_qwen_import(raw, runtime.tokenizer, import_route+suffix,
                                          require_eos=args.require_import_eos)
        except ValueError as exc:
            error = str(exc)
        proposals.append(JointProposal('import_fragments', arguments or {}, import_route+suffix, raw, error=error))
    depth, budget = graph_event_coordinates(node.actions, args.graph_initial_budget)
    candidates = runtime.graph.candidates(node.state, limit=args.flow_proposals,
        event_depth=depth, remaining_budget=budget, beam_size=args.graph_beam,
        source_width=args.graph_source_width, sink_width=args.graph_sink_width,
        move_width=args.graph_move_width)
    candidates += runtime.graph.exploration_candidates(node.state, count=args.exploration_proposals,
        rng=rng, event_depth=depth, remaining_budget=budget)
    for event in candidates:
        proposals.append(JointProposal('apply_electron_flow', event.arguments, flow_route,
            runtime.tokenizer.decode(flow_route), event.moves))
    proposals.append(JointProposal('finish_trace', {}, finish_route, runtime.tokenizer.decode(finish_route)))
    return freeze_joint_support(proposals)


def score_joint_support(runtime, node, support, *, original=False):
    import torch
    from crn_jepa.grounded_actions import GroundedMove, MechETGroundedInventory
    prompt = input_tokens(runtime, node)
    policy = runtime.original_graph if original else runtime.graph.model.policy
    has_flow = any(item.name == 'apply_electron_flow' for item in support)
    graph = runtime.graph._build_graph(node.state, None) if has_flow else None
    inventory = MechETGroundedInventory(node.state) if has_flow else None
    if has_flow:
        depth, budget = graph_event_coordinates(node.actions, runtime.args.graph_initial_budget)
        nodes, context, prefix = policy.encode_directional_observation(graph,
            event_depth=depth, remaining_budget=budget, direction='retrosynthetic_inverse')
    if original:
        runtime.model.set_adapter('original')
    try:
        result = []
        for proposal in support:
            qwen = qwen_continuation_log_probability(runtime.model, prompt, proposal.token_ids)
            flow = (grounded_event_log_probability(policy, graph, inventory,
                tuple(GroundedMove.from_raw(move) for move in proposal.moves), nodes, context, prefix)
                if proposal.name == 'apply_electron_flow' else None)
            result.append(joint_action_log_probability(proposal, qwen, flow))
        return result
    finally:
        if original:
            runtime.model.set_adapter('policy')
            trainable_qwen(runtime.model)


def trainable_qwen(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('lora_' in name and '.policy.' in name)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('no trainable Qwen policy LoRA parameters')
    model.eval()  # dropout off for generation, scoring and gradient recomputation
    return parameters


def execute_proposal(node, proposal, args):
    if proposal.error:
        return None, proposal.error
    return execute(node, Action(proposal.name, proposal.arguments, proposal.raw, 0.,
        max(1, len(proposal.token_ids))), max_imports=args.max_imports,
        reject_target_retained_finish=True)


def import_participation(node, flow_map_sets=None):
    """Audit actual imported map participation; purpose never gates chemistry."""
    flow_map_sets = flow_map_sets or []
    records = []
    for index, action in enumerate(node.actions):
        if action['name'] != 'import_fragments':
            continue
        before = mapped_atom_numbers(action['state_before'])
        after = (node.actions[index+1]['state_before'] if index+1<len(node.actions) else node.state)
        added = set(mapped_atom_numbers(after))-set(before)
        used = any(added & maps for maps in flow_map_sets)
        if not flow_map_sets:
            from mechet.natural_language_electron_flow import compile_event_arguments
            for later in node.actions[index+1:]:
                if later['name'] == 'apply_electron_flow':
                    for move in compile_event_arguments(later['state_before'], later['arguments']):
                        touched = set(move['source']['atoms']) | set(move['sink']['atoms'])
                        used = used or bool(added & touched)
        records.append(dict(action_index=index, imported_maps=sorted(added),
            declared_purposes=[str(f.get('purpose', '')) for f in action['arguments']['fragments']],
            used_in_flow=bool(used)))
    return records


def verified_replay(runtime, anchor, trace, index):
    """Private label only at a genuinely identical state, never into the slate."""
    if trace is None or index is None or index >= len(trace['reference']):
        return None
    gold = trace['nodes'][index]
    if visible(gold.state) != visible(anchor.state):
        return None
    record = trace['nodes'][index+1].actions[-1]
    name, arguments = record['name'], record['arguments']
    moves = ()
    if name == 'apply_electron_flow':
        if index not in trace['graph_moves']:
            return None
        aligned = align_reference_moves(gold.state, anchor.state, trace['graph_moves'][index])
        if aligned is None:
            return None
        moves = tuple(move.to_raw() for move in aligned)
        arguments = runtime.graph._render(anchor.state, moves)
        tokens = delegated_route_tokens(runtime.tokenizer, name)
        raw = runtime.tokenizer.decode(tokens, skip_special_tokens=False)
    elif name == 'finish_trace':
        tokens = delegated_route_tokens(runtime.tokenizer, name)
        raw = runtime.tokenizer.decode(tokens, skip_special_tokens=False)
    else:
        raw = tool_prefix(name)+json.dumps(arguments, ensure_ascii=False)+'}\n</tool_call>'+runtime.tokenizer.eos_token
        tokens = tuple(runtime.tokenizer.encode(raw, add_special_tokens=False))
    proposal = JointProposal(name, dict(arguments), tokens, raw, moves)
    child, error = execute_proposal(anchor, proposal, runtime.args)
    if child is None or visible(child.state) != visible(trace['nodes'][index+1].state):
        return None
    return proposal


def joint_training_parser():
    p = training_parser()
    p.description = __doc__
    p.set_defaults(sample_reactions=257167, rl_updates=257167, max_import_tokens=384,
                   save_every=32)
    p.add_argument('--qwen-learning-rate', type=float, default=1e-6)
    p.add_argument('--import-proposals', type=int, default=4)
    p.add_argument('--flow-proposals', type=int, default=4)
    return p


def main():
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from mechet.graph_earho import AdaptiveHorizon
    from scripts.pr83_resume import save_joint_training_state, export_joint_policy, validate_joint_weights, joint_source_fingerprints
    args = prepare_args(joint_training_parser().parse_args())
    args.require_import_eos = True
    if (args.sample_reactions != 257167 or args.rl_updates != 257167
        or args.source_sha256 != SOURCE_SHA or file_sha256(args.data) != SOURCE_SHA
        or args.data.name != 'train.jsonl' or args.compact_history
        or args.graph_rl_head or args.tool_route != 'greedy_name'
        or args.rollout_beam_width != 1 or args.graph_flow_topk != 1
        or args.dtype != 'bfloat16' or not args.no_4bit
        or args.actor_reward != 'endpoint' or args.policy_objective != 'conditional_endpoint'
        or file_sha256(args.policy_adapter/'adapter_model.safetensors') != ADAPTER_SHA):
        raise ValueError('joint formal protocol requires pinned complete strict train and State-SFT parents')
    if not (2 <= args.slate <= 32 and 1 <= args.import_proposals <= 8
        and 1 <= args.flow_proposals <= 8 and 0 <= args.exploration_proposals <= 8
        and 0 < args.qwen_learning_rate <= 1e-4 and 0 < args.learning_rate <= 1e-4
        and args.save_every > 0 and 0 < args.reference_weight <= 1
        and 0 <= args.slate_kl_weight <= 1 and 0 < args.full_episode_fraction <= 1
        and args.temperature > 0 and 2 <= args.continuation_horizon <= 40):
        raise ValueError('invalid joint optimization contract')
    rows = read_selected(args.data, args.sample_reactions, args.seed)
    if len(rows) != args.sample_reactions or len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('strict train denominator or ID uniqueness mismatch')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    sources = joint_source_fingerprints(ROOT, args.crn_root)
    settings = {key: value for key, value in vars(args).items()
                if key not in {'data', 'output', 'resume', 'policy_adapter', 'crn_root',
                               'graph_checkpoint', 'graph_rl_head'}}
    manifest = dict(algorithm=ALGORITHM, selected_ids=[str(r['id']) for r in rows],
        train_source_sha256=SOURCE_SHA, train_source_path=str(args.data.resolve()),
        qwen_adapter_sha256=ADAPTER_SHA, crn_pretrained_sha256=args.graph_sha256,
        settings=settings, **sources)
    saved = prepare_training_resume(args.output, manifest=manifest, resume=args.resume)
    args.capture_generations = True
    runtime = HybridRuntime(args, 0)
    runtime.model.load_adapter(args.policy_adapter, adapter_name='original', is_trainable=False)
    runtime.model.set_adapter('policy')
    runtime.model.peft_config['policy'].inference_mode = False
    qwen_parameters = trainable_qwen(runtime.model)
    runtime.original_graph = copy.deepcopy(runtime.graph.model.policy).eval()
    runtime.original_graph.graph_encoder = runtime.graph.model.policy.graph_encoder
    for p in runtime.original_graph.parameters():
        p.requires_grad_(False)
    graph_parameters = graph_head_parameters(runtime.graph.model)
    parameters = qwen_parameters+graph_parameters
    optimizer = torch.optim.AdamW([dict(params=qwen_parameters, lr=args.qwen_learning_rate),
                                  dict(params=graph_parameters, lr=args.learning_rate)])
    horizon = AdaptiveHorizon(current=2, maximum=args.continuation_horizon,
        min_effective_groups=args.min_effective_groups,
        min_productive_fraction=args.productive_pass_rate,
        full_episode_fraction=args.full_episode_fraction)
    stats = dict(processed=0, optimizer_updates=0, endpoint_contrast_updates=0,
        replay_only_updates=0, exact_autonomous=0, reference_failures=0,
        endpoint_positive_draws=0, import_frontiers=0, exceptions=0)
    cursor, journals = 0, None
    if saved:
        state = torch.load(saved, map_location='cpu', weights_only=False)
        if state['manifest'] != manifest:
            raise ValueError('joint resume lineage changed')
        validate_joint_weights(get_peft_model_state_dict(runtime.model, adapter_name='policy'),
            {k:v for k,v in runtime.graph.model.policy.state_dict().items()
             if not k.startswith('graph_encoder.')}, state)
        set_peft_model_state_dict(runtime.model, state['qwen_policy'], adapter_name='policy')
        policy_state = runtime.graph.model.policy.state_dict()
        policy_state.update(state['graph_policy'])
        runtime.graph.model.policy.load_state_dict(policy_state, strict=True)
        optimizer.load_state_dict(state['optimizer'])
        rng.setstate(state['rng_state'])
        torch.set_rng_state(state['torch_rng_state'])
        torch.cuda.set_rng_state_all(state['cuda_rng_state'])
        stats, cursor, journals = state['stats'], state['next_index'], state['journals']
        if not 0 <= cursor <= len(rows) or stats['processed'] != cursor:
            raise ValueError('joint resume processed-root cursor mismatch')
        horizon = AdaptiveHorizon(**state['horizon'])
    audit = AuditJournal(args.output/'training_groups.jsonl',
        restored_state=journals['groups'] if journals else None)
    episodes = AuditJournal(args.output/'autonomous_train_episodes.jsonl',
        restored_state=journals['episodes'] if journals else None)
    round_status = dict(total=0, effective=0, productive=0, endpoint=0)
    if saved:
        round_status = state['round_status']

    def save(index):
        save_joint_training_state(args.output/'latest_train_state.pt', dict(
            manifest=manifest, next_index=index, optimizer=optimizer.state_dict(),
            qwen_policy=get_peft_model_state_dict(runtime.model, adapter_name='policy'),
            graph_policy={k:v for k,v in runtime.graph.model.policy.state_dict().items()
                          if not k.startswith('graph_encoder.')},
            rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all(), stats=stats,
            journals=dict(groups=audit.checkpoint_state(), episodes=episodes.checkpoint_state()),
            horizon=asdict(horizon), round_status=round_status))
    if not saved:
        save(0)
    started = time.perf_counter()
    try:
        for index in range(cursor, len(rows)):
            row, full_requested = rows[index], rng.random()<horizon.full_episode_fraction
            updates_before, aborted = stats['optimizer_updates'], False
            try:
                initial = root_node(row)
                end, episode, raw = follow(runtime, initial, args, args.max_decisions)
                base_return = endpoint_return(end, row, initial.state)
                stats['exact_autonomous'] += int(base_return)
                episodes.append(dict(id=row['id'], return_=base_return, attempts=raw,
                    import_audit=import_participation(end) if end else []))
                try:
                    trace = replay_train_trace(row)
                except (ValueError, KeyError) as exc:
                    trace = None
                    stats['reference_failures'] += 1
                    audit.append(dict(id=row['id'], status='reference_unavailable', error=str(exc)))
                anchor, decision = select_joint_frontier(episode, trace['reference'] if trace else None)
                if anchor is None:
                    audit.append(dict(id=row['id'], status='no_divergence'))
                    continue
                stats['import_frontiers'] += int(raw[decision].get('action', {}).get('name')=='import_fragments') if raw[decision]['action'] else 0
                support = collect_joint_support(runtime, anchor, args, rng)
                proposal_generations = list(runtime.raw_generations)
                with torch.no_grad():
                    scores = torch.stack(score_joint_support(runtime, anchor, support))
                    original = torch.stack(score_joint_support(runtime, anchor, support, original=True))
                draws = iid_slate_indices(slate_log_probabilities(scores,
                    temperature=args.temperature), args.slate, rng)
                remaining = min(args.max_decisions-len(anchor.actions)-1,
                    args.max_decisions if full_requested else horizon.current-1)
                by_index, by_successor, branches = {}, {}, []
                for support_index in dict.fromkeys(draws):
                    proposal = support[support_index]
                    child, error = execute_proposal(anchor, proposal, args)
                    key = (executed_successor_key(child.state), child.next_map,
                        json.dumps(child.imported, sort_keys=True), tuple(sorted(child.visited)),
                        json.dumps(child.actions, sort_keys=True)) if child else None
                    if key is not None and key in by_successor:
                        terminal, history, result = by_successor[key]
                    else:
                        terminal, _, history = (follow(runtime, child, args, remaining)
                            if child is not None else (None, [], []))
                        result = endpoint_return(terminal, row, initial.state)
                        if key is not None:
                            by_successor[key] = (terminal, history, result)
                    by_index[support_index] = result
                    branches.append(dict(support_index=support_index, accepted=child is not None,
                        error=error, successor_state=child.state if child else None,
                        endpoint_return=result, terminal_state=terminal.state if terminal else None,
                        attempts=history, import_audit=import_participation(terminal) if terminal else []))
                returns = [by_index[d] for d in draws]
                coefficients, objective = conditional_score_gradients(scores, draws, returns,
                    temperature=args.temperature, actor_enabled=full_requested,
                    kl_weight=args.slate_kl_weight, original_scores=original)
                contrast = full_requested and len(set(returns))>1
                replay = verified_replay(runtime, anchor, trace, decision)
                optimizer.zero_grad(set_to_none=True)
                for proposal, coefficient in zip(support, coefficients, strict=True):
                    if coefficient.item():
                        probability, = score_joint_support(runtime, anchor, (proposal,))
                        (probability*coefficient).backward()
                replay_loss = 0.
                if replay is not None:
                    probability, = score_joint_support(runtime, anchor, (replay,))
                    replay_term = -args.reference_weight*probability
                    replay_loss = float(replay_term.detach())
                    replay_term.backward()
                has_gradient = any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in parameters)
                grad_norm = 0.
                if has_gradient:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
                    optimizer.step()
                    stats['optimizer_updates'] += 1
                    stats['endpoint_contrast_updates'] += int(contrast)
                    stats['replay_only_updates'] += int(not contrast and replay is not None)
                stats['endpoint_positive_draws'] += int(sum(returns)) if full_requested else 0
                audit.append(dict(id=row['id'], status='updated' if has_gradient else 'no_supported_gradient',
                    anchor_index=decision, anchor_state=anchor.state, full_budget=full_requested,
                    horizon=horizon.current, input_token_ids=list(input_tokens(runtime, anchor)),
                    input_text=prompt_for(runtime, anchor), support=[asdict(p) for p in support],
                    proposal_generations=proposal_generations, complete_scores=scores.cpu().tolist(),
                    original_scores=original.cpu().tolist(), draws=list(draws), returns=returns,
                    branches=branches, objective=objective, replay_loss=replay_loss,
                    replay_action=asdict(replay) if replay else None, grad_norm=grad_norm,
                    endpoint_contrast=contrast))
                round_status['total'] += 1
                round_status['effective'] += int(contrast)
                round_status['productive'] += int(full_requested and any(returns))
                round_status['endpoint'] += int(full_requested and any(returns))
            except (ValueError, KeyError) as exc:
                stats['exceptions'] += 1
                audit.append(dict(id=row['id'], status='failed_root', error=f'{type(exc).__name__}:{exc}'))
            except BaseException:
                aborted = True
                raise
            finally:
                if not aborted:
                    stats['processed'] += 1
                    if (index+1)%args.curriculum_round==0:
                        horizon, promoted = horizon.advance(total_groups=round_status['total'],
                            effective_groups=round_status['effective'], productive_groups=round_status['productive'],
                            endpoint_groups=round_status['endpoint'])
                        audit.append(dict(status='curriculum', after_root=index+1, promoted=promoted, horizon=horizon.current))
                        round_status = dict(total=0, effective=0, productive=0, endpoint=0)
                    if (index+1)%args.save_every==0 or (updates_before==0 and stats['optimizer_updates']==1):
                        save(index+1)
                    print(json.dumps(dict(stage=ALGORITHM, total=len(rows), horizon=horizon.current,
                        elapsed_seconds=round(time.perf_counter()-started, 2), **stats)), flush=True)
        save(len(rows))
        if not stats['optimizer_updates']:
            raise RuntimeError('no joint updates; see preserved complete root audits')
        contract = export_joint_policy(runtime, args.output, manifest, stats)
        (args.output/'train_report.json').write_text(json.dumps(dict(
            manifest=manifest, stats=stats, pair=contract, locked_test_accessed=False,
            heldout_gain_measured=False), indent=2)+'\n')
    finally:
        audit.close()
        episodes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
