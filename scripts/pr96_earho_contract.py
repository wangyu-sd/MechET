"""Small, testable invariants for the PR96 actor-only EARHO adaptation.

Chemical classes here pool sampled first actions, not an exhaustive probability
mass over the infinite language policy. The loss is a clipped sampled-action
surrogate; the reference regularizer is evaluated on those behavior samples.
"""
from pathlib import Path

PARENT_SHA = 'cd5f2f9dbdc883674be2f87781452366a3fb2c01c7c930482e0b5e4b3dab49ca'


def validate_hardware(names, expected):
    if expected not in ('H20', 'A100') or len(names) != 8 or not all(expected in name for name in names):
        raise ValueError('EARHO_HARDWARE_MISMATCH')


def validate(cfg):
    if cfg.get('algorithm') != 'pr96_earho_v1':
        raise ValueError('EARHO_ALGORITHM_MISMATCH')
    if cfg.get('parent_sha256') != PARENT_SHA:
        raise ValueError('EARHO_PARENT_MISMATCH')
    if cfg.get('train_reactions') != 10137:
        raise ValueError('EARHO_TRAIN_DENOMINATOR_MISMATCH')
    if cfg.get('world_size') != 8 or cfg.get('candidates') != 2:
        raise ValueError('EARHO_DISTRIBUTION_OR_CANDIDATE_CONTRACT_CHANGED')
    for key in ('train_file', 'history_file'):
        if Path(cfg[key]).name != 'train.jsonl' or 'DIAGNOSTIC' in cfg[key].upper():
            raise ValueError('EARHO_TRAIN_ONLY_SOURCE_REQUIRED')


def rounds(ids, size):
    if len(set(ids)) != len(ids):
        raise ValueError('EARHO_DUPLICATE_TRAIN_IDS')
    if size <= 0:
        raise ValueError('EARHO_INVALID_ROUND_SIZE')
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


def first_divergence(episode, calls, reference_states, equivalent, reference_terminals=None):
    if episode['structural_exact']:
        return None
    for call in calls:
        depth = call['depth']
        terminal_wrong = (reference_terminals is not None and depth+1 < len(reference_states)
                          and (call.get('predicted_name') == 'finish_trace') != reference_terminals[depth+1])
        if (depth + 1 >= len(reference_states) or call['error'] or terminal_wrong
                or not equivalent(call['state_after'], reference_states[depth + 1])):
            return min(depth, len(reference_states) - 2)
    # An unfinished but reference-aligned prefix is corrected at its next step.
    return min(len(calls), len(reference_states) - 2)


def assign_advantages(rows, equivalent):
    classes = []
    for row in rows:
        matched = None
        if row.get('first_state') is not None:
            matched = next((c for c in classes
                            if c[0].get('first_state') is not None
                            and c[0]['first_terminal'] == row['first_terminal']
                            and equivalent(c[0]['first_state'], row['first_state'])), None)
        if matched is None:
            classes.append([row])
        else:
            matched.append(row)
    values = [sum(r['reward'] for r in c) / len(c) for c in classes]
    positive = any(r['positive'] for r in rows)
    advantages = ([v - (sum(values) - v)/(len(values)-1) for v in values]
                  if positive and len(values) > 1 else [0.] * len(values))
    for index, (members, advantage) in enumerate(zip(classes, advantages)):
        for row in members:
            row.update(successor_class=index, advantage=advantage,
                       policy_weight=1 / (len(classes) * len(members)))
    return {'classes': len(classes), 'effective': any(a != 0. for a in advantages),
            'productive': positive}


def promote(horizon, summaries, maximum, minimum_effective, threshold):
    local = [v for v in summaries if not v['rehearsal']]
    effective = sum(v['effective'] for v in local)
    rate = sum(v['productive'] for v in local)/max(len(local), 1)
    return min(maximum, horizon + int(effective >= minimum_effective and rate >= threshold))


def token_logps(model, prompt, response):
    import torch
    if not prompt or not response:
        raise ValueError('EARHO_EMPTY_POLICY_SEQUENCE')
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + response], device=device)
    positions = torch.arange(len(prompt)-1, len(prompt)+len(response)-1, device=device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   logits_to_keep=positions, use_cache=False)
    selected = torch.tensor(response, device=device)
    return result.logits[0].float().log_softmax(-1).gather(-1, selected[:, None]).squeeze(-1)


def policy_loss(new, behavior, reference, advantage, clip, beta):
    import torch
    ratio = (new - behavior).exp()
    pg = -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).mean()
    delta = reference - new
    regularizer = (delta.exp() - delta - 1).mean()
    return pg + beta * regularizer, pg, regularizer


def synchronize_gradients(model, denominator):
    """Exact global mean despite unequal local groups and optional PG records.

    Accumulate sums locally, then reduce each trainable tensor in fixed order.
    No collector or rank-dependent autograd path enters a DDP collective.
    Missing local gradients contribute zero. Optimizer steps remain collective.
    """
    import torch
    import torch.distributed as dist
    if denominator <= 0:
        raise ValueError('EARHO_ZERO_GLOBAL_BATCH')
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if dist.is_initialized():
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(denominator)


def continue_branches(branches, horizon, generate, execute, journal):
    """First actions have already executed; spend at most H-1 further actions."""
    if horizon < 1:
        raise ValueError('EARHO_INVALID_HORIZON')
    for depth in range(1, horizon):
        active = [b for b in branches if not b['error'] and not b['node'].terminal]
        if not active:
            break
        outputs = generate([b['node'] for b in active])
        if len(outputs) != len(active):
            raise ValueError('EARHO_BACKEND_DROPPED_BRANCH')
        for branch, output in zip(active, outputs):
            previous = branch['node']
            child, error = execute(previous, output)
            journal(branch, previous, child, error, output, depth)
            branch['error'] = error
            if child is not None:
                branch['node'] = child
    for branch in branches:
        branch['censored'] = not branch['error'] and not branch['node'].terminal


def verify_checkpoint(path, contract_sha):
    import hashlib
    import json
    path = Path(path)
    manifest = json.loads((path / 'complete.json').read_text())
    if manifest['contract_sha256'] != contract_sha:
        raise ValueError('EARHO_RESUME_CONTRACT_MISMATCH')
    required = {'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'trainer_state.json'}
    required.update(f'rng.rank{rank:02d}.pt' for rank in range(8))
    if not required.issubset(manifest['files']):
        raise ValueError('EARHO_RESUME_COMPONENTS_MISSING')
    for name, expected in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('EARHO_RESUME_INVALID_FILE')
        digest = hashlib.sha256()
        with (path/name).open('rb') as reader:
            for block in iter(lambda: reader.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError('EARHO_RESUME_HASH_MISMATCH:' + name)
    state = json.loads((path/'trainer_state.json').read_text())
    keys = ('next_round', 'horizon', 'optimizer_step', 'visited_train_roots',
            'train_denominator', 'completed_train_ids_sha256')
    if any(key not in state or manifest.get(key) != state[key] for key in keys):
        raise ValueError('EARHO_RESUME_CURSOR_MISMATCH')
    if (any(type(state[k]) is not int for k in keys[:-1])
            or not 1 <= state['next_round'] <= 80
            or not 2 <= state['horizon'] <= 12 or state['optimizer_step'] < 0
            or state['train_denominator'] != 10137
            or state['visited_train_roots'] != min(state['next_round']*128, 10137)
            or path.name != f"round-{state['next_round']:05d}"):
        raise ValueError('EARHO_RESUME_CURSOR_OUTSIDE_FROZEN_SCHEDULE')
    return manifest


def checkpoint_staging(root, next_round):
    import tempfile
    parent = Path(root)/'checkpoints'
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'round-{next_round:05d}.building-', dir=parent))


def discover_checkpoint(root, contract_sha):
    import re
    root = Path(root)
    candidates = sorted(p for p in (root/'checkpoints').glob('round-*')
                        if p.is_dir() and re.fullmatch(r'round-[0-9]{5}', p.name))
    if not candidates:
        if (root/'latest_checkpoint.json').exists():
            raise ValueError('EARHO_RESUME_PUBLISHED_CHECKPOINT_MISSING')
        return None
    selected = candidates[-1]
    verify_checkpoint(selected, contract_sha)
    return selected
