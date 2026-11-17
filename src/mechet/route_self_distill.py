"""TRAIN-only offline self-search target preparation, never on-policy GRPO.

This module does not load a model or call any external teacher. A receipt
manifest is a trusted output of a separate verifier; checking its bindings and
hashes here does not independently replay chemistry or authenticate its author.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile

OBJECTIVE = 'offline_search_selected_native_ce_not_grpo'
PROPOSALS = frozenset({'independent_policy_sample', 'puct_search', 'deduplicated_search', 'rejection_sampled_search'})
HEADER_FIELDS = ('node_id', 'source_actor_sha', 'behavior_policy_version', 'refit_config_sha', 'budget_sha',
                 'tokenizer_sha', 'tool_schema_sha', 'executor_sha', 'stock_contract_sha',
                 'prompt_token_ids', 'conditioning')
HASH_FIELDS = ('source_actor_sha', 'refit_config_sha', 'budget_sha', 'tokenizer_sha',
               'tool_schema_sha', 'executor_sha', 'stock_contract_sha')


def content_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('INVALID_SHA:' + label)
    return value


def _tokens(values, label, eos=None):
    if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
        raise ValueError('INVALID_TOKEN_IDS:' + label)
    if eos is not None and (len(values) < 2 or values[-1] != eos or eos in values[:-1]):
        raise ValueError('NATIVE_ACTION_EOS_REQUIRED')
    return values


def _training(row):
    if row.get('split') != 'training':
        raise ValueError('TRAINING_ONLY')
    if type(row.get('external_teacher_calls')) is not int or row['external_teacher_calls'] != 0:
        raise ValueError('EXTERNAL_TEACHER_FORBIDDEN_STANDARD')


def _receipt(candidate, receipts):
    rid = candidate.get('receipt_id')
    if not isinstance(rid, str) or not rid or rid not in receipts:
        raise ValueError('MISSING_VERIFIED_RECEIPT')
    receipt = receipts[rid]
    if candidate.get('receipt_sha256') != content_sha256(receipt):
        raise ValueError('RECEIPT_SHA_MISMATCH:' + rid)
    if any(receipt.get(k) != v for k, v in candidate.items() if k != 'receipt_sha256'):
        raise ValueError('RECEIPT_BINDING_MISMATCH:' + rid)
    _training(receipt)
    return receipt


def _positive_receipt(receipt):
    return (receipt.get('verified') is True and receipt.get('completed') is True
            and receipt.get('censored') is False and receipt.get('executor_accepted') is True)


def prepare_state(row, receipts, *, expected_actor_sha, training_target_ids, eos_token_id, temperature=1.0):
    """Build q_T on distinct eligible semantics, retaining observed native tokens.

    Duplicate semantic actions use the arithmetic mean of their verified Qs;
    the first eligible recorded tokenization is the representative. Trusted
    completed outcomes INCLUDING excluded rejections determine informativeness.
    Only eligible positive actions enter q_T/CE. One success against a verified
    failed semantic sibling can therefore supply a unit-probability target.
    This is offline supervision, not a behavior or chemistry probability.
    """
    _training(row)
    _sha(expected_actor_sha, 'expected_actor')
    if row.get('source_actor_sha') != expected_actor_sha:
        raise ValueError('SOURCE_ACTOR_MISMATCH')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('INVALID_TEMPERATURE')
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError('INVALID_EOS')
    for field in HASH_FIELDS:
        _sha(row.get(field), field)
    for field in ('node_id', 'behavior_policy_version'):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError('MISSING_CONDITIONING:' + field)
    _tokens(row.get('prompt_token_ids'), 'prompt')
    conditioning = row.get('conditioning')
    if not isinstance(conditioning, dict):
        raise ValueError('MISSING_CONDITIONING')
    if conditioning.get('target_id') not in training_target_ids:
        raise ValueError('TRAIN_TARGET_NOT_ALLOWLISTED')
    if not isinstance(conditioning.get('reaction_occurrence_id'), str) or not conditioning['reaction_occurrence_id']:
        raise ValueError('MISSING_CONDITIONING_OCCURRENCE')
    for field in ('observation_sha', 'history_sha', 'route_context_sha', 'inventory_sha'):
        _sha(conditioning.get(field), field)
    binding = {field: row[field] for field in HEADER_FIELDS}
    state_sha = content_sha256(binding)
    result = dict(binding, kind='self_search', state_key=state_sha, objective=OBJECTIVE,
                  split='training', targets=[], status='skipped', source_selection='refit_search',
                  temperature=float(temperature), duplicate_Q_aggregation='arithmetic_mean',
                  external_teacher_calls=0)
    groups, outcomes, seen, excluded = {}, {}, set(), Counter()
    if not isinstance(row.get('candidates'), list):
        raise ValueError('CANDIDATES_LIST_REQUIRED')
    for candidate in row['candidates']:
        if candidate.get('source_actor_sha') != expected_actor_sha:
            raise ValueError('CANDIDATE_ACTOR_MISMATCH')
        if candidate.get('conditioning_sha') != state_sha:
            raise ValueError('EXACT_CONDITIONING_MISMATCH')
        for field in ('behavior_policy_version', 'refit_config_sha', 'budget_sha'):
            if candidate.get(field) != row[field]:
                raise ValueError('CANDIDATE_BINDING_MISMATCH:' + field)
        if candidate.get('proposal_kind') not in PROPOSALS:
            raise ValueError('UNKNOWN_PROPOSAL_PROVENANCE')
        if candidate.get('generated_role') != 'assistant':
            raise ValueError('ASSISTANT_GENERATED_TOKENS_REQUIRED')
        receipt = _receipt(candidate, receipts)
        rid = candidate['receipt_id']
        if rid in seen:
            raise ValueError('DUPLICATE_RECEIPT:' + rid)
        seen.add(rid)
        trusted = (receipt.get('verified') is True and receipt.get('completed') is True
                   and receipt.get('censored') is False)
        semantic, q = candidate.get('semantic_key'), candidate.get('Q_search')
        if trusted:
            if not isinstance(semantic, str) or not semantic:
                raise ValueError('SEMANTIC_KEY_REQUIRED')
            if type(q) not in (int, float) or not math.isfinite(q):
                raise ValueError('NONFINITE_Q_SEARCH')
            outcomes.setdefault(semantic, []).append(float(q))
        if not _positive_receipt(receipt) or receipt.get('trust_tier') != 'reference_supported_closed':
            excluded[str(receipt.get('trust_tier', 'missing_tier'))] += 1
            continue
        generated = _tokens(candidate.get('generated_token_ids'), 'assistant', eos_token_id)
        group = groups.setdefault(semantic, {'semantic_key': semantic, 'generated_token_ids': list(generated),
                                             'receipt_ids': [], 'proposal_kinds': [], 'values': []})
        group['receipt_ids'].append(rid)
        group['proposal_kinds'].append(candidate['proposal_kind'])
        group['values'].append(float(q))
    result['excluded_candidates'] = dict(excluded)
    outcome_means = [math.fsum(q/len(values) for q in values) for values in outcomes.values()]
    result['informativeness'] = {'trusted_semantic_actions': len(outcomes),
                                 'includes_verified_rejected_outcomes': True,
                                 'Q_min': min(outcome_means, default=None),
                                 'Q_max': max(outcome_means, default=None)}
    if not groups:
        result['reason'] = 'all_ineligible'
        return result
    if len(outcomes) < 2:
        result['reason'] = 'insufficient_distinct_actions'
        return result
    targets = []
    for group in groups.values():
        values = group.pop('values')
        group['Q_search'] = math.fsum(value / len(values) for value in values)
        targets.append(group)
    maximum = max(t['Q_search'] for t in targets)
    if min(outcome_means) == max(outcome_means):
        result['reason'] = 'noninformative_values'
        return result
    weights = [math.exp((t['Q_search'] - maximum) / temperature) for t in targets]
    total = math.fsum(weights)
    for target, weight in zip(targets, weights):
        target['probability'] = weight / total
    result.update(status='prepared', targets=targets)
    return result


def prepare_anchor(row, receipts, *, expected_actor_sha, expected_source_sha, eos_token_id):
    """Accept only explicitly sourced TRAIN replay-verified anchor actions."""
    _training(row)
    kind = row.get('anchor_kind')
    if kind not in {'original_sft', 'actor_state'}:
        raise ValueError('UNKNOWN_ANCHOR_KIND')
    _sha(expected_source_sha, 'anchor_source')
    if row.get('source_artifact_sha') != expected_source_sha:
        raise ValueError('ANCHOR_SOURCE_MISMATCH')
    if kind == 'actor_state' and row.get('source_actor_sha') != expected_actor_sha:
        raise ValueError('ANCHOR_ACTOR_MISMATCH')
    receipt = _receipt(row, receipts)
    if not _positive_receipt(receipt):
        raise ValueError('ANCHOR_NOT_REPLAY_VERIFIED')
    if row.get('generated_role') != 'assistant':
        raise ValueError('ASSISTANT_GENERATED_TOKENS_REQUIRED')
    _tokens(row.get('prompt_token_ids'), 'anchor_prompt')
    _tokens(row.get('generated_token_ids'), 'anchor_assistant', eos_token_id)
    if not isinstance(row.get('source_example_id'), str) or not row['source_example_id']:
        raise ValueError('ANCHOR_EXAMPLE_ID_REQUIRED')
    return {'kind': kind, 'status': 'prepared', 'objective': OBJECTIVE, 'split': 'training',
            'state_key': content_sha256(row), 'source_artifact_sha': expected_source_sha,
            'source_actor_sha': row.get('source_actor_sha'), 'source_example_id': row['source_example_id'],
            'prompt_token_ids': list(row['prompt_token_ids']), 'receipt_ids': [row['receipt_id']],
            'targets': [{'generated_token_ids': list(row['generated_token_ids']), 'probability': 1.0}],
            'external_teacher_calls': 0}


def mix_examples(search_states, anchors, *, sft_weight=0.0, actor_weight=0.0):
    """Assign declared source mass equally per prepared state, not per token."""
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in [sft_weight, actor_weight]):
        raise ValueError('INVALID_MIXTURE_WEIGHT')
    if sft_weight + actor_weight >= 1:
        raise ValueError('SEARCH_MIXTURE_MASS_REQUIRED')
    buckets = {'self_search': list(search_states), 'original_sft': [], 'actor_state': []}
    for anchor in anchors:
        if anchor.get('kind') not in {'original_sft', 'actor_state'}:
            raise ValueError('UNKNOWN_ANCHOR_KIND')
        buckets[anchor['kind']].append(anchor)
    masses = {'self_search': 1-sft_weight-actor_weight, 'original_sft': sft_weight, 'actor_state': actor_weight}
    result, seen = [], set()
    for rows in buckets.values():
        for row in rows:
            if row['state_key'] in seen:
                raise ValueError('DUPLICATE_MIXTURE_STATE')
            seen.add(row['state_key'])
    for kind, rows in buckets.items():
        mass = masses[kind]
        if mass and not rows:
            if kind == 'self_search' and not anchors and sft_weight == actor_weight == 0:
                continue  # Explicit empty coverage report, never a trainable target.
            raise ValueError('MISSING_ANCHOR_OR_SEARCH:' + kind)
        for row in rows:
            if row.get('status') != 'prepared' or row.get('objective') != OBJECTIVE or row.get('split') != 'training':
                raise ValueError('UNPREPARED_MIXTURE_SOURCE')
            if mass:
                result.append(dict(row, mixture_weight=mass/len(rows)))
    return result


def native_assistant_ce(logits, prompt_token_ids, generated_token_ids, weights, eos_token_id):
    """Weighted summed native-action NLL; response includes EOS, no length norm.

    ``logits[b,t]`` predicts token t+1 in prompt+generated tokens. Prompt/tool
    tokens belong only to conditioning. Padding, prompt targets and zero-weight
    examples are not read at all. The caller performs the actual model forward.
    """
    import torch
    batch = len(prompt_token_ids)
    if not (len(generated_token_ids) == len(weights) == batch == logits.shape[0]) or logits.ndim != 3:
        raise ValueError('CE_BATCH_SHAPE_MISMATCH')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('INVALID_CE_WEIGHTS')
    bi, positions, labels, token_weights = [], [], [], []
    for index, (prompt, generated, weight) in enumerate(zip(prompt_token_ids, generated_token_ids, weights)):
        _tokens(prompt, 'prompt'); _tokens(generated, 'assistant', eos_token_id)
        if len(prompt) + len(generated) > logits.shape[1] or max(prompt+generated) >= logits.shape[2]:
            raise ValueError('CE_TOKEN_SHAPE_MISMATCH')
        if weight:
            bi.extend([index]*len(generated))
            positions.extend(range(len(prompt)-1, len(prompt)+len(generated)-1))
            labels.extend(generated)
            token_weights.extend([weight]*len(generated))
    selected = logits[bi, positions]
    if not torch.isfinite(selected).all():
        raise ValueError('NONFINITE_RESPONSE_LOGITS')
    log_probs = torch.log_softmax(selected, dim=-1)
    target_log_probs = log_probs.gather(1, torch.tensor(labels, device=logits.device).unsqueeze(1)).squeeze(1)
    wt = torch.tensor(token_weights, device=logits.device, dtype=log_probs.dtype)
    return -(target_log_probs*wt).sum()/sum(weights)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare_dataset(input_path, receipt_manifest_path, training_targets_path, output_dir, *,
                    expected_actor_sha, eos_token_id, temperature=1.0, anchors_path=None,
                    sft_source_sha=None, actor_source_sha=None, sft_weight=0.0, actor_weight=0.0):
    """Prepare an immutable Ceph data artifact; never start distillation training."""
    paths = [Path(p).resolve() for p in [input_path, receipt_manifest_path, training_targets_path, output_dir]]
    if anchors_path:
        paths.append(Path(anchors_path).resolve())
    if any(not str(p).startswith('/aaa/fionafyang/buddy1/') for p in paths):
        raise ValueError('SHARED_CEPH_PATH_REQUIRED')
    output = paths[3]
    if output.exists():
        raise FileExistsError(output)
    receipt_manifest = json.loads(paths[1].read_text())
    _sha(receipt_manifest.get('verifier_sha'), 'verifier')
    receipts = receipt_manifest['receipts']
    training_ids = set()
    for row in _jsonl(paths[2]):
        tid = row.get('target_id', row.get('task_id', row.get('id')))
        if not isinstance(tid, str) or not tid or tid in training_ids:
            raise ValueError('INVALID_TRAINING_ID_MANIFEST')
        training_ids.add(tid)
    if not training_ids:
        raise ValueError('EMPTY_TRAINING_ALLOWLIST')
    # This preparer keeps normalized state records in memory. A producer for a
    # large corpus should shard this artifact; no silent subset limit is used.
    prepared, skipped, seen, contract = [], [], set(), None
    for row in _jsonl(paths[0]):
        state = prepare_state(row, receipts, expected_actor_sha=expected_actor_sha,
                              training_target_ids=training_ids, eos_token_id=eos_token_id, temperature=temperature)
        signature = tuple(row[k] for k in HASH_FIELDS) + (row['behavior_policy_version'],)
        if contract is None:
            contract = signature
        if signature != contract:
            raise ValueError('MIXED_DATASET_ACTOR_OR_RUNTIME_CONTRACT')
        if state['state_key'] in seen:
            raise ValueError('DUPLICATE_DECISION_STATE')
        seen.add(state['state_key'])
        (prepared if state['status'] == 'prepared' else skipped).append(state)
    anchors = []
    if anchors_path:
        for row in _jsonl(anchors_path):
            source_sha = sft_source_sha if row.get('anchor_kind') == 'original_sft' else actor_source_sha
            anchors.append(prepare_anchor(row, receipts, expected_actor_sha=expected_actor_sha,
                                          expected_source_sha=source_sha, eos_token_id=eos_token_id))
    examples = mix_examples(prepared, anchors, sft_weight=sft_weight, actor_weight=actor_weight)
    input_hashes = {str(p): _file_sha(p) for p in paths if p != output}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.self-distill-', dir=output.parent) as temp:
        staging = Path(temp)/'artifact'; staging.mkdir()
        for filename, rows in [('targets.jsonl', examples), ('skipped.jsonl', skipped)]:
            with (staging/filename).open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        manifest = {'schema_version': 'pr109_self_distill_preparation_v1', 'objective': OBJECTIVE,
                    'source_actor_sha': expected_actor_sha, 'initialized_from_actor_sha': expected_actor_sha,
                    'trained_checkpoint': None, 'trainable': bool(prepared),
                    'preparer_source_sha256': _file_sha(__file__),
                    'runtime_contract': dict(zip(HASH_FIELDS + ('behavior_policy_version',), contract)) if contract else None,
                    'prepared_states': len(prepared), 'skipped_states': len(skipped),
                    'prepared_anchors': len(anchors), 'temperature': temperature, 'eos_token_id': eos_token_id,
                    'sft_weight': sft_weight, 'actor_weight': actor_weight,
                    'sft_anchor_source_sha': sft_source_sha, 'actor_anchor_source_sha': actor_source_sha,
                    'receipt_verifier_sha': receipt_manifest['verifier_sha'], 'input_sha256': input_hashes,
                    'output_sha256': {p.name: _file_sha(p) for p in staging.iterdir()},
                    'external_teacher_calls': 0, 'trust_tier': 'reference_supported_closed_only',
                    'duplicate_Q_aggregation': 'arithmetic_mean',
                    'informativeness': 'all_trusted_completed_noncensored_semantics_including_rejections',
                    'status': 'prepared_not_trained'}
        (staging/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
        # Exclusive directory creation preserves another preparer's ownership.
        output.mkdir()
        for payload in sorted(staging.iterdir(), key=lambda p: p.name == 'manifest.json'):
            # No destination exists in our newly created directory; publish
            # manifest last so interrupted copies cannot look complete.
            payload.rename(output/payload.name)
    return manifest
