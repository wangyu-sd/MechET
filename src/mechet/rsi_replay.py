"""Authentic TRAIN-only Stage-II tool decisions, replayed before retention.

The reader parses only selected source-ID records. Private endpoint fields in
the source row are never rendered; the original two-message prefix is checked
against the executor's actual state and accepted history at every action.
"""
from collections import defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re

from .rsi_contract import fingerprint, stable_seed, write_json


def verify_reaction(rows, tokenizer, source_id, train_ids, *, mapped_target=None):
    from mechet.assistant_masking import render_chat, render_qwen_sft_tool_prefix
    from mechet.in_place_grounded_flow import mapped_atom_numbers
    from scripts.rsi_runtime import reverse_prompt
    from scripts.run_natural_language_value_search import Node, Action, execute, private_product_state, visible, TOOLS
    if source_id not in train_ids or not source_id.startswith('flower_mech_proof_train_'):
        raise ValueError('retention requires authenticated TRAIN source ID')
    if not rows:
        raise ValueError('empty replay source reaction')
    state = private_product_state(mapped_target or rows[0]['target_smiles'])
    if visible(state) != rows[0]['target_smiles']:
        raise ValueError('authenticated mapped source target changed public product')
    node = Node(target=visible(state), state=state, next_map=max(mapped_atom_numbers(state))+1,
                visited={visible(state)})
    result = []
    for index, row in enumerate(rows):
        m = row['metadata']
        if row['source_id'] != source_id or m.get('reaction_id') != source_id or m.get('decision_index') != index:
            raise ValueError('TRAIN stable source-ID / decision-index crosswalk failed')
        if m.get('history_contract') != 'executor_compact_accepted_actions_v1' or m.get('executor_replayed') is not True:
            raise ValueError('unverified source history contract')
        if m.get('history_model_visible_gold_horizon') is not False or m.get('history_contains_failed_actions') is not False:
            raise ValueError('future / failed action leakage in replay source')
        if row['target_smiles'] != rows[0]['target_smiles'] or row['tools'] != TOOLS or node.terminal:
            raise ValueError('reaction target / tools / terminal discontinuity')
        messages = row['messages']
        if [x['role'] for x in messages] != ['system', 'user', 'assistant', 'tool']:
            raise ValueError('unexpected assistant/tool exchange')
        prompt = render_qwen_sft_tool_prefix(tokenizer, messages[:2], tools=row['tools'])
        if prompt != reverse_prompt(node, tokenizer):
            raise ValueError('original Stage-II prompt differs from replayed state/history')
        calls = messages[2].get('tool_calls', [])
        if len(calls) != 1:
            raise ValueError('replay requires one complete tool call')
        call = calls[0]['function']; name, arguments = call['name'], call['arguments']
        full = render_chat(tokenizer, messages[:3], tools=row['tools'])
        if not full.startswith(prompt):
            raise ValueError('assistant prefix/mask mismatch')
        ids = tokenizer.encode(full[len(prompt):], add_special_tokens=False)
        eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if eos not in ids:
            raise ValueError('supervised action missing assistant EOS')
        ids = ids[:ids.index(eos)+1]
        before = node.state
        node, error = execute(node, Action(name, arguments, '', 0., len(ids)),
                              max_imports=32, reject_target_retained_finish=True)
        if node is None or error:
            raise ValueError('authentic source action failed replay: '+str(error))
        observed = json.loads(messages[3]['content'])
        state_key = 'derived_precursor' if name == 'finish_trace' else 'current_state'
        if messages[3].get('name') != name or observed.get('ok') is not True or observed.get(state_key) != visible(node.state):
            raise ValueError(f'stored tool result differs from actual replayed state: {source_id} decision={index} '
                             f'tool={name} stored={observed.get(state_key)!r} replayed={visible(node.state)!r}')
        result.append({'id': row['id'], 'source_id': source_id, 'decision_index': index,
            'prompt_ids': tokenizer.encode(prompt, add_special_tokens=False), 'completion_ids': ids,
            'prompt_sha': fingerprint(prompt), 'tool': name, 'executor_verified': True,
            'state_before': before, 'state_after': node.state,
            'source_row_sha': fingerprint(row), 'source_split': 'train'})
    if not node.terminal:
        raise ValueError('incomplete authentic reaction replay')
    return result


def select_actions(rows, maximum=4):
    selected = []; used = set()
    # Include tool diversity, then a deterministic extra event where available.
    for tool in ('import_fragments', 'apply_electron_flow', 'finish_trace'):
        candidates = [r for r in rows if r['tool'] == tool]
        if candidates and len(selected) < maximum:
            r = min(candidates, key=lambda x: stable_seed(17, x['id']))
            selected.append(r); used.add(r['id'])
    for r in sorted(rows, key=lambda x: stable_seed(17, x['id'])):
        if r['id'] not in used and len(selected) < maximum:
            selected.append(r); used.add(r['id'])
    return selected


def replay_source_ids(proof_path, included, count=512):
    """Stable bottom-k independent TRAIN IDs without decoding 257K proofs."""
    included = set(included); heap = []; train_ids = set()
    pattern = re.compile(rb'"id"\s*:\s*"([^"]+)"')
    with Path(proof_path).open('rb') as f:
        for line in f:
            match = pattern.search(line[:500])
            if not match:
                raise ValueError('source ID absent from frozen proof prefix')
            identifier = match[1].decode()
            if not identifier.startswith('flower_mech_proof_train_') or identifier in train_ids:
                raise ValueError('duplicate/non-TRAIN proof source')
            train_ids.add(identifier)
            if identifier in included:
                continue
            key = -stable_seed(17, 'independent-real-replay', identifier)
            item = (key, identifier)
            if len(heap) < count-len(included): heapq.heappush(heap, item)
            elif item > heap[0]: heapq.heapreplace(heap, item)
    if not included <= train_ids or len(train_ids) != 257167:
        raise ValueError('frozen strict TRAIN universe/crosswalk changed')
    return sorted(included | {x[1] for x in heap})


def mapped_replay_targets(path, identifiers, expected_sha, output):
    """Read the actual Stage-I→II parent, not a remapped proof approximation.

    That parent can include explicit product H nodes absent in the proof view,
    and its original atom ordering fixes the executor's hidden Kekule choice.
    No GT future action is used to choose a mapped representation.
    """
    output = Path(output); path = Path(path); selected = set(identifiers)
    identity = {'path': str(path), 'source_sha': expected_sha, 'ids': sorted(selected),
                'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    if output.exists():
        cached = json.loads(output.read_text())
        if cached['identity'] != identity or fingerprint(cached['targets']) != cached['targets_sha']:
            raise ValueError('authenticated original mapped target cache changed')
        return cached['targets']
    digest = hashlib.sha256(); targets = {}
    pattern = re.compile(rb'"source_id"\s*:\s*"([^"]+)"')
    with path.open('rb') as f:
        for line in f:
            digest.update(line); match = pattern.search(line[:500])
            if not match:raise ValueError('original trace source-ID prefix missing')
            identifier = match[1].decode()
            if identifier not in selected:continue
            row = json.loads(line)
            if row['source_id'] != identifier or identifier in targets:
                raise ValueError('original mapped trace source crosswalk failed')
            targets[identifier] = row['target_smiles']
    if digest.hexdigest() != expected_sha or set(targets) != selected:
        raise ValueError('original mapped trace source SHA/coverage failed')
    write_json(output, {'identity': identity, 'targets_sha': fingerprint(targets), 'targets': targets})
    return targets


def build_replay_pool(path, identifiers, tokenizer, expected_sha, output, references):
    output = Path(output)
    contract = fingerprint([str(path), sorted(identifiers), expected_sha, 'real-history-execute-v1'])
    if (output/'manifest.json').exists():
        previous = json.loads((output/'manifest.json').read_text())
        if previous['contract'] != contract:
            raise ValueError('replay resume source contract changed')
        rows = json.loads((output/'actions.json').read_text())
        if fingerprint(rows) != previous['actions_sha']:
            raise ValueError('cached replay pool corrupt')
        return rows, previous
    selected = set(identifiers); groups = defaultdict(list); digest = hashlib.sha256()
    pattern = re.compile(rb'"source_id"\s*:\s*"([^"]+)"')
    cache = output/'selected_source_rows.json'
    cache_identity = output/'selected_source_rows.identity.json'
    stat = Path(path).stat()
    if cache.exists() and cache_identity.exists():
        old = json.loads(cache_identity.read_text())
        cached = json.loads(cache.read_text())
        if old != {'contract': contract, 'rows_sha': fingerprint(cached), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
            raise ValueError('replay selected-source cache changed; requires source re-audit')
        groups = cached
    else:
      with Path(path).open('rb') as f:
        for i, line in enumerate(f, 1):
            digest.update(line); match = pattern.search(line[:500])
            if not match:
                raise ValueError('source-ID missing from Stage-II prefix')
            identifier = match[1].decode()
            if identifier in selected:
                row = json.loads(line)
                if row['source_id'] != identifier:
                    raise ValueError('Stage-II source crosswalk failed')
                groups[identifier].append(row)
            if i % 200000 == 0:
                print({'phase': 'real-replay-stream', 'source_rows_scanned': i, 'selected_sources_seen': len(groups)}, flush=True)
      if digest.hexdigest() != expected_sha or set(groups) != selected:
          raise ValueError('Stage-II source SHA / selected source coverage failed')
      write_json(cache, dict(groups))
      write_json(cache_identity, {'contract': contract, 'rows_sha': fingerprint(dict(groups)),
                                'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    pool = []; verified_count = 0
    from concurrent.futures import ThreadPoolExecutor
    def verify(identifier):
        return verify_reaction(groups[identifier], tokenizer, identifier, selected,
                               mapped_target=references[identifier]['replay_mapped_target'])
    with ThreadPoolExecutor(max_workers=8) as workers:
        for i, verified in enumerate(workers.map(verify, sorted(selected)), 1):
            verified_count += len(verified)
            for action in select_actions(verified):
                action['reference_class'] = references[action['source_id']]['kind']
                pool.append(action)
            if i % 32 == 0:
                print({'phase': 'real-replay-execute', 'verified_reactions': i, 'retained_actions': len(pool)}, flush=True)
    manifest = {'contract': contract, 'source_sha256': expected_sha, 'source_path': str(path),
        'independent_train_reactions': len(selected), 'retained_actions': len(pool),
        'verified_full_trajectory_actions': verified_count, 'actions_sha': fingerprint(pool),
        'test_used': False, 'assistant_only_eos_included': True,
        'mapped_source_target_sha': fingerprint({k:references[k]['replay_mapped_target'] for k in sorted(selected)}),
        'prompt_parity': 'byte_exact_runtime_state_plus_compact_accepted_history'}
    write_json(output/'actions.json', pool); write_json(output/'manifest.json', manifest)
    return pool, manifest
