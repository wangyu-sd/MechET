#!/usr/bin/env python3
"""Rebuild every frozen executable mech31k trace from a main-product-only root.

CPU-only data/GT verification. Never samples, trains, imports a policy adapter,
or replaces old artifacts. Every reference is independently replayed through
the actual public-root action executor and compared at every electron event.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from mechet.inverse_trace_data import build_inverse_tool_sft_row
from mechet.in_place_grounded_flow import (
    extract_import_fragments, mapped_atom_numbers, retain_mapped_components,
    schedule_imports,
)
from scripts.build_natural_language_event_sft import convert_row, _worker_init
from scripts.run_natural_language_value_search import (
    Action, Node, execute, policy_prompt, product_only_private_state, visible,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def same_chemical_state(left, right):
    """Exact unlabeled graph equality, including stereo and explicit H atoms.

    Symmetric stereorings can have two RDKit canonical SMILES spellings. Use
    reciprocal full-size, chirality-aware graph matching, not stereo removal.
    """
    from rdkit import Chem
    params = Chem.SmilesParserParams()
    params.removeHs = False
    a, b = (Chem.MolFromSmiles(text, params) for text in (left, right))
    if a is None or b is None:
        return False
    if (a.GetNumAtoms(), a.GetNumBonds()) != (b.GetNumAtoms(), b.GetNumBonds()):
        return False
    for mol in (a, b):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return (a.HasSubstructMatch(b, useChirality=True)
            and b.HasSubstructMatch(a, useChirality=True))


def process_pair(pair):
    legacy, forward = map(json.loads, pair)
    rid = str(legacy['source_id'])
    if rid != str(forward['id']):
        raise ValueError(f'SOURCE_ID_MISMATCH: {rid} != {forward["id"]}')
    source_digest = hashlib.sha256(json.dumps(forward, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if source_digest != legacy['metadata']['source_forward_trace_sha256']:
        raise ValueError(f'SOURCE_TRACE_HASH_MISMATCH: {rid}')
    inverse = build_inverse_tool_sft_row(
        forward, product_reference=legacy['metadata']['source_product_reference'],
        observation_mode='action_delta', main_product_only=True,
    )
    if inverse['full_precursor_state'] != legacy['full_precursor_state']:
        raise ValueError(f'REFERENCE_ENDPOINT_CHANGED: {rid}')
    old_steps = legacy['metadata']['trace_plan']['steps']
    steps = inverse['metadata']['trace_plan']['steps']
    if [s['moves'] for s in steps] != [s['moves'] for s in old_steps]:
        raise ValueError(f'REFERENCE_ELECTRON_MOVES_CHANGED: {rid}')
    decisions = convert_row(inverse)
    target = decisions[0]['target_smiles']
    if '.' in target:
        raise ValueError(f'ROOT_NOT_SINGLE_COMPONENT: {rid}')
    initial = product_only_private_state(target)
    node = Node(target=target, state=initial, next_map=max(mapped_atom_numbers(initial)) + 1,
                visited={visible(initial)})
    source_present = set(mapped_atom_numbers(inverse['target_smiles']))
    schedules = schedule_imports(extract_import_fragments(inverse), steps)
    flow_index = 0
    names = []
    for decision in decisions:
        decision['metadata']['input_condition'] = 'main_product_only_v1'
        prompt = policy_prompt(target, node.state, include_inventory=True,
                               actions=node.actions, compact_history=False)
        if prompt != decision['messages'][1]['content']:
            raise ValueError(f'PUBLIC_PREFIX_MISMATCH: {rid} decision={len(names)}')
        function = decision['messages'][2]['tool_calls'][0]['function']
        name, arguments = function['name'], function['arguments']
        child, error = execute(node, Action(name, arguments, '', 0, 0), max_imports=32,
                               reject_target_retained_finish=False)
        if child is None:
            raise ValueError(f'PUBLIC_REFERENCE_EXECUTION_FAILED: {rid} decision={len(names)} {error}')
        if name == 'import_fragments':
            for fragment in schedules[flow_index]:
                source_present.update(mapped_atom_numbers(fragment))
        elif name == 'apply_electron_flow':
            expected = retain_mapped_components(steps[flow_index]['state_after'], source_present)
            if not same_chemical_state(child.state, expected):
                raise ValueError(f'PUBLIC_REFERENCE_SUCCESSOR_MISMATCH: {rid} event={flow_index}')
            flow_index += 1
        node = child
        names.append(name)
    if not node.terminal or not same_chemical_state(node.state, legacy['full_precursor_state']):
        raise ValueError(f'PUBLIC_REFERENCE_ENDPOINT_MISMATCH: {rid}')
    if flow_index != len(steps):
        raise ValueError(f'FLOW_COUNT_CHANGED: {rid}')
    return {
        'inverse': inverse, 'decisions': decisions,
        'audit': {
            'source_id': rid, 'target_smiles': target,
            'input_condition': 'main_product_only_v1', 'initial_components': 1,
            'first_tool': names[0], 'decision_count': len(names),
            'import_decisions': names.count('import_fragments'),
            'flow_successors_match_source': flow_index, 'model_prefix_parity': True,
            'reference_endpoint_exact': True, 'original_moves_preserved': True,
            'gold_in_model_evaluation': False,
        },
    }


def _worker(pair):
    try:
        signal.setitimer(signal.ITIMER_REAL, 180)
        return True, process_pair(pair)
    except Exception as exc:
        source = json.loads(pair[0])
        return False, {'source_id': str(source['source_id']),
                       'error': f'{type(exc).__name__}:{exc}'}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _init_worker():
    _worker_init()
    import torch
    torch.set_num_threads(1)


def paired_lines(legacy_file, forward_file):
    with legacy_file.open() as left, forward_file.open() as right:
        for a, b in zip_longest(left, right):
            if a is None or b is None:
                raise ValueError('SOURCE_ROW_COUNT_MISMATCH')
            yield a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--source-dir', default='data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    source = args.workspace / args.source_dir
    code_paths = (
        ROOT / 'src/mechet/inverse_trace_data.py',
        ROOT / 'src/mechet/in_place_grounded_flow.py',
        ROOT / 'scripts/build_natural_language_event_sft.py', Path(__file__),
        ROOT / 'scripts/run_natural_language_value_search.py',
        ROOT / 'src/mechet/forward_expert.py',
        ROOT / 'src/mechet/natural_language_electron_flow.py',
    )
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_paths}
    manifest = json.loads((source / 'manifest.json').read_text())
    status = json.loads((source / 'ARTIFACT_STATUS.json').read_text())
    if status.get('training_allowed') is not True:
        raise ValueError('Source trace artifact is not validated')
    if args.output_dir.exists():
        raise ValueError('Refusing to overwrite an existing output directory')
    args.output_dir.mkdir(parents=True)
    for name in ('inverse', 'state_sft', 'audits'):
        (args.output_dir / name).mkdir()
    started = time.monotonic()
    reports = {}
    for split in ('valid', 'test', 'train'):
        legacy = source / f'{split}.jsonl'
        source_report = json.loads((source / f'{split}.report.json').read_text())
        forward = args.workspace / source_report['input']
        if sha(legacy) != manifest['splits'][split]['sha256'] or sha(forward) != source_report['input_sha256']:
            raise ValueError(f'SOURCE_FILE_HASH_MISMATCH: {split}')
        counter = Counter()
        ids = set()
        files = {name: args.output_dir / name / f'{split}.jsonl'
                 for name in ('inverse', 'state_sft', 'audits')}
        unresolved = args.output_dir / 'audits' / f'{split}.unresolved.jsonl'
        with (files['inverse'].open('w') as inverse_out,
              files['state_sft'].open('w') as sft_out,
              files['audits'].open('w') as audit_out,
              unresolved.open('w') as bad,
              mp.get_context('fork').Pool(max(1, args.workers), initializer=_init_worker) as pool):
            for ok, value in pool.imap(_worker, paired_lines(legacy, forward), chunksize=2):
                rid = value['audit']['source_id'] if ok else value['source_id']
                if rid in ids:
                    raise ValueError(f'DUPLICATE_SOURCE_ID: {split} {rid}')
                ids.add(rid)
                if not ok:
                    bad.write(json.dumps(value) + '\n')
                    counter['unresolved'] += 1
                else:
                    inverse_out.write(json.dumps(value['inverse'], ensure_ascii=False) + '\n')
                    for row in value['decisions']:
                        sft_out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    audit_out.write(json.dumps(value['audit']) + '\n')
                    counter['reference_pass'] += 1
                    counter['decision_rows'] += value['audit']['decision_count']
                    counter['flow_events'] += value['audit']['flow_successors_match_source']
                    counter['first_' + value['audit']['first_tool']] += 1
                if len(ids) % 100 == 0:
                    for writer in (inverse_out, sft_out, audit_out, bad):
                        writer.flush()
                    print(json.dumps({'stage': 'main-product-full-reference-build',
                        'split': split, 'processed': len(ids), 'expected': manifest['splits'][split]['rows'],
                        'passed': counter['reference_pass'], 'unresolved': counter['unresolved'],
                        'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
        if len(ids) != manifest['splits'][split]['rows']:
            raise ValueError(f'DENOMINATOR_CHANGED: {split}')
        reports[split] = {
            'reactions': len(ids), **dict(counter), 'source_sha256': sha(legacy),
            'forward_sha256': sha(forward),
            'outputs': {k: {'path': str(p), 'sha256': sha(p)} for k, p in files.items()},
            'unresolved_file': str(unresolved), 'unresolved_sha256': sha(unresolved),
        }
        (args.output_dir / f'{split}.report.json').write_text(json.dumps(reports[split], indent=2) + '\n')
        print(json.dumps({'split_complete': split, **reports[split]}), flush=True)
    complete = all(r['reference_pass'] == r['reactions'] and not r.get('unresolved', 0)
                   for r in reports.values())
    result = {
        'artifact_type': 'mech31k_main_product_only_first_use_v1', 'complete': complete,
        'input_condition': 'main_product_only_v1', 'reports': reports,
        'reference_replay_verified': complete, 'training_allowed': False,
        'training_disabled_reason': 'User requested data/GT/evaluation only; no tokenizer audit or training authorized.',
        'denominator_scope': 'all frozen current-compiler executable trace-view reactions, not all31199 raw reactions',
        'source_manifest_sha256': sha(source / 'manifest.json'),
        'code_sha256': code_hashes,
        'code_unchanged_during_build': all(sha(p) == code_hashes[str(p.relative_to(ROOT))]
                                         for p in code_paths),
        'elapsed_seconds': time.monotonic() - started,
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output_dir / 'ARTIFACT_STATUS.json').write_text(json.dumps({
        'status': 'reference_validated' if complete else 'unresolved_retained',
        'training_allowed': False, 'reference_replay_verified': complete,
    }, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
