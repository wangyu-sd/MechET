#!/usr/bin/env python3
"""Build the entire named strict FlowER universe from ORIGINAL forward rows.

Bounded CPU multiprocessing, no chemistry/length/dedup filtering, fail closed
on any replay error or missing reaction. Next states are labels/tool results,
never policy inputs. Existing inverse artifacts are read-only eligibility lists.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))
from mechet.forward_qwen_events import build_transition_row, forward_prompt
from mechet.mech_graph import parse_flower_line
from mechet.proof_program import _canonical_mapped

EXPECTED = {'train': 257167, 'valid': 2890, 'test': 28967}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert_group(task: tuple[str, list[str]]) -> dict[str, Any]:
    reaction_id, lines = task
    rows, errors = [], []
    edges, incoming = defaultdict(list), set()
    states = {}
    converted = []
    terminal_rows = 0
    for index, line in enumerate(lines):
        try:
            parsed = parse_flower_line(line)
            if parsed is None:
                raise ValueError('invalid source transition')
            before, after, _ = parsed
            key = _canonical_mapped(before)
            next_key = _canonical_mapped(after)
            row = build_transition_row(line, reaction_id=reaction_id,
                                       decision_index=index)
            function = row['messages'][2]['tool_calls'][0]['function']
            states.setdefault(key, before)
            states.setdefault(next_key, after)
            if function['name'] == 'finish_forward':
                terminal_rows += 1
            else:
                edges[key].append((next_key, index, function))
                incoming.add(next_key)
            converted.append((index, key, before, row))
            rows.append(row)
        except Exception as exc:
            errors.append({'reaction_id': reaction_id, 'decision_index': index,
                           'error': str(exc), 'source_line': line.strip()})
    # Export order is not a topological order. Build one deterministic recorded
    # root-to-state path before rendering histories; never concatenate branches.
    histories = {key: ([], []) for key in states if key not in incoming}
    queue = deque(histories)
    while queue:
        key = queue.popleft()
        history, indices = histories[key]
        for next_key, index, function in edges[key]:
            if next_key not in histories:
                histories[next_key] = (history + [function], indices + [index])
                queue.append(next_key)
    for index, key, before, row in converted:
        if key not in histories:
            errors.append({'reaction_id': reaction_id, 'decision_index': index,
                           'error': 'UNROOTED_RECORDED_FORWARD_STATE',
                           'source_line': lines[index].strip()})
            continue
        history, indices = histories[key]
        row['messages'][1]['content'] = forward_prompt(before, history)
        row['metadata']['history_decision_indices'] = indices
        row['metadata']['history_contract'] = 'recorded_forward_graph_path_actions_v2'
    if not terminal_rows:
        errors.append({'reaction_id': reaction_id, 'error': 'MISSING_RECORDED_TERMINAL'})
    return {'rows': rows, 'errors': errors, 'terminal_rows': terminal_rows}


def source_groups(path: Path):
    """Group by reaction ID, not by adjacent runs in the original export.

    FlowER may interleave alternative paths of one reaction with other IDs.
    Index byte ranges rather than retaining the multi-GB training file in RAM.
    Both first-seen reaction order and every within-reaction row are preserved.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    offset, current = 0, None
    heartbeat = time.monotonic()
    with path.open('rb') as handle:
        for number, encoded in enumerate(handle, 1):
            line = encoded.decode('utf-8')
            parsed = parse_flower_line(line)
            if parsed is None:
                raise ValueError(f'malformed source row {path}:{number}')
            trajectory_id = parsed[2]
            end = offset + len(encoded)
            blocks = ranges.setdefault(trajectory_id, [])
            if trajectory_id == current:
                blocks[-1] = (blocks[-1][0], end)
            else:
                blocks.append((offset, end))
            current = trajectory_id
            offset = end
            if time.monotonic() - heartbeat >= 45:
                print({'phase': 'forward-source-index', 'source': str(path),
                       'rows': number, 'reactions': len(ranges), 'bytes': offset}, flush=True)
                heartbeat = time.monotonic()
        for trajectory_id, blocks in ranges.items():
            group = []
            for start, end in blocks:
                handle.seek(start)
                group.extend(handle.read(end - start).decode('utf-8').splitlines(keepends=True))
            yield trajectory_id, group


def eligibility(proof_file: Path, expected: int) -> dict[str, str]:
    ids = {}
    with proof_file.open() as handle:
        for line in handle:
            row = json.loads(line)
            trajectory_id = str(row['metadata']['trajectory_id'])
            if trajectory_id in ids:
                raise ValueError(f'duplicate eligible reaction {trajectory_id}')
            ids[trajectory_id] = row['id']
    if len(ids) != expected:
        raise ValueError(f'eligibility count {len(ids)} != {expected}')
    return ids


def build_split(raw: Path, proof_file: Path, output: Path, expected: int, workers: int) -> dict[str, Any]:
    ids = eligibility(proof_file, expected)
    print({'phase': 'forward-build', 'source': str(raw), 'eligible_reactions': len(ids),
           'cpu_workers': workers}, flush=True)
    pending, covered, excluded = deque(), set(), []
    rows = failures = terminals = completed = source_rows = 0
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never append to a partial build: reconstruct the artifact deterministically.
    error_file = output.with_suffix('.errors.jsonl')
    with output.open('w') as target, error_file.open('w') as errors:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            def consume():
                nonlocal rows, failures, terminals, completed
                trajectory_id, future = pending.popleft()
                result = future.result()
                covered.add(trajectory_id)
                for row in result['rows']:
                    target.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                for error in result['errors']:
                    errors.write(json.dumps(error, ensure_ascii=False) + '\n')
                rows += len(result['rows'])
                failures += len(result['errors'])
                terminals += result['terminal_rows']
                completed += 1
                if completed % 1000 == 0:
                    print({'phase': 'forward-build', 'split': output.stem, 'reactions': completed,
                           'decisions': rows, 'errors': failures, 'seconds': round(time.monotonic()-started)}, flush=True)
            for trajectory_id, lines in source_groups(raw):
                if trajectory_id not in ids:
                    if trajectory_id not in {'RC', 'PC', 'PM', 'RS'}:
                        raise ValueError(f'unexpected source-only trajectory {trajectory_id}; refusing new filtering')
                    excluded.append({'trajectory_id': trajectory_id, 'transitions': len(lines)})
                    continue
                pending.append((trajectory_id, pool.submit(convert_group, (ids[trajectory_id], lines))))
                source_rows += len(lines)
                if len(pending) >= workers * 2:
                    consume()
            while pending:
                consume()
    missing = sorted(set(ids) - covered)
    return {'file': str(output), 'rows': rows, 'source_rows': source_rows,
            'source_rows_preserved': rows == source_rows,
            'reactions': len(covered), 'terminal_rows': terminals,
            'errors': failures, 'missing_reaction_ids': missing, 'excluded_source_groups': excluded,
            'source_path': str(raw), 'source_sha256': sha256(raw),
            'eligibility_path': str(proof_file), 'eligibility_sha256': sha256(proof_file),
            'output_sha256': sha256(output), 'errors_file': str(error_file)}


def render_training_config(template: dict[str, Any], splits: dict[str, Any]) -> dict[str, Any]:
    if template.get('initial_adapter_path'):
        raise ValueError('forward training must not use initial_adapter_path from the reverse branch')
    cfg = copy.deepcopy(template)
    cfg['contract']['expected_train_rows'] = splits['train']['rows']
    cfg['contract']['expected_validation_rows'] = splits['valid']['rows']
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--proof-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config-template', type=Path, required=True)
    parser.add_argument('--rendered-config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / 'ARTIFACT_STATUS.json'
    if (args.output_dir / 'manifest.json').exists():
        raise ValueError('refusing to overwrite an already sealed forward artifact')
    status_path.write_text(json.dumps({'training_allowed': False, 'status': 'building'}))
    source_status = json.loads((args.proof_dir / 'ARTIFACT_STATUS.json').read_text())
    if source_status.get('rows') != EXPECTED or source_status.get('training_allowed') is not True:
        raise ValueError('unapproved eligibility universe')
    workers = max(1, min(args.workers, len(os.sched_getaffinity(0))))
    splits = {}
    for split, name in [('valid', 'val'), ('test', 'test'), ('train', 'train')]:
        splits[split] = build_split(args.raw_dir / (name + '.txt'), args.proof_dir / (split + '.jsonl'),
                                    args.output_dir / (split + '.jsonl'), EXPECTED[split], workers)
        report = {'artifact_type': 'forward_natural_language_event_v1', 'reaction_denominator': EXPECTED,
                  'official_reaction_denominator': {'train': 257171, 'valid': 2890, 'test': 28971},
                  'source': 'original forward FlowER state transitions', 'splits': splits,
                  'final_product_model_visible': False, 'raw_explicit_hydrogens_retained': True,
                  'length_filtering': False, 'overlap_filtering': False, 'dedup_filtering': False,
                  'training_allowed': False}
        (args.output_dir / 'build_progress.json').write_text(json.dumps(report, indent=2))
        if (splits[split]['errors'] or splits[split]['missing_reaction_ids']
                or not splits[split]['source_rows_preserved'] or splits[split]['reactions'] != EXPECTED[split]):
            raise ValueError(f'full forward split failed: inspect {args.output_dir}/build_progress.json')
    report.update(training_allowed=True, status='validated_complete')
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2))
    status_path.write_text(json.dumps({'artifact_id': 'flower_forward_qwen_event_v1',
                                      'training_allowed': True, 'status': 'validated',
                                      'reaction_denominator': EXPECTED}, indent=2))
    cfg = render_training_config(yaml.safe_load(args.config_template.read_text()), splits)
    args.rendered_config.parent.mkdir(parents=True, exist_ok=True)
    args.rendered_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps({'phase': 'forward-build-complete', 'splits': splits}), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
