#!/usr/bin/env python3
"""Full input-ledger replay: verified recovery memory and reference-supported SFT.

Input JSONL rows contain task_id and snapshot (RouteEpisode.snapshot()). All rows
remain in accounting. No sampling limit or gold electron actions are supported.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from mechet.external_route_rl import HiddenRouteReward
from mechet.qwen3_continuation import shared_path, verify_files
from mechet.route_memory_curation import curate_episode
from mechet.route_training_contract import atomic_json

DATASET_SHA = 'f11e6909e466c003ce112018db4d07cffa4033dce2984a0a8ff1cae931e8aa79'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--executor-version', required=True)
    args = p.parse_args()
    dataset, source, output = map(shared_path, (args.dataset, args.episodes, args.output))
    if sha(dataset/'manifest.json') != DATASET_SHA:
        raise ValueError('APPROVED_EXTERNAL_TRAIN_MANIFEST_REQUIRED')
    manifest = json.loads((dataset/'manifest.json').read_text())
    needed = ('training.jsonl','stock.txt','references.sqlite','training_protected_identities.json')
    verify_files({str(dataset/name):manifest['artifacts'][name] for name in needed})
    tasks = {r['task_id']:r['target'] for r in map(json.loads,(dataset/'training.jsonl').read_text().splitlines())}
    if len(tasks) != 23730:
        raise ValueError('FULL_EXTERNAL_TRAIN_DENOMINATOR_REQUIRED')
    stock = set((dataset/'stock.txt').read_text().splitlines())
    protected = set(json.loads((dataset/'training_protected_identities.json').read_text()))
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(stock_version=manifest['artifacts']['stock.txt'], executor_version=args.executor_version)
    entries, seen, counts = {}, set(), Counter()
    source_before = source.stat()
    supervision_path, rejected_path = output/'supervision.jsonl', output/'excluded.jsonl'
    with HiddenRouteReward(dataset/'references.sqlite', stock, split='training') as oracle, \
            source.open() as reader, supervision_path.open('w') as supervised, rejected_path.open('w') as excluded:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            counts['input_episodes'] += 1
            try:
                row = json.loads(line)
                result = curate_episode(row['snapshot'], task_id=row['task_id'], public_tasks=tasks,
                    protected=protected, output_dir=output/'receipts', oracle=oracle, binding=binding, stock=stock)
                uid = result['source_snapshot_sha256']
                if uid in seen:
                    counts['duplicate_episodes'] += 1
                    continue
                seen.add(uid)
                counts['replayed_episodes'] += 1
                counts['reference_verified_episodes'] += result['reference_verified']
                for item in result['memory']['entries']:
                    entries[item['id']] = item
                for item in result['supervision']:
                    supervised.write(json.dumps(item, ensure_ascii=False)+'\n')
                    counts['supervised_decisions'] += 1
            except (ValueError, KeyError, TypeError) as error:
                # Explicit exclusion ledger; never convert failed verification
                # into a negative chemistry label or an eligible training row.
                counts['excluded_episodes'] += 1
                excluded.write(json.dumps(dict(line=line_number, reason=str(error)))+'\n')
    source_after = source.stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
        raise ValueError('EPISODE_SOURCE_CHANGED_DURING_REPLAY')
    memory = dict(schema='route_experience_memory_v2', entries=[entries[k] for k in sorted(entries)])
    atomic_json(output/'memory.json', memory)
    atomic_json(output/'manifest.json', dict(schema='route_memory_dataset_v1', status='replayed',
        source_sha256=sha(source), dataset_manifest_sha256=DATASET_SHA, split='training',
        source_episode_denominator=counts['input_episodes'], counts=dict(counts), memory_entries=len(entries),
        executor_version=args.executor_version, artifacts={n:sha(output/n) for n in
            ('supervision.jsonl','excluded.jsonl','memory.json')},
        memory_adaptation_ready=bool(counts['supervised_decisions']),
        chemical_validation='hidden_TRAIN_reference_support_only',
        ratio='not_selected_until_actual_eligible_counts'))
    print(json.dumps(dict(counts=counts, memory_entries=len(entries), output=str(output))))


if __name__ == '__main__':
    main()
