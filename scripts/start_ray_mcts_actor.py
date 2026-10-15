#!/usr/bin/env python3
"""Own the persistent shared MCTS service for one eight-GPU continuation."""
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

from mechet.shared_electron_search import read_record, fingerprint
from scripts.infer_dynamic_electron_trees import dump


def storage_paths(manifest, output):
    previous = Path(manifest['mcts_shared_root'])
    if manifest.get('tree_store_backend') == 'ray_memory_grouped_WAL_v1':
        # Reopen the same append-only journal. It is a persistent shared store,
        # not a legacy directory containing per-node JSON files.
        header = read_record(previous / 'contract.json')
        return previous, Path(header['legacy_root']) if header['legacy_root'] else None
    return Path(output) / 'ray_tree_store', previous


def load_resume_manifests(source):
    rows = []
    keys = ('world', 'source_sha', 'adapter_sha', 'revision', 'seed', 'program_goal',
            'history_contract', 'max_model_len', 'shared_cache_namespace', 'mcts_shared_root',
            'source_protocol', 'evaluation_unit', 'stopping_contract')
    for rank in range(8):
        row = json.loads((Path(source) / 'tree' / f'rank{rank:02d}' / 'manifest.json').read_text())
        if row.get('rank') != rank or row.get('world') != 8 or row.get('shard_rows') != 390:
            raise ValueError('full3120 shard contract mismatch')
        if rows and any(row.get(k) != rows[0].get(k) for k in keys):
            raise ValueError('mixed resume shard contract')
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--address-file', type=Path, required=True)
    args = p.parse_args()
    rows = load_resume_manifests(args.source_output); manifest = rows[0]
    from scripts.infer_single_reaction_mcts import SOURCE_SHA, ADAPTER_SHA
    from scripts.ray_mcts_store import single_planning_contract, start_actor
    if manifest['source_sha'] != SOURCE_SHA or manifest['adapter_sha'] != ADAPTER_SHA:
        raise ValueError('source/actor resume SHA mismatch')
    contract = single_planning_contract(manifest['shared_cache_namespace'],
        manifest['max_model_len'], manifest['seed'], manifest['program_goal'])
    first = next((args.source_output / 'tree/rank00/mcts').glob('*.json'))
    if read_record(first)['planning_contract'] != contract:
        raise ValueError('legacy planner source contract changed')
    root, legacy = storage_paths(manifest, args.output)
    import ray
    if ray.__version__ != '2.44.1': raise ValueError('pinned Ray2.44.1 required')
    namespace = 'mechet-' + fingerprint(str(args.output.resolve()))[:12]
    ctx = ray.init(num_cpus=min(32, len(os.sched_getaffinity(0))), include_dashboard=False,
                   object_store_memory=256 * 1024 * 1024, namespace=namespace,
                   _node_ip_address='127.0.0.1', log_to_driver=True)
    actor = start_actor(root, contract, legacy, namespace=namespace)
    ray.get(actor.call.remote('identity', (), {}))  # constructor/durable recovery must succeed
    dump(args.address_file, {'address': ctx.address_info['gcs_address'], 'namespace': namespace,
                            'actor': 'single-reaction-tree', 'store_root': str(root)})
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    try:
        while not stopping.is_set():
            print({'phase': 'Ray-MCTS-owner', **ray.get(actor.call.remote('metrics', (), {}))}, flush=True)
            stopping.wait(45)
    finally:
        try: ray.get(actor.call.remote('close', (), {}))
        finally: ray.shutdown()
    return 0


if __name__ == '__main__': raise SystemExit(main())
