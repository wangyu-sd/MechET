"""Full3120 saved-native recovery audit; no model or active rollout leases."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from scripts.mcts_resume import (case_evidence, resume_lineage,
                                _restore_completed_native_contexts)
from scripts.ray_mcts_store import MemoryMCTSCore
from mechet.shared_electron_search import (durable_json, fingerprint, read_record,
                                          restore_tree, tree_snapshot, context_record)

REPO = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
SOURCE = REPO / 'outputs/eval/mech_uspto31k_full_mcts_ray_20261015_02'
PREVIOUS = REPO / 'outputs/eval/mech_uspto31k_mcts_ray_migration_20261015_03'
OUTPUT = REPO / 'outputs/analysis/mcts_native_resume_audit_20261016_02'
DATA = REPO / 'data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl'


def main():
    started = time.monotonic()
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == '819215c5d40720afd998aaf77b7b938045d8171a5487b26bf929897f65a957c4'
    rows = [json.loads(s) for s in DATA.open() if s.strip()]
    assert len(rows) == len({r['id'] for r in rows}) == 3120
    OUTPUT.mkdir(parents=True, exist_ok=False)
    store_source = PREVIOUS / 'ray_tree_store'
    header = read_record(store_source / 'contract.json')
    shadow = OUTPUT / 'shadow_readthrough_store'; shadow.mkdir()
    shutil.copyfile(store_source / 'transactions.jsonl', shadow / 'transactions.jsonl')
    core = MemoryMCTSCore(shadow, header['contract'], legacy_root=header['legacy_root'])
    totals = {'reports': 0, 'checkpoints': 0, 'completed': 0, 'pending': 0,
              'terminal_programs': 0, 'reconstructed_native_terminals': 0,
              'serialized_contexts_verified': 0}
    repairs = []; seen = set()
    try:
        def audit_rank(rank):
            totals = {'reports': 0, 'checkpoints': 0, 'completed': 0, 'pending': 0,
                      'terminal_programs': 0, 'reconstructed_native_terminals': 0,
                      'serialized_contexts_verified': 0}
            repairs = []; seen = set()
            source = SOURCE / 'tree' / f'rank{rank:02d}'
            previous = PREVIOUS / 'tree' / f'rank{rank:02d}'
            lineage = resume_lineage(source)
            for row in rows[rank::8]:
                identifier = row['id']; name = fingerprint(identifier) + '.json'
                assert identifier not in seen; seen.add(identifier)
                report, checkpoint, journal = case_evidence(lineage, identifier)
                if report:
                    actual = json.loads(report.read_text())
                    expected = json.loads((previous / 'evaluations' / name).read_text())
                    for key in ('resumed_from', 'tree_artifact'):
                        actual.pop(key, None); expected.pop(key, None)
                    assert actual == expected, identifier
                    totals['reports'] += 1
                    continue
                saved = read_record(checkpoint)
                expected = read_record(previous / 'mcts' / name)
                for key in ('case_id', 'root', 'planning_contract', 'seen_states',
                            'blocked', 'completed_programs', 'program_goal', 'pending_rollouts'):
                    assert saved[key] == expected[key], (identifier, key)
                for key in ('completed_rollouts', 'positive_backups', 'censored_rollouts'):
                    assert saved['metrics'][key] == expected['metrics'][key], (identifier, key)
                c = journal.restore(stock=set())
                c.completed_programs = set(saved['completed_programs'])
                c.mcts_store = core; c.applied = set(); c.mcts_root = saved['root']
                c.seen_states = set(saved['seen_states'])
                before_work = dict(c.work)
                missing = c.completed_programs.difference(c.contexts)
                if missing:
                    states = core.snapshots(saved['seen_states'])
                    for value in states.values():
                        # RouteNetwork has no planner hydrate method. The own
                        # predecessor is already part of this native journal.
                        if not value['state']['solved']:
                            assert value['state']['active'] in c.contexts, identifier
                    _restore_completed_native_contexts(c, states,
                        checkpoint_history=(root / 'mcts' / name for root, _ in lineage))
                    assert {k:v for k,v in c.work.items() if k != 'unique_chemical_successors'} == {
                        k:v for k,v in before_work.items() if k != 'unique_chemical_successors'}, identifier
                    repairs.append({'id': identifier, 'rank': rank,
                                    'terminal_contexts': sorted(missing)})
                    totals['reconstructed_native_terminals'] += len(missing)
                assert c.completed_programs.issubset(c.contexts), identifier
                assert all(c.contexts[k].node.terminal for k in c.completed_programs), identifier
                for tree in c.trees.values():
                    # Exercise the exact journal snapshot -> JSON -> reload
                    # boundary that the first migration failed to verify.
                    restored = restore_tree(json.loads(json.dumps(tree_snapshot(tree))))
                    assert set(restored.contexts) == set(tree.contexts), identifier
                    for key in c.completed_programs.intersection(tree.contexts):
                        assert context_record(restored.contexts[key]) == context_record(tree.contexts[key]), identifier
                    totals['serialized_contexts_verified'] += len(restored.contexts)
                totals['checkpoints'] += 1
                totals['completed'] += saved['metrics']['completed_rollouts']
                totals['pending'] += len(saved['pending_rollouts'])
                totals['terminal_programs'] += len(saved['completed_programs'])
                if totals['checkpoints'] % 100 == 0:
                    print(json.dumps({'phase': 'full-native-resume-audit', 'rank': rank, 'processed': len(seen), **totals}), flush=True)
            for item in lineage:
                raw = item[0] / 'raw_generations.jsonl'
                if item[0].parent.parent == SOURCE:
                    assert not raw.exists() or raw.stat().st_size == 0
            return totals, repairs, seen
        with ThreadPoolExecutor(max_workers=8) as pool:
            for shard_totals, shard_repairs, shard_ids in pool.map(audit_rank, range(8)):
                assert not seen.intersection(shard_ids)
                seen.update(shard_ids); repairs.extend(shard_repairs)
                for key, value in shard_totals.items(): totals[key] += value
        assert len(seen) == 3120
        assert totals['reports'] == 125 and totals['checkpoints'] == 2995
        # Reports also carry their completed attempts/terminal-program counts.
        with_reports_completed = totals['completed']
        with_reports_terminals = totals['terminal_programs']
        for rank in range(8):
            for report in (PREVIOUS / 'tree' / f'rank{rank:02d}' / 'evaluations').glob('*.json'):
                r = json.loads(report.read_text())
                with_reports_completed += r['mcts']['completed_rollouts']
                with_reports_terminals += r['collected_programs']
        assert with_reports_completed == 8874 and totals['pending'] == 19948
        assert with_reports_terminals == 4445
        result = {'denominator': 3120, 'all_ids_verified': True,
                  'model_generation': False, 'source_writes': False,
                  'active_rollout_leases_acquired': False,
                  'native_snapshot_roundtrip_verified': True,
                  'UID_paths_and_local_attempt_counters_preserved': True,
                  'source': str(SOURCE), 'totals': totals, 'repairs': repairs,
                  'completed_including_reports': with_reports_completed,
                  'terminal_programs_including_reports': with_reports_terminals,
                  'elapsed_seconds': time.monotonic() - started,
                  'producer_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), Path('scripts/mcts_resume.py'))}}
        durable_json(OUTPUT / 'audit_complete.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        durable_json(OUTPUT / 'audit_failed.json', {'error': str(exc), 'processed': len(seen), 'totals': totals})
        raise
    finally:
        core.close()


if __name__ == '__main__':
    main()
