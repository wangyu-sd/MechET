"""Formal prepared independent-route controls: validate, run, collect.

Validation and collection are CPU/file operations. Model execution is a
separate subcommand requiring explicit authorization; this module never
submits a platform task. Episode counts are independent full-route attempts,
not chemical-depth limits, ranked answers, or new MCTS budgets.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]

from mechet.analysis.literature_sources import file_sha256
from mechet.dynamic_electron_tree import chemical_key
from mechet.reverse_route_network import stock_closure
from mechet.shared_electron_search import durable_json, fingerprint as record_fingerprint
from scripts.independent_route_control import fingerprint, prepare_plan
from scripts.independent_route_runtime import make_vllm_transport, run_episodes, summarize_reports

ACTOR_SHA = '40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75'
PLAN_SHA = '49f0889eb3b630ee30184fdfe33050954ec5d0b6c3406c1cb2fcc461b8ed73d9'


@dataclass(frozen=True)
class ValidatedPlan:
    path: Path
    sha256: str
    manifest: dict
    rows: list
    paired_ids: list
    stocks: dict


def validate_plan(plan_path, assets_path, *, expected_actor_sha=ACTOR_SHA, expected_plan_sha=PLAN_SHA):
    """Reproduce the original full preparation, not just its claimed hashes.

    Expected-hash parameters support isolated software fixtures; the CLI does
    not expose overrides. The formal CLI remains pinned to the approved plan.
    """
    path, assets_path = Path(plan_path).resolve(), Path(assets_path).resolve()
    if file_sha256(path) != expected_plan_sha:
        raise ValueError('approved independent plan checksum differs')
    plan = json.loads(path.read_bytes())
    source_files = plan['source_files']
    freezes = [Path(p) for p in source_files if Path(p).name == 'prediction_freeze_manifest.json']
    if len(freezes) != 1 or str(assets_path) not in source_files:
        raise ValueError('plan must bind original prediction freeze and route assets')
    if file_sha256(assets_path) != source_files[str(assets_path)]:
        raise ValueError('route assets changed')
    adapter = Path(plan['actor_weight_path']).parent
    # The unchanged original preparer rechecks every source, physical actor,
    # eight runtime headers, all460 IDs and the fixed200 observed pairing.
    with tempfile.TemporaryDirectory(prefix='mechet-independent-plan-check-') as temporary:
        regenerated_root = Path(temporary)/'rederived'
        regenerated = prepare_plan(freezes[0].parent, assets_path, adapter,
                                  regenerated_root, expected_actor_sha=expected_actor_sha)
        if regenerated != plan:
            raise ValueError('independent plan differs from full source rederivation')
        for name, record in plan['output_files'].items():
            saved = path.parent/name
            if (file_sha256(saved) != record['sha256'] or
                    saved.read_bytes() != (regenerated_root/name).read_bytes()):
                raise ValueError('prepared full-population artifact changed: '+name)
    rows = [json.loads(line) for line in (path.parent/'targets_product_only.jsonl').read_text().splitlines()]
    paired = json.loads((path.parent/'paired_observed_target_ids.json').read_bytes())
    assets = json.loads(assets_path.read_bytes())
    stocks_by_name = {name: frozenset(Path(s['canonical_path']).read_text().splitlines())
                      for name, s in assets['stocks'].items()}
    stocks = {c: stocks_by_name[value['stock']] for c, value in assets['cohorts'].items()}
    return ValidatedPlan(path, expected_plan_sha, plan, rows, paired, stocks)


def shard_contract(loaded, rank):
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('formal independent control requires rank0..7 of eight workers')
    dependencies = sorted({REPO/'scripts/independent_route_cli.py',
        REPO/'scripts/independent_route_runtime.py', REPO/'scripts/independent_route_control.py',
        REPO/'scripts/rolling_vllm_search.py', REPO/'scripts/vllm_safe_lora_expand.py',
        REPO/'scripts/run_natural_language_value_search.py',
        REPO/'scripts/build_natural_language_event_sft.py',
        REPO/'scripts/eval_natural_language_event_local.py', *(REPO/'src/mechet').glob('*.py')})
    return {'schema': 'independent_full_route_worker_v1', 'rank': rank, 'world': 8,
        'plan_sha256': loaded.sha256, 'actor_weight_sha256': loaded.manifest['actor_weight_sha256'],
        'full_target_denominator': 460, 'paired_observed_denominator': 200,
        'assigned_target_ids': [r['id'] for r in loaded.rows[rank::8]],
        'episode_counts': [1, 4, 10], 'episodes_per_target': 10, 'seed': 17,
        'sampling': loaded.manifest['sampling'], 'shared_frontier': False,
        'shared_policy_samples': False, 'reference_visible_to_policy': False,
        'stop_after_first_success': False, 'matched_compute_claim': False,
        'producer_files': {str(p.relative_to(REPO)): file_sha256(p) for p in dependencies}}


def run_worker(loaded, output_root, *, rank, authorized_model_execution=False,
               inflight=128, cpu_workers=16, transport_factory=make_vllm_transport):
    if not authorized_model_execution:
        raise PermissionError('separate model-execution authorization required')
    contract = shard_contract(loaded, rank)
    root = Path(output_root)/'tree'/f'rank{rank:02d}'
    root.mkdir(parents=True, exist_ok=False)
    durable_json(root/'manifest.json', contract)
    startup_done = threading.Event()
    def startup_heartbeat():
        while not startup_done.wait(45):
            print({'phase': 'independent-actor-initialization', 'rank': rank,
                   'full_target_denominator': 460, 'policy_decoding_started': False}, flush=True)
    heartbeat = threading.Thread(target=startup_heartbeat, name='independent-startup-log', daemon=True)
    heartbeat.start()
    interrupted = threading.Event()
    old_term = None
    if threading.current_thread() is threading.main_thread():
        old_term = signal.signal(signal.SIGTERM, lambda *_: interrupted.set())
    transport = None
    print({'phase': 'independent-source-contract-verified', 'rank': rank,
           'assigned_targets': len(contract['assigned_target_ids']), 'full_target_denominator': 460}, flush=True)
    try:
        tokenizer, transport, runtime = transport_factory(Path(loaded.manifest['actor_weight_path']).parent,
            authorized_model_execution=True, inflight=inflight, seed=17)
        durable_json(root/'backend_runtime.json', runtime)
        startup_done.set()
        progress = run_episodes(loaded.rows[rank::8], loaded.stocks, tokenizer, transport, root/'runtime',
            episode_count=10, seed=17, inflight=inflight, cpu_workers=cpu_workers,
            max_model_len=8192, heartbeat_seconds=45, stop_event=interrupted)
        durable_json(root/'worker_finished.json', {'contract': contract, 'progress': progress})
        return progress
    except BaseException as exc:
        durable_json(root/'worker_error.json', {'error_type': type(exc).__name__, 'message': str(exc),
                                              'full_result_complete': False})
        raise
    finally:
        startup_done.set(); heartbeat.join(timeout=1)
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
        close = getattr(transport, 'close', None)
        if close is not None:
            close()


def collect_results(loaded, output_root):
    """Collect available complete records without dropping absent/censored IDs.

    Audits raw returned token work even for episodes lacking final reports.
    Atomic checksum envelopes are read once; temporary files are not results.
    This is not an assertion that a live writer has stopped.
    """
    root = Path(output_root)
    reports, missing, rank_status = [], [], []
    audit = hashlib.sha256(); artifacts = 0
    raw_total = progress_total = 0
    work_complete = True

    def read(path):
        nonlocal artifacts
        data = path.read_bytes(); value = json.loads(data)
        if record_fingerprint(value['payload']) != value['sha256']:
            raise ValueError('corrupt independent result record: '+str(path))
        audit.update((str(path.relative_to(root))+'\0'+hashlib.sha256(data).hexdigest()+'\n').encode())
        artifacts += 1
        return value['payload']

    unexpected = [p.name for p in (root/'tree').glob('rank*') if p.name not in {f'rank{i:02d}' for i in range(8)}]
    if unexpected:
        raise ValueError('unexpected independent shard directories')
    for rank in range(8):
        shard = root/'tree'/f'rank{rank:02d}'
        if not shard.exists():
            missing.append(rank); work_complete = False
            continue
        header = read(shard/'manifest.json')
        if header != shard_contract(loaded, rank):
            raise ValueError('independent worker source/protocol/shard contract differs')
        assigned = loaded.rows[rank::8]
        identities = {fingerprint(['independent_complete_route_v1', r['id'],
            chemical_key(r['product_unmapped']), ep, 17]): (r, ep) for r in assigned for ep in range(10)}
        runtime = shard/'runtime'; rank_tokens = 0; rank_reports = 0
        for episode in sorted((runtime/'episodes').glob('*')):
            if episode.name not in identities:
                raise ValueError('foreign independent episode outside assigned shard')
            row, ep = identities[episode.name]
            requests = {p.stem: read(p) for p in sorted((episode/'requests').glob('*.json'))}
            raw_records = {p.stem: read(p) for p in sorted((episode/'raw_outputs').glob('*.json'))}
            generations = {p.stem: read(p) for p in sorted((episode/'generations').glob('*.json'))}
            episode_tokens = 0
            for index, raw in raw_records.items():
                req = requests.get(index)
                if (req is None or req['episode_uid'] != episode.name or raw['episode_uid'] != episode.name
                        or req['context'] != raw['context'] or raw['draw_index'] != req['draw_index']
                        or index != f"{raw['draw_index']:09d}"):
                    raise ValueError('raw output has no matching private episode input')
                episode_tokens += sum(len(g['token_ids']) for g in raw['outputs'])
            rank_tokens += episode_tokens
            report_path = episode/'report.json'
            if not report_path.exists():
                work_complete = False
                continue
            report = read(report_path)
            if (report['id'] != row['id'] or report['episode_index'] != ep or
                    report['episode_uid'] != episode.name or report['cohort'] != row['cohort']):
                raise ValueError('report target/episode/shard ownership differs')
            if report['observed_generation_work']['sampled_tokens'] != episode_tokens:
                raise ValueError('raw token evidence missing or differs from reported work')
            if report['draws'] != len(generations) or not set(generations) <= set(raw_records):
                raise ValueError('parsed generation/draw evidence differs')
            if report['search_complete'] and not (len(requests) == len(raw_records) == report['draws']):
                raise ValueError('complete episode lacks complete input/output evidence')
            tree = read(episode/'tree.json')
            target = chemical_key(row['product_unmapped']); stock = loaded.stocks[row['cohort']]
            if tree['case_id'] != row['id'] or tree['target'] != target:
                raise ValueError('saved tree belongs to a different target')
            discovered = {target}|{p for edge in tree['reaction_edges'] for p in edge['precursors']}
            closed = stock_closure(stock & discovered, tree['reaction_edges'])
            if (report['already_in_stock'] != (target in stock) or
                    report['stock_solved'] != (target not in stock and target in closed)):
                raise ValueError('reported stock-AND closure differs from saved reaction graph')
            journals = list((episode/'journal').glob('*/contract.json'))
            if len(journals) != 1 or read(journals[0])['case_id'] != episode.name:
                raise ValueError('private native journal missing or belongs to another episode')
            event_paths = sorted((journals[0].parent/'events').glob('*.json'))
            if not event_paths or [p.name for p in event_paths] != [f'{i:09d}.json' for i in range(len(event_paths))]:
                raise ValueError('native journal contains missing commits')
            events = [read(p) for p in event_paths]
            if events[0].get('kind') != 'initial' or any(e.get('kind') != 'expansion' for e in events[1:]):
                raise ValueError('invalid native journal event sequence')
            if report['search_complete'] and len(events) != report['draws']+1:
                raise ValueError('complete episode lacks native transition evidence')
            reports.append(report); rank_reports += 1
        raw_total += rank_tokens
        progress_path = runtime/'progress.json'
        progress = read(progress_path) if progress_path.exists() else None
        progress_tokens = progress['output_tokens'] if progress is not None else 0
        progress_total += max(rank_tokens, progress_tokens)
        rank_complete = bool(progress and progress['full_result_complete'] and
                             rank_reports == len(assigned)*10 and
                             progress['expected_episodes'] == len(assigned)*10)
        if not rank_complete or progress_tokens != rank_tokens or progress.get('unobserved_aborted_output_work'):
            work_complete = False
        rank_status.append({'rank': rank, 'reported_episodes': rank_reports,
            'raw_output_tokens': rank_tokens, 'progress_output_tokens': progress_tokens,
            'progress_claims_shard_complete': bool(progress and progress['full_result_complete']),
            'full_shard_records_complete': rank_complete})
    result = summarize_reports(loaded.rows, reports, episode_counts=[1, 4, 10], paired_ids=loaded.paired_ids)
    result['reported_episode_output_tokens'] = result.pop('output_tokens')
    result.update({'schema': 'independent_full_route_collection_v1', 'plan_sha256': loaded.sha256,
        'full_target_denominator': 460, 'paired_observed_denominator': 200,
        'missing_ranks': missing, 'rank_status': rank_status, 'raw_output_tokens': raw_total,
        'output_tokens_lower_bound': progress_total, 'work_complete': work_complete,
        'complete': not missing and work_complete and result['prefixes']['10']['pending_or_censored_episodes'] == 0,
        'audited_artifact_count': artifacts, 'audited_artifact_digest': audit.hexdigest(),
        'new_model_inference': False, 'collection_scope': 'atomic available records; not a liveness check'})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'run', 'collect'):
        sub = commands.add_parser(name)
        sub.add_argument('--plan', type=Path, required=True)
        sub.add_argument('--route-assets', type=Path, required=True)
        if name != 'validate':
            sub.add_argument('--output', type=Path, required=True)
        if name == 'run':
            sub.add_argument('--rank', type=int, required=True)
            sub.add_argument('--inflight', type=int, default=128)
            sub.add_argument('--cpu-workers', type=int, default=16)
            sub.add_argument('--authorize-model-execution', action='store_true')
        elif name == 'collect':
            sub.add_argument('--receipt', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'run' and not args.authorize_model_execution:
        parser.error('run requires separate --authorize-model-execution; validation alone never loads a model')
    loaded = validate_plan(args.plan, args.route_assets)
    if args.command == 'validate':
        result = {'schema': 'independent_plan_validation_v1', 'complete': True,
            'plan_sha256': loaded.sha256, 'full_target_denominator': len(loaded.rows),
            'paired_observed_denominator': len(loaded.paired_ids), 'new_model_inference': False,
            'shard_target_counts': [len(shard_contract(loaded, i)['assigned_target_ids']) for i in range(8)]}
    elif args.command == 'run':
        result = run_worker(loaded, args.output, rank=args.rank, authorized_model_execution=True,
                            inflight=args.inflight, cpu_workers=args.cpu_workers)
    else:
        result = collect_results(loaded, args.output)
        if args.receipt is not None:
            if args.receipt.exists():
                raise FileExistsError('preserve previous collection receipt: '+str(args.receipt))
            durable_json(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
