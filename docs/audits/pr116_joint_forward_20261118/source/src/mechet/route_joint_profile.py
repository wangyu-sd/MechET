"""Compare four-H20 teacher placements using one frozen, complete TRAIN batch."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
from tempfile import mkdtemp
from threading import Event, Thread
import time

from .external_route_data import file_sha
from .route_joint_cache import content_key
from .route_joint_schedule import ACTOR_SHA
from .route_training_contract import atomic_json

MODES = ('colocated', 'time_slice')
MEMORY_FRACTION = .95


def _read(path):
    return json.loads(Path(path).read_bytes())


def bind_profile_source(control_contract, batch_manifest):
    batch_manifest = Path(batch_manifest)
    cfg = _read(control_contract)
    batch = _read(batch_manifest)
    root = Path(cfg['output_dir'])
    first = root/'checkpoints'/'round-000001'
    receipt = _read(first/'complete.json')
    if (cfg['step_budget']['arm'] != 'R' or cfg['parent_sha256'] != ACTOR_SHA
            or cfg['world_size'] != 4 or cfg['gpu_name'] != 'H20' or cfg['inference_backend'] != 'vllm'
            or batch['control_contract_sha256'] != file_sha(control_contract)
            or batch['original_actor_sha256'] != ACTOR_SHA or batch['source_round'] != 0
            or batch['source_checkpoint_receipt_sha256'] != file_sha(first/'complete.json')
            or batch['train_targets'] != 32 or receipt['visited_target_episodes'] != 32
            or receipt['optimizer_step'] != 1 or len(receipt['round_metrics']) != 4):
        raise ValueError('profile requires the complete frozen initial R training batch')
    groups = {}
    write_timings = []
    for rank in range(4):
        label = f'rank{rank:02d}'
        groups[str(root/'round-000000'/label/'rewarded_episodes.json')] = batch['source_rank_group_sha256'][label]
        files = sorted((root/'runtime_profiles'/label).glob('cold-000001-*.json'))
        traces = [_read(path) for path in files]
        publication = [t['phases']['checkpoint.publish']['sum_seconds'] for t in traces
                       if t['contract_sha256'] == batch['control_contract_sha256']
                       and 'checkpoint.publish' in t['phases']]
        if not publication:
            raise ValueError('profile requires the recorded H20 checkpoint publication cost')
        write_timings.append(max(publication))
    queries = batch_manifest.parent/'queries.json'
    if file_sha(queries) != batch['queries_sha256']:
        raise ValueError('profile teacher input manifest changed')
    return dict(manifest=str(batch_manifest.resolve()), manifest_sha256=file_sha(batch_manifest),
                expected_teacher_queries=batch['distinct_teacher_queries'],
                expected_teacher_input_sha256=content_key(_read(queries)),
                group_files_sha256=groups, optimizer_seed_path=str(first/'optimizer.pt'),
                optimizer_seed_sha256=receipt['payload_sha256']['optimizer.pt'],
                collection_seconds=max(s['sampling_seconds'] for s in receipt['round_metrics']),
                checkpoint_seconds=max(write_timings), memory_fraction=MEMORY_FRACTION)


def choose_profile(reports, protocol, source_manifest_sha256, queries):
    if len(reports) != 2 or {r['profile'] for r in reports} != set(MODES):
        raise ValueError('compare both measured teacher placements')
    valid = []
    for report in reports:
        if (report['source_manifest_sha256'] != source_manifest_sha256 or report['protocol'] != protocol
                or report['train_targets'] != 32 or len(report['hardware']) != 4
                or any('H20' not in device for device in report['hardware'])):
            raise ValueError('teacher profiles require a common frozen H20 condition')
        if report['status'] == 'PASS':
            elapsed = report['complete_step_cost_seconds']
            if (report['teacher_queries'] != queries or not 0 < report['observed_teacher_queries'] <= queries
                    or not math.isfinite(elapsed) or elapsed <= 0):
                raise ValueError('invalid full-batch teacher measurement')
            valid.append(report)
        elif report['status'] != 'FAIL':
            raise ValueError('teacher profile measurement is incomplete')
    if not valid:
        raise ValueError('no measured teacher placement fits')
    best = min(valid, key=lambda r: (-r['observed_teacher_queries'], -r['valid_conditional_groups'],
                                     r['complete_step_cost_seconds'], MODES.index(r['profile'])))
    return best['profile']


@contextmanager
def observe_device(torch, local):
    stop = Event()
    memory = {'device_total_bytes': 0, 'peak_observed_used_bytes': 0, 'memory_samples': 0}

    def sample():
        free, total = torch.cuda.mem_get_info(local)
        memory['device_total_bytes'] = total
        memory['peak_observed_used_bytes'] = max(memory['peak_observed_used_bytes'], total-free)
        memory['memory_samples'] += 1

    def loop():
        try:
            while not stop.wait(.2):
                sample()
        except Exception as exc:
            memory['error'] = repr(exc)

    worker = None
    try:
        sample()
        worker = Thread(target=loop, daemon=True)
        worker.start()
    except Exception as exc:
        memory['error'] = repr(exc)
    try:
        yield memory
    finally:
        stop.set()
        if worker is not None and worker.is_alive():
            worker.join()
        try:
            sample()
        except Exception as exc:
            memory['error'] = repr(exc)


def run_profile(cfg, actor, reference, optimizer, local, root):
    import torch
    import torch.distributed as dist
    from .external_route_rl import optimize_routes
    from .route_joint_phase import attach_joint_evidence
    from .route_vllm_backend import bind_rollout_adapter

    source = cfg['profile_replay']
    options = dict(cfg['joint_forward'])
    mode = options['profile']
    rank = dist.get_rank()
    hardware = torch.cuda.get_device_name(local)
    error = ''
    try:
        if torch.cuda.device_count() != 4 or 'H20' not in hardware:
            raise ValueError('teacher profile requires exactly four visible H20s')
        path, digest = sorted(source['group_files_sha256'].items())[rank]
        if file_sha(path) != digest:
            raise ValueError('profile source receipts changed')
        groups = _read(path)
        seed = torch.load(source['optimizer_seed_path'], map_location='cpu', weights_only=True)
        optimizer.load_state_dict(seed)
        del seed
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()
    except Exception as exc:
        error = repr(exc)
    errors = [None]*4
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError('cannot prepare the frozen joint profile: '+repr(errors))
    shared = [None]
    if rank == 0:
        try:
            shared[0] = (mkdtemp(prefix='profile-cold-cache-', dir=root), None)
        except OSError as exc:
            shared[0] = (None, repr(exc))
    dist.broadcast_object_list(shared, src=0)
    cache, cache_error = shared[0]
    if cache_error:
        raise RuntimeError(cache_error)
    options['cache_dir'] = cache
    cfg['joint_forward'] = options
    teacher = {}; learner = {}; memory = {}; elapsed = 0.
    try:
        with observe_device(torch, local) as memory:
            try:
                bind_rollout_adapter(cfg, 'round000000:'+cfg['parent_sha256'])
            except Exception as exc:
                error = repr(exc)
            errors = [None]*4
            dist.all_gather_object(errors, error)
            if any(errors):
                error = 'teacher profile native warmup failed: '+repr(errors)
            else:
                began = time.monotonic()
                teacher = attach_joint_evidence(groups, cfg)
                learner = optimize_routes(actor, reference, optimizer, groups, cfg)
                torch.cuda.synchronize(local)
                elapsed = time.monotonic()-began
    except Exception as exc:
        error = repr(exc)
    parts = [None]*4
    dist.all_gather_object(parts, dict(hardware=hardware, memory=memory, error=error,
        teacher=teacher, learner=learner, measured_seconds=elapsed))
    outcome = [None]
    if rank == 0:
        try:
            expected = source['expected_teacher_queries']
            passed = all(not x['error'] and not x['memory'].get('error')
                         and x['memory']['peak_observed_used_bytes'] <= source['memory_fraction']*x['memory']['device_total_bytes']
                         and x['teacher']['global_distinct'] == expected
                         and x['teacher']['global_executed_queries'] == expected
                         and x['teacher']['global_input_sha256'] == source['expected_teacher_input_sha256']
                         and x['teacher']['available'] > 0 and x['learner']['optimizer_updated'] for x in parts)
            dynamic = max(x['measured_seconds'] for x in parts)
            report = dict(profile=mode, status='PASS' if passed else 'FAIL',
                contract_sha256=cfg['_sha'], protocol=options['teacher_protocol'],
                source_manifest_sha256=source['manifest_sha256'], train_targets=32,
                hardware=[x['hardware'] for x in parts], teacher_queries=expected,
                observed_teacher_queries=parts[0]['teacher'].get('available',0),
                valid_conditional_groups=sum(x['learner'].get('conditional_groups',0) for x in parts),
                complete_step_cost_seconds=source['collection_seconds']+dynamic+source['checkpoint_seconds'],
                archived_collection_seconds=source['collection_seconds'], archived_checkpoint_seconds=source['checkpoint_seconds'],
                measured_teacher_and_learner_seconds=dynamic, memory_fraction=source['memory_fraction'],
                per_rank=parts)
            atomic_json(root/'teacher_profile_receipt.json', report)
            outcome[0] = report['status'], None
        except Exception as exc:
            outcome[0] = 'FAIL', repr(exc)
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    dist.broadcast_object_list(outcome, src=0)
    if outcome[0][1]:
        raise RuntimeError('teacher profile publication failed: '+outcome[0][1])
    return outcome[0][0]
