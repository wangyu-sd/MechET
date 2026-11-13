"""Host-side dependency gate; never reserve GPUs while tokenizing."""
import json
import time
from pathlib import Path

from .qwen3_continuation import validate_cache, validate_config, verify_bindings


def waiting_status(prior, *, task_flag):
    # Readiness and submission lifecycle are different axes. An interrupted
    # create/start intent must survive a later missing-cache wait.
    result = dict(prior)
    result.update(readiness='waiting_for_cpu_cache', task_flag=task_flag, updated_at=time.time())
    if result.get('state') not in ('submit_intent', 'created', 'submission_uncertain'):
        result.update(state='waiting_for_cpu_cache', gpu_reserved=False)
    return result


def submission_inputs(plan):
    from .qwen3_continuation import verify_files, shared_path
    for key in ('training_config', 'task_config'):
        if str(shared_path(plan[key])) not in plan['files_sha256']:
            raise ValueError('QWEN3_SUBMISSION_INPUT_BINDING_REQUIRED')
    verify_files(plan['files_sha256'])
    cfg = json.loads(shared_path(plan['training_config']).read_text())
    task = json.loads(shared_path(plan['task_config']).read_text())
    validate_submission(cfg, task)
    return cfg, task


def cache_readiness(cfg):
    validate_config(cfg)
    manifest = Path(cfg['pretokenized_cache_dir'])/'manifest.json'
    if not manifest.exists():
        return 'waiting_for_cpu_cache'
    cache = json.loads(manifest.read_text())
    validate_cache(cfg, cache)
    verify_bindings(cfg, cache)
    return 'cache_verified'


def validate_submission(cfg, task):
    validate_config(cfg)
    if (cfg['qwen3_continuation']['world_size'] != 8
            or task.get('host_num') != 1 or task.get('host_gpu_num') != 8
            or task.get('business_flag') != 'AI4LSLab_KEMO_ZJK_H20'
            or task.get('keep_running_after_trainer_finish') is not False):
        raise ValueError('QWEN3_EIGHT_H20_SUBMISSION_REQUIRED')
