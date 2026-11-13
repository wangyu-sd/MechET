#!/usr/bin/env python3
"""Wait on CPU dependencies, then create/start one immutable eight-H20 task."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src'), str(Path(__file__).resolve().parents[1])]
from mechet.qwen3_continuation import shared_path, verify_files
from mechet.qwen3_queue import cache_readiness, validate_submission, waiting_status, submission_inputs
from mechet.route_training_contract import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    plan_path = shared_path(args.plan)
    plan = json.loads(plan_path.read_text())
    cfg = json.loads(shared_path(plan['training_config']).read_text())
    task_path = shared_path(plan['task_config'])
    task = json.loads(task_path.read_text())
    validate_submission(cfg, task)
    directory = plan_path.parent
    with (directory/'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = directory/'queue_status.json'
        status = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if status.get('state') == 'submitted':
            print(json.dumps(status), flush=True)
            return
        verify_files(plan['files_sha256'])
        while True:
            ready = cache_readiness(cfg)
            if ready == 'cache_verified':
                break
            status = waiting_status(status, task_flag=task['task_flag'])
            atomic_json(receipt_path, status)
            print(json.dumps(status), flush=True)
            if not args.wait:
                return
            time.sleep(30)
        cfg, task = submission_inputs(plan)  # Recheck all bindings AFTER indefinite CPU waiting.
        client = str(shared_path(plan['taiji_client']))
        # An interrupted submission must be resolved against the same task flag,
        # never create a second name or silently allocate another eight GPUs.
        if status.get('state') in ('submit_intent', 'created', 'submission_uncertain'):
            listed = subprocess.run([client, 'instance_list', task['task_flag']],
                                    capture_output=True, text=True, timeout=45)
            ids = re.findall(r'\b[0-9a-f]{32}\b', listed.stdout)
            if ids:
                atomic_json(receipt_path, dict(state='submitted', task_flag=task['task_flag'],
                    instances=ids, recovered_intent=True, updated_at=time.time()))
                return
            raise RuntimeError('RESOLVE_EXISTING_SUBMIT_INTENT_BEFORE_RETRY')
        atomic_json(receipt_path, dict(state='submit_intent', task_flag=task['task_flag'], updated_at=time.time()))
        created = subprocess.run([client, 'create', '--type', 'task', '--simple_config', str(task_path)],
                                 capture_output=True, text=True, timeout=120)
        if created.returncode or '[error]' in created.stdout.lower():
            atomic_json(receipt_path, dict(state='submission_uncertain', task_flag=task['task_flag'], updated_at=time.time()))
            raise RuntimeError('TAIJI_CREATE_NOT_CONFIRMED; inspect task flag, private command output not echoed')
        atomic_json(receipt_path, dict(state='created', task_flag=task['task_flag'], updated_at=time.time()))
        started = subprocess.run([client, 'start', '--task_flag', task['task_flag']],
                                 capture_output=True, text=True, timeout=120)
        if started.returncode or '[error]' in started.stdout.lower():
            raise RuntimeError('TAIJI_START_NOT_CONFIRMED; task created, do not recreate')
        status = dict(state='submitted', task_flag=task['task_flag'], updated_at=time.time(),
                      gpu_count=8, gpu_type='H20', cache_verified=True)
        atomic_json(receipt_path, status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
