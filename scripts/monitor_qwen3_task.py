#!/usr/bin/env python3
"""Persist actual platform/checkpoint health after submission, without retries."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time


def parse_instances(text):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    rows = []
    for line in text.splitlines():
        cols = [v.strip() for v in line.split('|')]
        if len(cols) >= 8 and re.fullmatch(r'[0-9a-fA-F]{32}', cols[2]):
            rows.append(dict(created_at=cols[1], instance=cols[2], success=cols[3].lower()=='true',
                             platform_state=cols[4], platform_time=cols[6]))
    if not rows:
        raise ValueError('NO_PLATFORM_INSTANCE_ROW')
    return max(rows, key=lambda row: row['created_at'])


def checkpoint_progress(output):
    observed = []
    for directory in Path(output).glob('checkpoint-*'):
        try:
            state = json.loads((directory/'trainer_state.json').read_text())
            observed.append((int(state['global_step']), directory))
        except (OSError, ValueError, KeyError):
            continue
    if not observed:
        return dict(saved_optimizer_step=None, saved_checkpoint=None)
    step, directory = max(observed, key=lambda item: item[0])
    return dict(saved_optimizer_step=step, saved_checkpoint=str(directory),
                binding_manifest_present=(directory/'qwen3_continuation_manifest.json').is_file())


def atomic_json(path, value):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-dir', type=Path, required=True)
    args=parser.parse_args();directory=args.submission_dir.resolve()
    if not directory.is_relative_to('/aaa/fionafyang/buddy1'):
        raise ValueError('SHARED_STORAGE_REQUIRED')
    receipt=json.loads((directory/'receipt.json').read_text())
    cfg=json.loads(Path(receipt['training_config']).read_text())
    client=Path('/aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/taiji_mount_bootstrap/taiji_client')
    last_state=None;last_log=0
    while True:
        now=datetime.datetime.now().astimezone().isoformat()
        try:
            result=subprocess.run([str(client),'instance_list',receipt['task_flag']],capture_output=True,text=True,timeout=45)
            if result.returncode:raise RuntimeError('INSTANCE_LIST_FAILED')
            state=parse_instances(result.stdout)
            state.update(observed_at=now,task_flag=receipt['task_flag'],**checkpoint_progress(cfg['output_dir']))
            state['health']=('completed' if state['success'] else 'failed') if state['platform_state']=='END' else 'active'
            atomic_json(directory/'health.json',state)
            with (directory/'health.jsonl').open('a') as stream:stream.write(json.dumps(state)+'\n')
            if state['platform_state']!=last_state or (state['platform_state']=='TRAINING_RUNNING' and time.monotonic()-last_log>300):
                logs=subprocess.run([str(client),'logs','--tail','150','--timestamps',receipt['task_flag'],state['instance']],capture_output=True,text=True,timeout=45)
                # Launcher logs may contain private init diagnostics. Keep them
                # private and never echo their contents into health/PR records.
                fd=os.open(directory/'launcher_latest.private.log',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                with os.fdopen(fd,'w') as stream:stream.write(logs.stdout+'\n'+logs.stderr)
                last_log=time.monotonic()
            print(json.dumps(state),flush=True);last_state=state['platform_state']
            if state['platform_state']=='END':return
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            atomic_json(directory/'monitor_error.json',dict(observed_at=now,error_type=type(error).__name__))
        time.sleep(45)


if __name__=='__main__':main()
