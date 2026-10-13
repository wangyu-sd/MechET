#!/usr/bin/env python3
"""Submit one immutable full3120 task using proven private Ceph init.

Generated private configuration never enters Git; the submission receipt
contains only public task/instance identifiers and frozen artifact hashes.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
SHARED = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
DONOR = 'meteor_mechet_pr83_flower_valid128_singlepass_8a100_qy_20261012_01'


def main():
    spec = importlib.util.spec_from_file_location('private_submission', SHARED/'scripts/submit_taiji_with_donor_init.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    config_path = REPO/'configs/taiji/meteor_mechet_uspto31k_full_dynamic_tree_8a100_qy_20261013.json'
    config = json.loads(config_path.read_text())
    helper.validate_submission_policy(config, config_path)
    if config['init_cmd'] != helper.PLACEHOLDER:
        raise ValueError('private credentials must not be committed')
    code_sha = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
    for path in ['src/mechet/dynamic_electron_tree.py','scripts/infer_dynamic_electron_trees.py',
                 'scripts/run_taiji_dynamic_electron_tree_a100.sh','scripts/summarize_dynamic_electron_trees.py']:
        subprocess.run(['git','-C',str(REPO),'diff','--exit-code','HEAD','--',path],check=True,capture_output=True)
    task = config['task_flag']
    receipt = SHARED/'outputs/submissions/dynamic_electron_tree_20261013/submission.json'
    if receipt.exists():
        raise ValueError('submission receipt exists; do not submit a duplicate job')
    config['start_cmd'] = config['start_cmd'].replace('REPLACE_DYNAMIC_TREE_CODE_SHA',code_sha)
    config['init_cmd'] = helper.recover_init_cmd(helper.DEFAULT_CLIENT,DONOR)
    helper.DEFAULT_RENDER_DIR.mkdir(parents=True,exist_ok=True)
    os.chmod(helper.DEFAULT_RENDER_DIR,0o700)
    private = helper.DEFAULT_RENDER_DIR/f'{task}.json'
    private.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    os.chmod(private,0o600)
    created = helper.run(helper.DEFAULT_CLIENT,'create','--type','task','--simple_config',str(private))
    if created.returncode or '[error]' in created.stdout.lower():
        raise RuntimeError(f'Taiji creation failed: {task}')
    started = helper.run(helper.DEFAULT_CLIENT,'start','--task_flag',task)
    if started.returncode or '[error]' in started.stdout.lower():
        raise RuntimeError(f'Taiji start failed: {task}')
    instances = helper.run(helper.DEFAULT_CLIENT,'instance_list',task)
    if instances.returncode:
        raise RuntimeError(f'cannot verify submission: {task}')
    identifiers = re.findall(r'\b[0-9a-f]{32}\b',instances.stdout,re.I)
    if not identifiers:
        raise RuntimeError(f'submitted instance absent: {task}')
    record = {'task_flag':task,'instance_id':identifiers[-1],'code_sha':code_sha,
              'donor_task':DONOR,'image':config['image_full_name'],'group':config['business_flag'],
              'location':config['location'],'gpus':8,'gpu_name':'A100','denominator':3120,
              'adapter_sha':'40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75',
              'source_sha':'819215c5d40720afd998aaf77b7b938045d8171a5487b26bf929897f65a957c4',
              'observed_instance_list':instances.stdout,'submission_success_not_execution_success':True}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
