"""Freeze an independent four-A100 task after the authentic JR corpus is exported."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT/'artifacts/pr116_joint_20261118'
SUBMISSIONS = ROOT/'outputs/submissions'
LABEL = 'pr116_offline_dpo_4a100_20261119_01'
FLAG = f'meteor_mechet_{LABEL}'
ENTRY_FILES = (
    'scripts/taiji_pr116_offline_dpo.sh',
    'scripts/taiji_run_with_heartbeat.sh',
    'scripts/prefetch_runtime_imports.py',
    'scripts/pr116_offline_launch_contract.py',
    'scripts/run_pr116_offline_dpo.py',
    'src/mechet/route_joint_offline_dpo.py',
    'src/mechet/route_transformers_import.py',
)


def freeze(corpus):
    sys.path.insert(0, str(RELEASE/'code'))
    from scripts.pr116_offline_launch_contract import digest, launch_inputs, read

    bindings = launch_inputs(corpus, RELEASE/'offline_dpo_runtime')
    donor = SUBMISSIONS/'pr116_step0_1a100_8_20261118_02'
    observed = read(donor/'platform_receipt.json')
    if observed['platform_state'] != 'END' or observed['success'] is not True:
        raise ValueError('offline DPO A100 donor did not complete')
    task = read(donor/'task_config.private.json')
    if task['task_flag'] != observed['task_flag'] or task['GPUName'] != 'A100':
        raise ValueError('offline DPO A100 allocation binding changed')
    folder = SUBMISSIONS/LABEL
    if folder.exists():
        raise FileExistsError(folder)
    with TemporaryDirectory(dir=SUBMISSIONS, prefix=LABEL+'-') as temp:
        working = Path(temp)/'frozen'
        working.mkdir()
        shutil.copytree(RELEASE/'code', working/'code', ignore=shutil.ignore_patterns('__pycache__'))
        code = folder/'code'
        cfg = dict(bindings, code_root=str(code), gpu_name='A100', world_size=4,
            output_dir=str(ROOT/'outputs/agent'/LABEL),
            entry_files_sha256={name: digest(working/'code'/name) for name in ENTRY_FILES})
        contract = working/'contract.json'
        contract.write_text(json.dumps(cfg, indent=2, sort_keys=True)+'\n')
        start = shlex.join(['exec', 'bash', str(code/'scripts/taiji_run_with_heartbeat.sh'),
            'bash', str(code/'scripts/taiji_pr116_offline_dpo.sh'), str(folder/'contract.json')])
        setup = f'TAIJI_HEARTBEAT_LABEL={FLAG} TAIJI_HEARTBEAT_SECONDS=45 TAIJI_MIRROR_PID1_STDOUT=1'
        task.update(task_flag=FLAG, readable_name='MechET PR116 offline native DPO 4 A100',
            GPUName='A100', host_num=1, host_gpu_num=4, keep_running_after_trainer_finish=False,
            task_description='TRL DPO on the frozen official-forward reward-only TRAIN preferences.',
            start_cmd='bash -lc '+shlex.quote(setup+' '+start))
        token = read(Path('/root/.taijiconfig'))['user']['token']
        task.update(token=token, Token=token)
        path = working/'task_config.private.json'
        with os.fdopen(os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600), 'w') as handle:
            json.dump(task, handle)
        manifest = dict(flag=FLAG, contract_sha256=digest(contract), stage='offline_dpo',
                        device='A100', world=4, pairs=bindings['pair_count'])
        (working/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        os.rename(working, folder)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    args = parser.parse_args()
    print(freeze(args.corpus))
