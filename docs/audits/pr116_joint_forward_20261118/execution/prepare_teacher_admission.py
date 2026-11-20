"""Freeze a one-GPU PR116 forward compatibility task on the complete TRAIN batch."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
from tempfile import TemporaryDirectory

from check_forward_model import file_sha


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RELEASE = ROOT/'artifacts/pr116_joint_20261118'
SUBMISSIONS = ROOT/'outputs/submissions'
DONORS = {'H20': 'pr116_step0_4h20_512_20261118_01', 'A100': 'pr116_step0_1a100_8_20261118_02'}


def freeze(gpu):
    bound = RELEASE/'teacher/bound_runtime.json'
    runtime = json.loads(bound.read_bytes())
    protocol = runtime['teacher_protocol']
    config = Path(runtime['teacher_config'])
    scorer = Path(runtime['teacher_script'])
    options = json.loads(config.read_bytes())
    weight = Path(options['checkpoint'])
    manifest = HERE/'full_first_train_batch_teacher_inputs/manifest.json'
    queries = manifest.with_name('queries.json')
    batch = json.loads(manifest.read_bytes())
    if (protocol['figshare_file'] != 55904912 or file_sha(config) != protocol['config_sha256']
            or file_sha(scorer) != protocol['script_sha256']
            or file_sha(weight) != protocol['weights_sha256']
            or options['checkpoint_sha256'] != protocol['weights_sha256']
            or batch['source_round'] != 0 or batch['train_targets'] != 32
            or file_sha(queries) != batch['queries_sha256']):
        raise ValueError('teacher admission requires the official weights and frozen TRAIN input bytes')

    source = SUBMISSIONS/DONORS[gpu]
    observed = json.loads((source/'platform_receipt.json').read_bytes())
    template = json.loads((source/'task_config.private.json').read_bytes())
    if (observed['platform_state'] != 'END' or observed['success'] is not True
            or template['task_flag'] != observed['task_flag'] or template['GPUName'] != gpu):
        raise ValueError('teacher admission requires a verified donor for the selected GPU pool')

    label = f'pr116_forward_admission_1{gpu.lower()}_20261119_01'
    folder = SUBMISSIONS/label
    if folder.exists():
        raise FileExistsError(folder)
    flag = 'meteor_mechet_'+label
    checker, heartbeat = 'check_forward_model.py', 'taiji_run_with_heartbeat.sh'
    shared = ROOT/'configs/runtime/shared_host_storage.env'
    with TemporaryDirectory(dir=SUBMISSIONS, prefix=label+'-') as temp:
        staged = Path(temp)/'frozen'
        staged.mkdir()
        shutil.copyfile(HERE/checker, staged/checker)
        shutil.copyfile(RELEASE/'code/scripts'/heartbeat, staged/heartbeat)
        bindings = {str(p): file_sha(p) for p in (bound, config, manifest, queries, scorer, shared)}
        bindings.update({str(folder/name): file_sha(staged/name) for name in (checker, heartbeat)})
        contract = staged/'contract.json'
        contract.write_text(json.dumps(dict(bound_config=str(bound), queries=str(queries),
            batch_manifest=str(manifest), gpu_kind=gpu, world_size=1, files_sha256=bindings), indent=2)+'\n')
        launch = shlex.join(['exec', 'bash', str(folder/heartbeat), runtime['teacher_python'], '-B',
                             str(folder/checker), '--frozen-contract', str(folder/'contract.json')])
        start = f'source {shlex.quote(str(shared))} && TAIJI_HEARTBEAT_LABEL={flag} TAIJI_HEARTBEAT_SECONDS=45 {launch}'
        token = json.loads(Path('/root/.taijiconfig').read_bytes())['user']['token']
        template.update(task_flag=flag, readable_name=f'MechET PR116 FlowER admission 1 {gpu}',
            host_num=1, host_gpu_num=1, GPUName=gpu, keep_running_after_trainer_finish=False,
            task_description='Frozen official forward model, complete 32-target TRAIN reaction queries.',
            start_cmd='bash -lc '+shlex.quote(start), token=token, Token=token)
        path = staged/'task_config.private.json'
        with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as dest:
            json.dump(template, dest)
        receipt = dict(flag=flag, stage='forward_admission', world=1, device=gpu, targets=32,
                       teacher_queries=batch['distinct_teacher_queries'], contract_sha256=file_sha(contract))
        (staged/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
        os.rename(staged, folder)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu', choices=DONORS)
    print(freeze(parser.parse_args().gpu))
