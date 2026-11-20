"""Advance the admitted PR116 jobs within one four-H20 teacher allocation."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from prepare_arm import profile_label, root, sub, release, freeze, sha
from submit import publish


HERE = Path(__file__).resolve().parent
POLL_SECONDS = 45
OFFLINE = 'pr116_offline_dpo_4a100_20261119_01'


def read(path):
    return json.loads(path.read_bytes())


def status(stage, **fields):
    state = {'stage': stage, **fields}
    record = HERE/'teacher_chain_status.json'
    if not record.is_file() or read(record) != state:
        publish(record, json.dumps(state, indent=2, sort_keys=True)+'\n')
        print(json.dumps(state), flush=True)


def platform(folder, mode):
    subprocess.run([sys.executable, '-B', str(HERE/'submit.py'), str(folder), mode],
                   cwd=root, check=True, timeout=240, stdout=subprocess.DEVNULL)
    return read(folder/'platform_receipt.json')


def launch(folder):
    status('submitting', task=folder.name)
    try:
        return platform(folder, 'start')
    except subprocess.SubprocessError:
        if not (folder/'start.requested').is_file():
            raise
        # A lost start response is reconciled by readback. Never start a second instance.
        status('reconciling_submission', task=folder.name)


def wait_for_end(folder):
    while True:
        try:
            state = platform(folder, 'observe')
        except (OSError, subprocess.SubprocessError) as exc:
            status('retrying_readback', task=folder.name, error=type(exc).__name__)
        else:
            status('waiting_for_task', task=folder.name, instance=state['instance'],
                   platform_state=state['platform_state'])
            if state['platform_state'] == 'END':
                if state['success'] is not True:
                    raise RuntimeError(f'Taiji task ended unsuccessfully: {folder.name}')
                return
        time.sleep(POLL_SECONDS)


def finish_profile(mode, start=False):
    folder = sub/profile_label(mode)
    if start:
        launch(folder)
    wait_for_end(folder)
    contract = folder/'contract.json'
    output = Path(read(contract)['output_dir'])
    measured = read(output/'teacher_profile_receipt.json')
    if measured['contract_sha256'] != sha(contract) or measured['profile'] != mode:
        raise ValueError(f'Taiji profile receipt does not bind {mode}')
    return measured


def submit_arm(arm, mode=None):
    slug = arm.lower().replace('-', '_')
    folder = sub/f'pr116_{slug}_64x32_4h20_20261118_01'
    if not (folder/'registered.json').is_file():
        status('freezing_contract', task=folder.name)
        freeze(arm, mode)
    launch(folder)
    return folder


def require_completed_training(folder):
    wait_for_end(folder)
    contract = folder/'contract.json'
    cfg = read(contract)
    output = Path(cfg['output_dir'])
    digest = sha(contract)
    complete = read(output/'completion.json')
    final = read(output/'checkpoints/round-000064/complete.json')
    if (complete['contract_sha256'] != digest or complete['visited_target_episodes'] != 2048
            or complete['validation_targets'] != 512):
        raise ValueError(f'Training completion does not match {folder.name}')
    if (final['contract_sha256'] != digest or final['next_round'] != 64
            or final['payload_sha256']['adapter_model.safetensors'] != complete['actor_sha256']):
        raise ValueError(f'Final checkpoint does not match {folder.name}')
    if cfg['step_budget']['arm'] in ('J', 'JR') and final['optimizer_step'] != 64:
        raise ValueError(f'Training ended without all 64 optimizer updates: {folder.name}')


def run():
    sys.path[:0] = [str(release/'code/src'), str(release/'code')]
    from mechet.route_joint_profile import choose_profile
    from mechet.route_joint_offline_dpo import export_corpus
    from prepare_offline_dpo import freeze as freeze_offline

    colocation = finish_profile('colocated')
    time_slice = finish_profile('time_slice', start=True)
    cfg = read(sub/profile_label('colocated')/'contract.json')
    replay = cfg['profile_replay']
    selected = choose_profile([colocation, time_slice], cfg['joint_forward']['teacher_protocol'],
                              replay['manifest_sha256'], replay['expected_teacher_queries'])
    status('measured_profile_selected', profile=selected)
    joint = submit_arm('J', selected)
    require_completed_training(joint)
    reward_only = submit_arm('JR', selected)
    require_completed_training(reward_only)
    distilled = submit_arm('J-distill')

    offline = sub/OFFLINE
    if not (offline/'manifest.json').is_file():
        corpus = release/'offline_native_dpo_corpus'
        if not (corpus/'manifest.json').is_file():
            status('exporting_reward_only_preferences', task=reward_only.name)
            export_corpus(reward_only/'contract.json', corpus)
        freeze_offline(corpus)
    launch(offline)
    require_completed_training(distilled)
    wait_for_end(offline)
    status('completed')


if __name__ == '__main__':
    with (HERE/'teacher_chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except Exception as exc:
            status('needs_attention', error=f'{type(exc).__name__}: {exc}')
            raise
