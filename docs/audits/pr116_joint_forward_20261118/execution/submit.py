"""Register, launch and read back one immutable Taiji task."""
import argparse
from datetime import datetime
import fcntl
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile


def publish(path, data):
    with NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as record:
        os.fchmod(record.fileno(), 0o644)
        record.write(data)
        record.flush()
        os.fsync(record.fileno())
    Path(record.name).replace(path)


def start_once(folder, observe, issue_start):
    """Reconcile a lost response before issuing another request for a task flag."""
    accepted = folder/'start.accepted'
    requested = folder/'start.requested'
    state = observe()
    if state.get('instance') or accepted.exists():
        if not accepted.exists():
            publish(accepted, datetime.now().astimezone().isoformat())
        return state
    if requested.exists():
        raise RuntimeError('previous Taiji start is unconfirmed; observe the same task flag')
    publish(requested, datetime.now().astimezone().isoformat())
    try:
        issue_start()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        state = observe()
        if not state.get('instance'):
            raise RuntimeError('Taiji start is unconfirmed; retain the existing request and observe') from exc
    else:
        publish(accepted, datetime.now().astimezone().isoformat())
        return observe()
    publish(accepted, datetime.now().astimezone().isoformat())
    return state


def main(folder, mode):
    folder = folder.resolve()
    manifest = json.loads((folder/'manifest.json').read_bytes())
    flag = manifest['flag']
    contract = folder/'contract.json'
    config = folder/'task_config.private.json'
    registered = folder/'registered.json'
    client = Path('artifacts/taiji_mount_bootstrap/taiji_client').resolve()
    parser_path = 'outputs/submissions/pr111_full_native_d3_resume_20261118_04/monitor_qwen3_task.py'
    spec = spec_from_file_location('taiji_monitor', parser_path)
    monitor = module_from_spec(spec)
    spec.loader.exec_module(monitor)

    def call(operation, *args):
        result = subprocess.run([str(client), operation, *map(str, args)], capture_output=True, timeout=55)
        data = (result.stdout+result.stderr).decode(errors='replace')
        path = folder/f'{operation}.private.log'
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w') as receipt:
            receipt.write(data)
        if result.returncode or '[error]' in data.lower():
            raise RuntimeError(f'{operation} failed; see private receipt')
        return data

    def require_remote_contract():
        detail = call('task_detail', flag)
        if str(contract) not in detail:
            raise ValueError('remote registration path mismatch')

    def register():
        digest = sha256(contract.read_bytes()).hexdigest()
        if digest != manifest['contract_sha256']:
            raise RuntimeError('frozen contract changed')
        record = dict(flag=flag, contract_sha256=digest)
        if registered.exists():
            if json.loads(registered.read_bytes()) != record:
                raise RuntimeError('registration changed')
            require_remote_contract()
            return
        try:
            call('create', '--type', 'task', '--scfg', config)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            try:
                require_remote_contract()
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                raise RuntimeError('Taiji registration could not be verified') from exc
        else:
            require_remote_contract()
        publish(registered, json.dumps(record)+'\n')

    def observe():
        listing = call('instance_list', flag)
        if listing.strip() == '[info][list empty]' and registered.is_file():
            state = 'REGISTERED_WITHOUT_INSTANCE'
            if (folder/'start.accepted').is_file():
                state = 'START_ACCEPTED_AWAITING_INSTANCE'
            elif (folder/'start.requested').is_file():
                state = 'START_UNCONFIRMED'
            return dict(instance=None, platform_state=state, success=None)
        return monitor.parse_instances(listing)

    if mode == 'observe':
        result = observe()
    else:
        with (folder/'submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            register()
            if mode == 'register':
                print(registered.read_text())
                return
            result = start_once(folder, observe, lambda: call('start', '--scfg', config, '--task_flag', flag))
    result.update(task_flag=flag, observed_at=datetime.now().astimezone().isoformat())
    publish(folder/'platform_receipt.json', json.dumps(result, indent=2)+'\n')
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('mode', choices=('register', 'start', 'observe'))
    args = parser.parse_args()
    main(args.folder, args.mode)
