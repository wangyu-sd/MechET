#!/usr/bin/env python3
"""Freeze committed code, pin task commit, then recover proven private Ceph init."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def freeze_runtime(repo, destination, *, crn_root=None):
    repo, destination = Path(repo), Path(destination)
    if destination.exists():
        raise FileExistsError('immutable runtime already exists')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '--name-only', commit], text=True).splitlines()
    paths = [p for p in ('scripts', 'src', 'configs', 'pyproject.toml', 'README.md') if p in tracked]
    if not paths:
        raise ValueError('no tracked runtime paths')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.joint-runtime-', dir=destination.parent))
    archive = staging/'runtime.tar'
    subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', '-o', str(archive), commit, *paths], check=True)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(staging)], check=True)
    # The tar is generated from our own verified commit, not external data.
    archive.unlink()
    if crn_root is not None:
        import shutil
        shutil.copytree(Path(crn_root)/'src', staging/'crn_snapshot'/'src',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = dict(commit=commit, sha256={str(p.relative_to(staging)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(staging.rglob('*')) if p.is_file()})
    (staging/'runtime_provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    os.replace(staging, destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--donor-task', required=True)
    parser.add_argument('--client', type=Path, default=Path('/usr/local/bin/taiji_client'))
    parser.add_argument('--crn-root', type=Path,
                        default=ROOT.parent/'CRN-JEPA-pr18')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    # Refuse to snapshot code that differs from the tested committed tree.
    for path in ('scripts', 'src', 'configs'):
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', 'HEAD', '--', path], check=True)
    provenance = freeze_runtime(ROOT, args.runtime, crn_root=args.crn_root)
    config = json.loads(args.config.read_text())
    config['start_cmd'] = config['start_cmd'].replace('__VERIFIED_RUNTIME_COMMIT__', provenance['commit'])
    if '__VERIFIED_RUNTIME_COMMIT__' in config['start_cmd'] or str(args.runtime) not in config['start_cmd']:
        raise ValueError('runtime does not match task configuration')
    config['task_description'] += ' Runtime commit: '+provenance['commit']+'.'
    private = Path.home()/'.taiji'/'rendered_configs'
    private.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 0o700)
    prepared = private/(config['task_flag']+'.pinned-template.json')
    prepared.write_text(json.dumps(config, indent=2)+'\n')
    os.chmod(prepared, 0o600)
    print(json.dumps(dict(runtime=str(args.runtime), commit=provenance['commit'], config=str(prepared))), flush=True)
    subprocess.run([os.sys.executable, str(ROOT/'scripts/submit_taiji_with_donor_init.py'),
        'submit' if args.submit else 'render', '--config', str(prepared),
        '--donor-task', args.donor_task, '--client', str(args.client)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
