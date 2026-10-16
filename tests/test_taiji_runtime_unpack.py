"""Regression: minimal Tencent images need no external zstd executable."""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/unpack_taiji_runtime.sh'


def archive(tmp_path):
    bundle = tmp_path / 'runtime.tar.gz'
    with tarfile.open(bundle, 'w:gz') as stream:
        item = tarfile.TarInfo('runtime/version.txt')
        payload = b'pinned-runtime\n'
        item.size = len(payload)
        stream.addfile(item, io.BytesIO(payload))
    target = tmp_path / 'unpacked'
    target.mkdir()
    return bundle, hashlib.sha256(bundle.read_bytes()).hexdigest(), target


def invoke(bundle, sha, target):
    # This matches the minimal image boundary: no conda zstd on PATH.
    return subprocess.run(['/bin/bash', str(SCRIPT), str(bundle), sha, str(target)],
                          env={**os.environ, 'PATH': '/usr/bin:/bin'},
                          capture_output=True, text=True)


def test_gzip_runtime_unpacks_without_conda_zstd(tmp_path):
    bundle, sha, target = archive(tmp_path)
    result = invoke(bundle, sha, target)
    assert result.returncode == 0, result.stderr
    assert (target / 'runtime/version.txt').read_bytes() == b'pinned-runtime\n'


def test_bad_runtime_hash_never_extracts(tmp_path):
    bundle, _, target = archive(tmp_path)
    result = invoke(bundle, '0' * 64, target)
    assert result.returncode != 0
    assert 'FAILED' in result.stdout + result.stderr
    assert not list(target.iterdir())
