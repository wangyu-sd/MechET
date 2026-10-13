"""Fallback dependencies must never shadow packages in the CUDA11 base."""
import os
from pathlib import Path
import subprocess
import sys


BOOTSTRAP = Path(__file__).resolve().parents[1] / 'scripts/vllm_cuda11_sitecustomize.py'


def test_missing_dependency_imports_without_shadowing_base(tmp_path):
    base = tmp_path / 'base'
    fallback = tmp_path / 'dependencies'
    base.mkdir(); fallback.mkdir()
    (base / 'sitecustomize.py').write_bytes(BOOTSTRAP.read_bytes())
    (base / 'priority_fixture.py').write_text('origin = "cuda11-base"\n')
    (fallback / 'priority_fixture.py').write_text('origin = "wrong-fallback"\n')
    (fallback / 'missing_fixture.py').write_text('value = 73\n')
    result = subprocess.run([sys.executable, '-c',
        'import priority_fixture, missing_fixture; '
        'assert priority_fixture.origin == "cuda11-base"; '
        'assert missing_fixture.value == 73'], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(base),
             'MECHET_VLLM_PUREPY_FALLBACK': str(fallback)})
    assert result.returncode == 0, result.stderr
