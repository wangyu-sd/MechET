"""Launcher unit tests; never allocate GPU resources."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('name', ['run_taiji_dynamic_electron_tree_a100.sh',
                                  'run_taiji_frozen_route_search_a100.sh'])
def test_new_search_launch_requires_explicit_output_before_bootstrap(name):
    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != 'MECHET_TREE_OUTPUT'}
    env['MECHET_TREE_CODE_SHA'] = 'unit-placeholder'
    run = subprocess.run(['bash', str(repo / 'scripts' / name)], env=env,
                         capture_output=True, text=True, timeout=10)
    assert run.returncode != 0
    assert 'MECHET_TREE_OUTPUT' in run.stderr
    assert 'staging' not in run.stdout
