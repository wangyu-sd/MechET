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


def test_route_reuse_requires_greedy_runtime_pin_before_bootstrap(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    env={k:v for k,v in os.environ.items() if k!='MECHET_GREEDY_CODE_SHA'}
    env.update(MECHET_TREE_CODE_SHA='unit-placeholder',MECHET_TREE_OUTPUT=str(tmp_path/'new'),
               MECHET_ROUTE_GREEDY_OUTPUT=str(tmp_path/'old'))
    run=subprocess.run(['bash',str(repo/'scripts/run_taiji_frozen_route_search_a100.sh')],
                       env=env,capture_output=True,text=True,timeout=10)
    assert run.returncode!=0
    assert 'MECHET_GREEDY_CODE_SHA' in run.stderr
    assert 'staging' not in run.stdout


@pytest.mark.parametrize('alias',[True,False])
def test_route_recovery_cannot_overwrite_preserved_or_partial_output(tmp_path,alias):
    repo=Path(__file__).resolve().parents[1]
    old=tmp_path/'old';old.mkdir()
    new=old if alias else tmp_path/'partial'
    if not alias:new.mkdir()
    env={**os.environ,'MECHET_TREE_CODE_SHA':'unit-placeholder',
         'MECHET_TREE_OUTPUT':str(new),'MECHET_ROUTE_GREEDY_OUTPUT':str(old),
         'MECHET_GREEDY_CODE_SHA':'b'*40}
    run=subprocess.run(['bash',str(repo/'scripts/run_taiji_frozen_route_search_a100.sh')],
                       env=env,capture_output=True,text=True,timeout=10)
    assert run.returncode!=0
    assert 'new output' in run.stderr
    assert 'staging' not in run.stdout
