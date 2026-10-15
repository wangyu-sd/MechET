import json
from pathlib import Path
import pytest


def test_bootstrap_reopens_existing_ray_store_instead_of_treating_wal_as_node_files(tmp_path):
    from scripts.start_ray_mcts_actor import storage_paths
    from mechet.shared_electron_search import durable_json
    old = tmp_path / 'existing_actor'; old.mkdir()
    durable_json(old / 'contract.json', {'contract': {'x': 1}, 'legacy_root': '/original/nodes'})
    manifest = {'tree_store_backend': 'ray_memory_grouped_WAL_v1', 'mcts_shared_root': str(old)}
    current, legacy = storage_paths(manifest, tmp_path / 'new_output')
    assert current == old and legacy == Path('/original/nodes')


def test_legacy_bootstrap_uses_read_only_original_namespace(tmp_path):
    from scripts.start_ray_mcts_actor import storage_paths
    current, legacy = storage_paths({'mcts_shared_root': '/frozen/legacy'}, tmp_path)
    assert current == tmp_path / 'ray_tree_store'
    assert legacy == Path('/frozen/legacy')


def test_bootstrap_rejects_mixed_shard_contracts(tmp_path):
    from scripts.start_ray_mcts_actor import load_resume_manifests
    for rank in range(8):
        p = tmp_path / 'tree' / f'rank{rank:02d}'; p.mkdir(parents=True)
        (p / 'manifest.json').write_text(json.dumps({'rank': rank, 'world': 8,
            'source_sha': 'correct' if rank < 7 else 'wrong', 'shard_rows': 390}))
    with pytest.raises(ValueError, match='contract'):
        load_resume_manifests(tmp_path)


def test_ray_launcher_requires_explicit_resume_source_before_environment_setup(tmp_path):
    import os
    import subprocess
    env = {**os.environ, 'MECHET_USE_RAY': '1', 'MECHET_TREE_CODE_SHA': 'unit-fixture',
           'MECHET_TREE_OUTPUT': str(tmp_path / 'new'),
           'MECHET_PRESERVED_GREEDY_OUTPUT': str(tmp_path / 'old_greedy')}
    env.pop('MECHET_TREE_RESUME_FROM', None)
    script = Path(__file__).resolve().parents[1] / 'scripts/run_taiji_single_mcts_rolling_a100.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'resume source required' in result.stderr
