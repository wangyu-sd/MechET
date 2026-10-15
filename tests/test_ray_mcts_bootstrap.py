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


def test_runtime_server_accepts_a_separate_client_process(tmp_path):
    pytest.importorskip('ray')
    import os
    import subprocess
    import sys
    import time
    from mechet.shared_electron_search import durable_json
    from scripts.infer_single_reaction_mcts import SOURCE_SHA, ADAPTER_SHA
    from scripts.ray_mcts_store import single_planning_contract
    source = tmp_path / 'source'
    contract = single_planning_contract('native', 8192, 17, 10)
    for rank in range(8):
        p = source / 'tree' / f'rank{rank:02d}'; p.mkdir(parents=True)
        (p / 'manifest.json').write_text(json.dumps({'rank': rank, 'world': 8, 'shard_rows': 390,
            'source_sha': SOURCE_SHA, 'adapter_sha': ADAPTER_SHA, 'max_model_len': 8192,
            'shared_cache_namespace': 'native', 'seed': 17, 'program_goal': 10,
            'mcts_shared_root': str(tmp_path / 'legacy')}))
    durable_json(source / 'tree/rank00/mcts/fixture.json', {'planning_contract': contract})
    address = tmp_path / 'address.json'
    server = subprocess.Popen([sys.executable, '-u', 'scripts/start_ray_mcts_actor.py',
        '--source-output', str(source), '--output', str(tmp_path / 'new'),
        '--address-file', str(address)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=os.environ.copy())
    try:
        deadline = time.monotonic() + 35
        while not address.exists():
            if server.poll() is not None: pytest.fail(server.stdout.read())
            if time.monotonic() > deadline: pytest.fail('tree owner readiness timed out')
            time.sleep(.1)
        info = json.loads(address.read_text())
        client = subprocess.run([sys.executable, '-c',
            "import sys; from scripts.ray_mcts_store import RayMCTSStore; "
            "s=RayMCTSStore('single-reaction-tree',namespace=sys.argv[1],address=sys.argv[2]); "
            "print(s.liveness_metrics()['backend'])", info['namespace'], info['address']],
            capture_output=True, text=True, timeout=25)
        assert client.returncode == 0, client.stderr
        assert 'ray_memory_grouped_WAL_v1' in client.stdout
    finally:
        server.terminate()
        try: server.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill(); server.communicate()
