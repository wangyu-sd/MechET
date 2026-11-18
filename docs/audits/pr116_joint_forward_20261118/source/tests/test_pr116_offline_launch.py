from hashlib import sha256
import json

import pytest

from scripts.pr116_offline_launch_contract import launch_inputs, verify_launch


def write(path, value):
    path.write_text(json.dumps(value))
    return sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    source = tmp_path / 'reward-contract.json'
    source_run = tmp_path / 'reward-only'
    source_run.mkdir()
    protocol = {'figshare_file': 55904912, 'weights_sha256': 'a'*64}
    source_sha = write(source, dict(step_budget={'arm': 'JR'}, joint_forward=dict(
        lambda_f=.1, lambda_d=0., teacher_protocol=protocol),
        runtime_root=str(tmp_path / 'runtime'), output_dir=str(source_run)))
    done_sha = write(source_run / 'completion.json', dict(
        contract_sha256=source_sha, visited_target_episodes=2048))
    c = tmp_path / 'corpus'
    c.mkdir()
    rows = c/'pairs.jsonl'
    rows.write_text('{}\n')
    manifest = dict(format='pr116_reward_only_native_dpo_v1', trl_version='1.9.2',
                    pair_count=1, source_train_targets=2048, source_contract=str(source),
                    source_contract_sha256=source_sha, source_completion_sha256=done_sha,
                    pairs_sha256=sha256(rows.read_bytes()).hexdigest(), teacher_protocol=protocol)
    write(c/'manifest.json', manifest)
    o = tmp_path/'overlay'
    (o/'packages').mkdir(parents=True)
    (o/'wheels').mkdir()
    wheels = {}
    for name in ('trl-1.9.2-py3-none-any.whl', 'datasets-4.7.0-py3-none-any.whl'):
        (o/'wheels'/name).write_bytes(b'wheel fixture')
        wheels[name] = sha256(b'wheel fixture').hexdigest()
    write(o/'runtime_admission_indexed.json', dict(raw_native_tokens_preserved=True,
          versions={'trl': '1.9.2', 'datasets': '4.7.0'}, wheels=wheels))
    return c, o, source, manifest


def test_offline_launch_binds_exact_pair_bytes_and_completed_reward_run(tmp_path):
    c, o, source, manifest = fixture(tmp_path)
    result = launch_inputs(c, o)
    assert result['corpus_manifest'] == str(c/'manifest.json')
    assert result['corpus_manifest_sha256'] == sha256((c/'manifest.json').read_bytes()).hexdigest()
    assert result['pair_count'] == 1
    assert result['runtime_root'] == str(tmp_path/'runtime')
    assert result['offline_packages'] == str(o/'packages')
    (c/'pairs.jsonl').write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='preference'):
        launch_inputs(c, o)


def test_offline_launch_refuses_unfinished_or_empty_corpus(tmp_path):
    c, o, source, manifest = fixture(tmp_path)
    (tmp_path/'reward-only'/'completion.json').unlink()
    with pytest.raises(FileNotFoundError):
        launch_inputs(c, o)
    manifest['pair_count'] = 0
    write(c/'manifest.json', manifest)
    with pytest.raises(ValueError, match='nonempty'):
        launch_inputs(c, o)


def test_offline_bootstrap_verifies_bound_runtime_and_exact_code(tmp_path):
    c, o, _, _ = fixture(tmp_path)
    launch = launch_inputs(c, o)
    entry = tmp_path/'learner.py'
    entry.write_text('pass\n')
    launch.update(code_root=str(tmp_path), gpu_name='A100', world_size=4,
                  entry_files_sha256={'learner.py': sha256(entry.read_bytes()).hexdigest()})
    cfg = tmp_path/'offline-launch.json'
    write(cfg, launch)
    assert verify_launch(cfg) == launch
    launch['world_size'] = 1
    write(cfg, launch)
    with pytest.raises(ValueError, match='binding'):
        verify_launch(cfg)
    launch['world_size'] = 4
    write(cfg, launch)
    entry.write_text('print(1)\n')
    with pytest.raises(ValueError, match='code'):
        verify_launch(cfg)
