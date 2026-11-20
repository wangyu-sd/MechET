"""Validate the independent A100 launch before any learner starts."""
import json
from hashlib import file_digest
from pathlib import Path
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def launch_inputs(corpus, overlay):
    corpus, overlay = Path(corpus).resolve(), Path(overlay).resolve()
    manifest_path = corpus/'manifest.json'
    manifest = read(manifest_path)
    if (manifest['format'] != 'pr116_reward_only_native_dpo_v1' or manifest['trl_version'] != '1.9.2'
            or manifest['source_train_targets'] != 2048 or manifest['pair_count'] < 1):
        raise ValueError('offline DPO launch requires a nonempty native preference corpus')
    if digest(corpus/'pairs.jsonl') != manifest['pairs_sha256']:
        raise ValueError('offline DPO preference payload changed')
    source_path = Path(manifest['source_contract'])
    source = read(source_path)
    teacher = source['joint_forward']
    if (digest(source_path) != manifest['source_contract_sha256'] or source['step_budget']['arm'] != 'JR'
            or teacher['lambda_f'] != .1 or teacher['lambda_d'] != 0.
            or teacher['teacher_protocol'] != manifest['teacher_protocol']
            or teacher['teacher_protocol']['figshare_file'] != 55904912):
        raise ValueError('offline DPO launch requires the original forward-reward-only arm')
    done = Path(source['output_dir'])/'completion.json'
    completion = read(done)
    if (digest(done) != manifest['source_completion_sha256']
            or completion['contract_sha256'] != manifest['source_contract_sha256']
            or completion['visited_target_episodes'] != 2048):
        raise ValueError('offline DPO source completion changed')
    admission_path = overlay/'runtime_admission_indexed.json'
    admission = read(admission_path)
    wheels = {'trl-1.9.2-py3-none-any.whl', 'datasets-4.7.0-py3-none-any.whl'}
    if (not admission['raw_native_tokens_preserved'] or admission['versions']['trl'] != '1.9.2'
            or admission['versions']['datasets'] != '4.7.0' or set(admission['wheels']) != wheels
            or not (overlay/'packages').is_dir()
            or any(digest(overlay/'wheels'/name) != receipt for name, receipt in admission['wheels'].items())):
        raise ValueError('offline DPO indexed runtime compatibility receipt changed')
    return dict(corpus_manifest=str(manifest_path), corpus_manifest_sha256=digest(manifest_path),
                runtime_root=source['runtime_root'], offline_overlay=str(overlay),
                offline_packages=str(overlay/'packages'), runtime_admission_sha256=digest(admission_path),
                pair_count=manifest['pair_count'])


def verify_launch(contract_path):
    cfg = read(contract_path)
    inputs = launch_inputs(Path(cfg['corpus_manifest']).parent, cfg['offline_overlay'])
    if (any(cfg[k] != value for k, value in inputs.items()) or cfg['world_size'] != 4
            or cfg['gpu_name'] != 'A100'):
        raise ValueError('offline DPO launch binding changed')
    code_root = Path(cfg['code_root'])
    if any(digest(code_root/name) != checksum for name, checksum in cfg['entry_files_sha256'].items()):
        raise ValueError('offline DPO launch code changed')
    return cfg


if __name__ == '__main__':
    cfg = verify_launch(sys.argv[1])
    for key in ('runtime_root', 'offline_packages', 'corpus_manifest', 'corpus_manifest_sha256', 'output_dir'):
        print(cfg[key])
