#!/usr/bin/env python3
"""Authenticate own Stage-II parent, index original bytes and freeze full-TRAIN jobs.

This CPU preparation never samples a policy or submits a Taiji task.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pr96_earho_contract import TRACKS, validate
from pr96_earho_storage import build_index, IndexedJsonl


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda:reader.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def task_payload(cfg, *, deployment, archive, contract, suffix):
    track, count, cards, gpu = cfg['track'], cfg['train_reactions'], cfg['world_size'], cfg['gpu_name']
    template = Path(__file__).resolve().parents[1]/'configs/taiji/meteor_mechet_pr96_flower_qwen35_08b_2stage_8h20_zjk_20261023_04.json'
    payload = json.loads(template.read_text())
    region = 'zjk' if gpu == 'H20' else 'qy'
    flag = f'meteor_mechet_pr96_{track}_earho_train{count}_{cards}{gpu.lower()}_{region}_{suffix}'
    payload.update(task_flag=flag, host_num=1, host_gpu_num=cards, GPUName=gpu,
        readable_name=f'meteor MechET PR96 {track} EARHO all{count} TRAIN ({cards}x{gpu})',
        output_base_model=f'MechET-PR96-{track}-EARHO-Qwen35-08B',
        task_description=f'Own authenticated PR96 {track} StageII parent, ALL {count} accepted source-arrow TRAIN reactions; original VALID denominator {cfg["validation_reactions"]}; not unfiltered endpoint data. Refreshed-policy K2 first-divergence frontier, successor-class pooling, adaptive H2->12,25%full-root rehearsal, clipped actor loss/fixed-original-parent sampledKL/separately accounted fallback replay. Same global64, two passes/128-root round. Shared byte indexes, round-bounded replay and bucketed gradients. No graph/learned critic/TEST labels or diagnostic adapter. Immutable parent-matching source/native CUDA11 runtime/Tencenttaiji6/proven privateCeph/default45sstdout. Actual GPU sampling and updates require post-allocation verification.',
        start_cmd=f"bash -lc 'TAIJI_HEARTBEAT_LABEL=meteor-mechet-pr96-{track}-earho TAIJI_HEARTBEAT_SECONDS=45 TAIJI_MIRROR_PID1_STDOUT=1 exec bash {deployment}/scripts/taiji_run_with_heartbeat.sh bash {deployment}/scripts/taiji_pr96_earho_bootstrap.sh {archive} {cfg['entry_archive_sha256']} {contract}'")
    if gpu == 'A100':
        payload.update(business_flag='AI4LSLab_KEMO_QY_A100', location='qy', quota_type='public')
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entry-archive', type=Path, required=True)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--entry-commit', required=True)
    p.add_argument('--suffix', required=True)
    p.add_argument('--track', choices=tuple(TRACKS), default='mech31k')
    p.add_argument('--world-size', type=int, choices=(4,8), default=4)
    p.add_argument('--gpu-name', choices=('H20', 'A100'), default='H20')
    p.add_argument('--indexes', type=Path, help='Optional CPU-prepared indexes, still hash-bound and source-stat verified')
    args = p.parse_args()
    root = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
    track = args.track
    flower = track == 'flower'
    run = root/'outputs/agent'/('pr96_flower_qwen35_08b_mainonly_20261023_06' if flower else 'pr96_mech31k_qwen35_08b_mainonly_20261022_01')
    ref = root/'data'/('pr96_flower_mainonly_source_arrows_v2_20261023_06' if flower else 'pr96_mech31k_mainonly_source_arrows_v2_20261022_01')
    approval = root/'data'/('pr96_flower_reference_approval_20261023_06' if flower else 'pr96_mech31k_reference_approval_20261022_01')/'reference_approval.json'
    count = TRACKS[track][1]
    submit = root/'outputs/submissions'/f'pr96_{track}_earho_train{count}_{args.suffix}'
    submit.mkdir(parents=True, exist_ok=False)
    output = root/'outputs/agent'/f'pr96_{track}_earho_train{count}_{args.suffix}'
    parent = run/('models/history/checkpoint-89442' if flower else 'models/history/checkpoint-660')
    parent_cfg = run/('performance_v1/history.yaml' if flower else 'history.yaml')
    import yaml
    parent_settings = yaml.safe_load(parent_cfg.read_text())['pr96']
    preflight = Path(parent_settings['preflight_manifest'])
    tokens = run/'tokens/history/manifest.json'
    history_manifest = root/'data'/('pr96_flower_mainonly_history_v2_20261023_06' if flower else 'pr96_mech31k_mainonly_history_v2_20261022_01')/'manifest.json'
    base = root/'artifacts/pr96_qwen35_base_manifest_20261022.json'
    train, history = ref/'inverse/train.jsonl', run/'views/history/train.jsonl'
    valid = root/'data'/('flower_inverse_tool_sft_action_delta_v1' if flower else 'mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824')/'valid.jsonl'
    protected = [train, history, valid, ref/'manifest.json', parent_cfg, base, approval, preflight, tokens, history_manifest,
        ref/'inverse/valid.jsonl', ref/'state_sft/valid.jsonl', ref/'frozen_valid_cohorts.json',
        parent/'adapter_model.safetensors', parent/'adapter_config.json', parent/'pr96_checkpoint_manifest.json',
        run/('models/state/checkpoint-119256' if flower else 'models/state/checkpoint-3300')/'adapter_model.safetensors',
        root/'configs/runtime/shared_host_storage.env',
        root/'artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl']
    cfg = dict(algorithm='pr96_earho_v1', track=track, entry_commit=args.entry_commit,
        train_file=str(train), history_file=str(history), validation_file=str(valid),
        parent=str(parent), parent_sha256=sha(parent/'adapter_model.safetensors'), parent_config=str(parent_cfg),
        base_manifest=str(base), output_dir=str(output), train_reactions=count, validation_reactions=2890 if flower else 1319,
        original_trace_view_train=257167 if flower else 10152,
        reference_quarantined_train=13414 if flower else 15, full_endpoint_train=257171 if flower else 24959,
        world_size=args.world_size, gpu_name=args.gpu_name, candidates=2, products_per_round=128,
        initial_horizon=2, maximum_horizon=12, full_episode_fraction=.25, minimum_effective_groups=16,
        productive_support_threshold=.1, optimization_passes=2, global_batch_size=64,
        learning_rate=5e-6, clip_epsilon=.2, kl_beta=.02, replay_weight=.1, max_grad_norm=.5,
        seed=17, generation_batch_size=64, loss_batch_size=4, cpu_threads_per_rank=4, max_new_tokens=768, max_decisions=64,
        reward=dict(structural_endpoint=1., reference_first_successor=.25, invalid_first_action=-.1),
        test_used=False, reference_endpoint_in_policy_prompt=False, learned_critic=False,
        fixed_parent_reference=True, replay_only_for_no_support=True,
        reference_code_archive=str(root/'artifacts'/('pr96_code_c42c94b_performance_20261023_01.tar.gz' if flower else 'pr96_code_731af13_20261023_02.tar.gz')),
        reference_code_archive_sha256='1f9f52078e5fadc83e40794dcdf235e0fb58ae717c782584f3b9bfb6b7e8a09f' if flower else '38623c2cd0ea2358fffa353e16e30ca75631addde2fad5548d69d457066ab5b6',
        runtime_archive=str(root/'artifacts/pr96_runtime_tf550_causalconv170_Tr320_py311_20261023_01.tar.gz'),
        runtime_archive_sha256='cf0276843d01faf90fd2c13529c578d464673ab4d756e8d6a289ca295c26d911',
        entry_archive=str(args.entry_archive), entry_archive_sha256=sha(args.entry_archive))
    validate(cfg)
    cfg['parent_acceptance_evidence'] = dict(preflight=str(preflight), token_manifest=str(tokens),
        note='Immutable construction training_allowed:false is pending-gate history; completed own StageII binds actual deployed preflight and full token audit. Original manifests are not rewritten.')
    if (sha(preflight) != parent_settings['preflight_manifest_sha256']
            or sha(tokens) != parent_settings['token_cache_manifest_sha256']):
        raise ValueError('EARHO_PARENT_ACCEPTANCE_EVIDENCE_CHANGED')
    # One CPU-only scan before allocating GPUs; no regenerated decisions or labels.
    cfg['indexes'] = {}
    for key in ('train_file','history_file'):
        index = (args.indexes or submit)/f'{key}.sqlite'
        if args.indexes:
            with IndexedJsonl(cfg[key],index) as store:
                if len(store.ids()) != count:
                    raise ValueError('EARHO_INDEX_DENOMINATOR_CHANGED')
        else:
            build_index(cfg[key],index,grouped=(key=='history_file'),expected_reactions=count)
        cfg['indexes'][key] = str(index)
        protected.append(index)
    cfg['files_sha256'] = {}
    for path in protected:
        cfg['files_sha256'][str(path)] = sha(path)
        print(json.dumps(dict(event='earho-freeze-input',path=str(path))),flush=True)
    reference_manifest = json.loads((ref/'manifest.json').read_text())
    parent_manifest = json.loads((parent/'pr96_checkpoint_manifest.json').read_text())
    if (cfg['files_sha256'][str(train)] != reference_manifest['reports']['train']['outputs']['inverse']['sha256']
            or cfg['files_sha256'][str(history)] != parent_manifest['lineage']['train_file_sha256']):
        raise ValueError('EARHO_TRAIN_NOT_THE_PARENT_AUTHORIZED_REFERENCE')
    contract = submit/'contract.json'
    contract.write_text(json.dumps(cfg, indent=2)+'\n')
    payload = task_payload(cfg,deployment=args.deployment,archive=args.entry_archive,contract=contract,suffix=args.suffix)
    path = submit/'task_config.json'
    path.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(task_flag=payload['task_flag'], contract=str(contract), config=str(path), output=str(output))))


if __name__ == '__main__':
    main()
