#!/usr/bin/env python3
"""Freeze a full-TRAIN EARHO contract and ordinary H20 payload; does not submit."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda:reader.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entry-archive', type=Path, required=True)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--entry-commit', required=True)
    p.add_argument('--suffix', default='20261023_01')
    p.add_argument('--gpu-name', choices=('H20', 'A100'), default='H20')
    args = p.parse_args()
    root = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
    run = root/'outputs/agent/pr96_mech31k_qwen35_08b_mainonly_20261022_01'
    ref = root/'data/pr96_mech31k_mainonly_source_arrows_v2_20261022_01'
    submit = root/'outputs/submissions'/f'pr96_mech31k_earho_train10137_{args.suffix}'
    submit.mkdir(parents=True, exist_ok=False)
    output = root/'outputs/agent'/f'pr96_mech31k_earho_train10137_{args.suffix}'
    parent = run/'models/history/checkpoint-660'
    parent_cfg = run/'history.yaml'
    base = root/'artifacts/pr96_qwen35_base_manifest_20261022.json'
    train = ref/'inverse/train.jsonl'
    history = run/'views/history/train.jsonl'
    valid = root/'data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/valid.jsonl'
    protected = [train, history, valid, ref/'manifest.json', parent_cfg, base,
        root/'data/pr96_mech31k_reference_approval_20261022_01/reference_approval.json',
        ref/'inverse/valid.jsonl', ref/'state_sft/valid.jsonl', ref/'frozen_valid_cohorts.json',
        parent/'adapter_model.safetensors', parent/'adapter_config.json', parent/'pr96_checkpoint_manifest.json',
        run/'models/state/checkpoint-3300/adapter_model.safetensors',
        root/'configs/runtime/shared_host_storage.env',
        root/'artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl']
    cfg = dict(algorithm='pr96_earho_v1', entry_commit=args.entry_commit,
        train_file=str(train), history_file=str(history), validation_file=str(valid),
        parent=str(parent), parent_sha256=sha(parent/'adapter_model.safetensors'), parent_config=str(parent_cfg),
        base_manifest=str(base), output_dir=str(output), train_reactions=10137, validation_reactions=1319,
        original_trace_view_train=10152, reference_quarantined_train=15, full_endpoint_train=24959,
        world_size=8, gpu_name=args.gpu_name, candidates=2, products_per_round=128, initial_horizon=2, maximum_horizon=12,
        full_episode_fraction=.25, minimum_effective_groups=16, productive_support_threshold=.1,
        optimization_passes=2, global_batch_size=64, learning_rate=5e-6, clip_epsilon=.2,
        kl_beta=.02, replay_weight=.1, max_grad_norm=.5, seed=17, generation_batch_size=64,
        cpu_threads_per_rank=2, max_new_tokens=768, max_decisions=64,
        reward=dict(structural_endpoint=1., reference_first_successor=.25, invalid_first_action=-.1),
        test_used=False, reference_endpoint_in_policy_prompt=False, learned_critic=False,
        fixed_parent_reference=True, replay_only_for_no_support=True,
        reference_code_archive=str(root/'artifacts/pr96_code_731af13_20261023_02.tar.gz'),
        reference_code_archive_sha256='38623c2cd0ea2358fffa353e16e30ca75631addde2fad5548d69d457066ab5b6',
        runtime_archive=str(root/'artifacts/pr96_runtime_tf550_causalconv170_Tr320_py311_20261023_01.tar.gz'),
        runtime_archive_sha256='cf0276843d01faf90fd2c13529c578d464673ab4d756e8d6a289ca295c26d911',
        entry_archive=str(args.entry_archive), entry_archive_sha256=sha(args.entry_archive),
        files_sha256={str(path):sha(path) for path in protected})
    contract = submit/'contract.json'
    contract.write_text(json.dumps(cfg, indent=2)+'\n')
    payload = json.loads((Path(__file__).resolve().parents[1]/'configs/taiji/meteor_mechet_pr96_flower_qwen35_08b_2stage_8h20_zjk_20261023_04.json').read_text())
    hardware = '8h20_zjk' if args.gpu_name == 'H20' else '8a100_qy'
    flag = f'meteor_mechet_pr96_mech31k_earho_train10137_{hardware}_{args.suffix}'
    payload.update(task_flag=flag,
        readable_name=f'meteor MechET PR96 mech31k EARHO StageIII all10137 TRAIN (8x{args.gpu_name})',
        GPUName=args.gpu_name,
        output_base_model='MechET-PR96-mech31k-EARHO-Qwen35-08B',
        task_description='Original three-stage EARHO continuation of authenticated PR96 History660, not TEST4 diagnostic: ALL10137 accepted TRAIN roots, repeated current-policy rollouts and first executed divergences, K2 successor-class alternatives, bounded adaptive H2->12 plus25%full-root rehearsal, success-gated clipped actor loss/fixed-original-parent sampled KL/separately accounted verified fallback replay. No graph/learned critic/TEST labels. Full original trace-view1319VALID before/after; not raw3120benchmark. Immutable original731 source/runtime, Tencenttaiji6/proven privateCeph/default45sstdout. Actual collection/update must be verified after allocation.',
        start_cmd=f"bash -lc 'TAIJI_HEARTBEAT_LABEL=meteor-mechet-pr96-earho TAIJI_HEARTBEAT_SECONDS=45 TAIJI_MIRROR_PID1_STDOUT=1 exec bash {args.deployment}/scripts/taiji_run_with_heartbeat.sh bash {args.deployment}/scripts/taiji_pr96_earho_bootstrap.sh {args.entry_archive} {cfg['entry_archive_sha256']} {contract}'")
    if args.gpu_name == 'A100':
        payload.update(business_flag='AI4LSLab_KEMO_QY_A100', location='qy', quota_type='public')
    path = submit/'task_config.json'
    path.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(task_flag=flag, contract=str(contract), config=str(path), output=str(output))))


if __name__ == '__main__':
    main()
