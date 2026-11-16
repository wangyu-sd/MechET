"""Freeze an external, target-only route-RL contract; does not submit a job."""
import argparse
import json
import math
from pathlib import Path

from mechet.external_route_data import file_sha, iter_records, protected_identities
from mechet.route_training_contract import atomic_json
from mechet.qwen35_protocol import OFFICIAL_REVISION

ROOT = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
ALGORITHM = 'external_target_only_route_dr_grpo_v2'
VALIDATION_PROTOCOL = dict(id='external_native_valid_g4_v2', seed=17, candidates=4,
    temperature=1.0, top_p=1.0, top_k=0, fractions=[0.0,0.25,0.5,0.75,1.0],
    selection_metric='verified_closed_at_4_lower_bound',
    tie_break=['censored_root_fraction','decision_requests_per_root','earlier_checkpoint'])
APPROVED_PROTECTED_SOURCES = {
    str(ROOT/'outputs/submissions/flower513_r1r2_full460_direct_mcts_4h20_20261103_01/public_rows.jsonl'):
        'aa979451ce27e62865901fea4dde29c59e50583b2928b59ff59b931b4afffa8f',
    str(ROOT/'outputs/submissions/flower513_simpretro172_direct_mcts_4h20_20261103_01/public_rows.jsonl'):
        '6afef4c013ff9ff39b2f8fa7693ec187b70273c8f22008a59bfe79d44a8b051a',
}


def source_files(code_root):
    return sorted(str(p.relative_to(code_root)) for folder in ('src/mechet','scripts')
                  for p in (code_root/folder).rglob('*.py'))


def verify_contract(path):
    cfg = json.loads(Path(path).read_text())
    from mechet.route_vllm_contract import validate_runtime
    validate_runtime(cfg)
    from mechet.route_precision import precision_mode
    precision_mode(cfg)
    if type(cfg.get('phase_profiling',False)) is not bool:
        raise ValueError('PHASE_PROFILING_BOOL_REQUIRED')
    if 'policy_allow_tf32' in cfg and type(cfg['policy_allow_tf32']) is not bool:
        raise ValueError('EXTERNAL_POLICY_TF32_FLAG_REQUIRED_BOOL')
    from mechet.endpoints import ROUTE_ENDPOINT_CONTRACT
    if cfg.get('route_endpoint_contract') != ROUTE_ENDPOINT_CONTRACT:
        raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
    from mechet.route_process_reward import ALGORITHM as PROCESS_ALGORITHM, validate_process_protocol
    validate_process_protocol(cfg)
    process = cfg.get('algorithm')==PROCESS_ALGORITHM
    if cfg.get('algorithm') not in (ALGORITHM,PROCESS_ALGORITHM):
        raise ValueError('EXTERNAL_ROUTE_ALGORITHM_REQUIRED')
    from mechet.route_policy_episode import AGENTIC_ROUTE_OBSERVATIONS
    observation=cfg.get('public_observation_contract')
    if observation not in ('native_explicit_H_preserving_v1',*AGENTIC_ROUTE_OBSERVATIONS):
        raise ValueError('UNRECOGNIZED_ROUTE_OBSERVATION_CONTRACT')
    if observation in AGENTIC_ROUTE_OBSERVATIONS and (type(cfg.get('max_consecutive_recoveries')) is not int
                                           or cfg['max_consecutive_recoveries']<1):
        raise ValueError('INVALID_RECOVERY_CEILING')
    from mechet.external_route_actor import verify_actor_binding
    verify_actor_binding(cfg)
    if (cfg.get('gt_weight') != 0. or cfg.get('learned_critic') is not False
            or cfg.get('test_used') is not False or cfg.get('episode_token_budget') is not None
            or cfg.get('optimization_passes') != 1):
        raise ValueError('FORBIDDEN_EXTERNAL_ROUTE_TRAINING_CONDITION')
    if any(k in cfg for k in ('history_file','train_file','teacher_actions')):
        raise ValueError('PRIVATE_TEACHER_DATA_IN_TRAINING_CONTRACT')
    from mechet.route_process_reward import VERSION as PROCESS_REWARD_VERSION
    expected_advantage=('centered_action_return_to_go_no_std' if process else 'centered_group_rewards_no_std')
    expected_reward=(PROCESS_REWARD_VERSION if process else
                     'hidden_split_reference_trusted_ancestry_AND_stock_closure_v2')
    if (cfg.get('advantage_estimator') != expected_advantage
            or cfg.get('loss_type') != 'dr_grpo' or cfg.get('scale_rewards') is not False
            or cfg.get('reward_contract') != expected_reward
            or cfg.get('validation_protocol') != VALIDATION_PROTOCOL
            or cfg.get('adam_betas') != [0.9,0.999] or cfg.get('adam_eps') != 1e-8
            or cfg.get('weight_decay') != 0.0):
        raise ValueError('APPROVED_DR_GRPO_PROTOCOL_CHANGED')
    if cfg['candidates'] < 2 or cfg['train_targets'] < 1 or cfg['epochs'] < 1:
        raise ValueError('INVALID_FULL_TRAINING_POPULATION')
    if cfg.get('gpu_name') not in ('H20','A100'):
        raise ValueError('EXPLICIT_GPU_ALLOCATION_REQUIRED')
    for key in ('world_size','products_per_round','generation_batch_size','loss_batch_size',
                'executor_workers_per_rank','cpu_threads_per_rank','max_context','max_action_tokens'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('INVALID_TRAINING_CAPACITY:'+key)
    for key in ('learning_rate','loss_token_normalizer','max_grad_norm','behavior_logprob_tolerance'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('INVALID_TRAINING_OBJECTIVE:'+key)
    coalesce=cfg.get('executor_coalesce_seconds',0.)
    if not isinstance(coalesce,(int,float)) or not math.isfinite(coalesce) or coalesce<0:
        raise ValueError('INVALID_EXECUTOR_COALESCING_DURATION')
    from mechet.route_generation_capacity import generation_batches,capacity_kwargs
    if type(cfg.get('training_gradient_checkpointing',False)) is not bool:
        raise ValueError('INVALID_POLICY_ACTIVATION_CHECKPOINTING')
    from mechet.route_training_batches import training_batches
    list(training_batches([],cfg['loss_batch_size'],cfg.get('training_max_padded_tokens',131072)))
    generation_batches([],max_batch_size=cfg['generation_batch_size'],**capacity_kwargs(cfg))
    if (cfg.get('starting_actor_condition')=='corrected_flower_stage2_history'
            and capacity_kwargs(cfg)['qkv_channels']!=6144):
        raise ValueError('GENERATION_QKV_CHANNELS_MISMATCH')
    if not 0 < cfg['clip_epsilon'] < 1 or not math.isfinite(cfg['kl_beta']) or cfg['kl_beta'] < 0:
        raise ValueError('INVALID_CLIP_OR_KL')
    if cfg['max_action_tokens'] >= cfg['max_context'] or not 0 <= cfg['partial_reward_weight'] < 1:
        raise ValueError('INVALID_CONTEXT_OR_REWARD')
    for name,expected in cfg['files_sha256'].items():
        if file_sha(name) != expected:
            raise ValueError('FROZEN_INPUT_HASH_CHANGED:'+name)
    code_root=Path(__file__).resolve().parents[1]
    from mechet.route_runtime_efficiency import profile_environment
    profile_environment(cfg,code_root)
    from mechet.route_transformers_import import validate_import_index
    validate_import_index(cfg)
    if source_files(code_root) != sorted(cfg['code_files_sha256']):
        raise ValueError('FROZEN_SOURCE_INVENTORY_CHANGED')
    for name,expected in cfg['code_files_sha256'].items():
        if file_sha(code_root/name) != expected:
            raise ValueError('FROZEN_CODE_HASH_CHANGED:'+name)
    from mechet.external_route_schedule import validate_deferred_baseline
    validate_deferred_baseline(cfg,code_root)
    return cfg


def prepare_contract(dataset, parent, base_manifest, output, contract_path, *,
                     expected_protected_records=632, world_size=4, candidates=4, epochs=1,
                     approved_protected_sources=None, gpu_name='H20',vllm_runtime=None):
    dataset, parent, base_manifest = Path(dataset).resolve(), Path(parent).resolve(), Path(base_manifest).resolve()
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('schema') != 'external_target_only_route_rl_v2' or manifest['reference_action_labels']:
        raise ValueError('EXTERNAL_TARGET_ONLY_DATASET_REQUIRED')
    expected_isolation={'train_original_molecules_intersect_valid_roots',
        'train_original_reactions_intersect_valid_reactions',
        'train_original_molecules_intersect_benchmark_targets',
        'train_root_ids_intersect_valid_root_ids'}
    audit=manifest.get('cross_split_audit',{})
    if not expected_isolation.issubset(audit) or any(audit[k]!=0 for k in expected_isolation):
        raise ValueError('CLEAN_V2_TRAIN_VALID_ISOLATION_REQUIRED')
    frozen_valid=manifest.get('fixed_validation',{})
    if (frozen_valid.get('targets') != manifest['validation_targets']
            or frozen_valid.get('public_sha256') != file_sha(dataset/'validation.jsonl')):
        raise ValueError('FROZEN_VALIDATION_POPULATION_CHANGED')
    if sum(manifest['protected_source_records'].values()) != expected_protected_records:
        raise ValueError('PROTECTED_EVALUATION_DENOMINATOR_CHANGED')
    approved=APPROVED_PROTECTED_SOURCES if approved_protected_sources is None else approved_protected_sources
    if manifest['protected_sources'] != approved or any(file_sha(p)!=h for p,h in approved.items()):
        raise ValueError('PROTECTED_EVALUATION_IDENTITIES_NOT_APPROVED')
    identities,_=protected_identities(approved)
    if set(json.loads((dataset/'protected_identities.json').read_text())) != identities:
        raise ValueError('PROTECTED_IDENTITY_UNION_CHANGED')
    validation_rows=list(iter_records(dataset/'validation.jsonl'))
    from scripts.train_external_route_rl import validate_public_tasks
    from mechet.external_route_data import chemical_identity
    validate_public_tasks(validation_rows)
    valid_roots={chemical_identity(row['target']) for row in validation_rows}
    if len(validation_rows) != manifest['validation_targets']:
        raise ValueError('FULL_VALIDATION_DENOMINATOR_CHANGED')
    if (set(json.loads((dataset/'training_protected_identities.json').read_text())) != identities|valid_roots
            or set(json.loads((dataset/'validation_protected_identities.json').read_text())) != identities):
        raise ValueError('SPLIT_PROTECTED_IDENTITY_SETS_CHANGED')
    if manifest['training_targets'] < 1:
        raise ValueError('EMPTY_TRAINING_POPULATION')
    files = {str(dataset/'manifest.json'): file_sha(dataset/'manifest.json')}
    for name,expected in manifest['artifacts'].items():
        if Path(name).name != name or file_sha(dataset/name) != expected:
            raise ValueError('DATASET_ARTIFACT_HASH_CHANGED:'+name)
        files[str(dataset/name)] = expected
    checkpoint_path=parent/'pr96_checkpoint_manifest.json'
    if not checkpoint_path.exists():checkpoint_path=parent/'complete.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    components = checkpoint.get('files',checkpoint.get('payload_sha256',{}))
    for name in ('adapter_model.safetensors','adapter_config.json'):
        if name not in components or file_sha(parent/name) != components[name]:
            raise ValueError('WRONG_OR_CHANGED_ACTOR_ADAPTER:'+name)
        files[str(parent/name)] = components[name]
    files[str(checkpoint_path)] = file_sha(checkpoint_path)
    base = json.loads(base_manifest.read_text())
    if base['revision'] != OFFICIAL_REVISION:
        raise ValueError('WRONG_OFFICIAL_BASE_REVISION')
    files[str(base_manifest)] = file_sha(base_manifest)
    for name,record in base['files'].items():
        path = Path(base['snapshot'])/name
        if file_sha(path) != record['sha256']:
            raise ValueError('BASE_WEIGHT_HASH_CHANGED:'+name)
        files[str(path)] = record['sha256']
    rows = list(iter_records(dataset/'training.jsonl'))
    validate_public_tasks(rows)
    if len(rows) != manifest['training_targets']:
        raise ValueError('PUBLIC_TRAINING_DENOMINATOR_CHANGED')
    code_root = Path(__file__).resolve().parents[1]
    frozen_sources=source_files(code_root)
    from mechet.route_runtime_efficiency import PROFILE,kernel_config_bindings
    from mechet.route_process_reward import ALGORITHM as PROCESS_ALGORITHM, CONFIG as PROCESS_REWARD_CONFIG, VERSION as PROCESS_REWARD_VERSION
    from mechet.route_native_observation import NATIVE_SFT_ROUTE
    cfg = dict(algorithm=PROCESS_ALGORITHM,dataset=str(dataset),
        route_endpoint_contract='target_heavy_atom_contributors_v1',
        dataset_manifest_sha256=file_sha(dataset/'manifest.json'),train_targets=len(rows),
        validation_targets=manifest['validation_targets'],parent=str(parent),
        parent_sha256=components['adapter_model.safetensors'],base_path=base['snapshot'],
        base_manifest=str(base_manifest),output_dir=str(Path(output).resolve()),
        world_size=world_size,gpu_name=gpu_name,candidates=candidates,epochs=epochs,products_per_round=64,
        collective_timeout_seconds=604800,
        optimization_passes=1,episode_token_budget=None,max_action_tokens=2048,max_context=32768,
        generation_batch_size=64,loss_batch_size=1,executor_workers_per_rank=16,cpu_threads_per_rank=4,
        training_gradient_checkpointing=True,
        policy_precision='bf16_autocast',policy_allow_tf32=True,
        phase_profiling=True,
        training_max_padded_tokens=131072,
        executor_coalesce_seconds=.25,generation_qkv_channels=6144,
        generation_max_qkv_elements=2**31-1,generation_attention_budget_bytes=12*1024**3,
        fla_kernel_profile=PROFILE if gpu_name=='H20' else None,
        runtime_config_files_sha256=kernel_config_bindings(code_root) if gpu_name=='H20' else {},
        learning_rate=2e-6,clip_epsilon=.2,kl_beta=.02,gt_weight=0.,max_grad_norm=.5,
        loss_token_normalizer=1024.,behavior_logprob_tolerance=.25,partial_reward_weight=.2,
        behavior_probability_contract='actual_sampler_denominator_drift_diagnostic_v1',
        learned_critic=False,test_used=False,seed=17,
        adam_betas=[0.9,0.999],adam_eps=1e-8,weight_decay=0.0,
        advantage_estimator='centered_action_return_to_go_no_std',
        loss_type='dr_grpo',scale_rewards=False,
        trl_reference_commit='44809ba3f3a136f3d62859ee981fee2506bce5c9',
        implementation='native_route_sampler_and_TRL_equivalent_loss_kernel',
        validation_protocol=VALIDATION_PROTOCOL,
        files_sha256=files,code_files_sha256={name:file_sha(code_root/name) for name in frozen_sources},
        public_observation_contract=NATIVE_SFT_ROUTE,
        max_consecutive_recoveries=1,
        reward_contract=PROCESS_REWARD_VERSION,
        process_reward=PROCESS_REWARD_CONFIG,
        protected_evaluation_records=expected_protected_records,
        censoring='exclude_whole_K_group_and_report_selection_bias',
        gradient_estimand='mean_target_mean_independent_episode_full_action_token_sum_fixed_normalizer',
        replay='original_durable_behavior_tokens_only_no_teacher_actions_or_MCTS_winner_replay')
    if vllm_runtime is not None:
        from mechet.route_vllm_contract import configure_runtime
        configure_runtime(cfg,vllm_runtime)
    path = Path(contract_path)
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path,cfg)
    verify_contract(path)
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    from mechet.external_route_actor import FLOWER_STAGE2_PIN
    parser.add_argument('--parent',type=Path,default=FLOWER_STAGE2_PIN.checkpoint)
    parser.add_argument('--base-manifest',type=Path,default=ROOT/'artifacts/pr96_qwen35_base_manifest_20261022.json')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--world-size',type=int,default=4)
    parser.add_argument('--candidates',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--gpu-name',choices=('H20','A100'),default='H20')
    parser.add_argument('--vllm-runtime-json',type=Path)
    args=parser.parse_args()
    cfg=prepare_contract(args.dataset,args.parent,args.base_manifest,args.output,args.contract,
                         world_size=args.world_size,candidates=args.candidates,epochs=args.epochs,gpu_name=args.gpu_name,
                         vllm_runtime=json.loads(args.vllm_runtime_json.read_text()) if args.vllm_runtime_json else None)
    print(json.dumps(dict(contract=str(args.contract),train_targets=cfg['train_targets'],
                         parent_sha256=cfg['parent_sha256']),ensure_ascii=False))
