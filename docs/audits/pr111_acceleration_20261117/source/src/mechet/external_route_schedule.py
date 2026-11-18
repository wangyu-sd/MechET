"""TRAIN-first scheduling with a provenance-preserving frozen baseline."""
import ast
import json
from pathlib import Path

from .external_route_data import file_sha,signature

COMPATIBLE_FUNCTIONS=('validate_public_tasks','make_episodes','score_groups',
                      'collect_round','backend_collection_metrics','full_validation')
CONTROLLER_FILES={'scripts/train_external_route_rl.py','scripts/prepare_external_route_rl.py',
                  'scripts/prepare_external_route_submission.py'}
CONDITION_FIELDS=('algorithm','dataset','dataset_manifest_sha256','parent','parent_sha256','base_path',
    'world_size','gpu_name','candidates','seed','max_action_tokens','max_context',
    'generation_batch_size','executor_workers_per_rank','products_per_round','partial_reward_weight',
    'training_gradient_checkpointing','loss_batch_size','training_max_padded_tokens',
    'inference_backend','vllm_runtime','route_memory','policy_protocol','policy_model_family','acceleration',
    'generation_qkv_channels','generation_max_qkv_elements','generation_attention_budget_bytes',
    'public_observation_contract','max_consecutive_recoveries','route_endpoint_contract',
    'executor_coalesce_seconds','fla_kernel_profile','runtime_config_files_sha256',
    'transformers_import_index',
    'validation_protocol','validation_targets','train_targets','epochs',
    'learning_rate','clip_epsilon','kl_beta','loss_token_normalizer','max_grad_norm',
    'adam_betas','adam_eps','weight_decay','runtime_root','runtime_archive_sha256')


def deferred(cfg):
    return cfg.get('initial_baseline',{}).get('timing')=='after_training'


def validation_due(cfg,boundaries,published_round):
    return published_round in boundaries and not (published_round==0 and deferred(cfg))


def required_prior_validations(cfg,boundaries,start):
    return [n for n in boundaries if n<start and validation_due(cfg,boundaries,n)]


def _canonical_ast(value):
    # Python3.13 changed ast.dump's default empty-field formatting and added
    # empty type_params fields. Preserve every populated AST field, with an
    # explicit node type and literal repr, independently of those defaults.
    if isinstance(value,ast.AST):
        return [type(value).__name__,[(name,_canonical_ast(item))
            for name,item in ast.iter_fields(value) if item is not None and item!=[]]]
    if isinstance(value,list):return [_canonical_ast(item) for item in value]
    return repr(value)


def function_signatures(path):
    nodes={n.name:n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    if not set(COMPATIBLE_FUNCTIONS)<=nodes.keys():
        raise ValueError('BASELINE_EVALUATOR_FUNCTION_MISSING')
    return {n:signature(_canonical_ast(nodes[n])) for n in COMPATIBLE_FUNCTIONS}


def _compatible(cfg,parent,code_root):
    if parent.get('initial_baseline'):
        raise ValueError('NESTED_BASELINE_CONTINUATION_FORBIDDEN')
    for key in CONDITION_FIELDS:
        if parent.get(key)!=cfg.get(key):raise ValueError('BASELINE_CONDITION_CHANGED:'+key)
    if Path(parent['output_dir']).resolve()==Path(cfg['output_dir']).resolve():
        raise ValueError('BASELINE_AND_TRAIN_OUTPUT_COLLISION')
    for name,digest in parent['code_files_sha256'].items():
        if name not in CONTROLLER_FILES and cfg['code_files_sha256'].get(name)!=digest:
            raise ValueError('BASELINE_SAMPLING_CHANGED:'+name)
    old_script=Path(parent['code_root'])/'scripts/train_external_route_rl.py'
    if file_sha(old_script)!=parent['code_files_sha256']['scripts/train_external_route_rl.py']:
        raise ValueError('BASELINE_ORIGINAL_CONTROLLER_HASH_CHANGED')
    old_functions=function_signatures(old_script)
    if old_functions!=function_signatures(Path(code_root)/'scripts/train_external_route_rl.py'):
        raise ValueError('BASELINE_EVALUATOR_CHANGED')
    return old_script,old_functions


def configure_deferred_baseline(cfg,parent_contract,code_root):
    path=Path(parent_contract).resolve();parent=json.loads(path.read_text())
    script,functions=_compatible(cfg,parent,code_root)
    cfg['initial_baseline']=dict(timing='after_training',parent_contract=str(path),
        parent_contract_sha256=file_sha(path),parent_output=parent['output_dir'],
        compatible_functions_sha256=functions,
        provenance='original durable baseline receipts; new controller recorded separately')
    cfg['files_sha256'][str(path)]=file_sha(path)
    cfg['files_sha256'][str(script)]=file_sha(script)


def configure_fresh_deferred_baseline(cfg):
    cfg['initial_baseline']=dict(timing='after_training',kind='fresh_frozen_initial_actor',
        parent_output=cfg['output_dir'],reference_actor_sha256=cfg['parent_sha256'])


def validate_deferred_baseline(cfg,code_root):
    record=cfg.get('initial_baseline')
    if record is None:return None
    if record.get('timing')!='after_training':raise ValueError('INVALID_BASELINE_TIMING')
    if record.get('kind')=='fresh_frozen_initial_actor':
        if ('parent_contract' in record or record.get('parent_output')!=cfg['output_dir'] or
                record.get('reference_actor_sha256')!=cfg['parent_sha256']):
            raise ValueError('FRESH_BASELINE_BINDING_CHANGED')
        return dict(cfg)
    path=Path(record['parent_contract'])
    if file_sha(path)!=record['parent_contract_sha256']:
        raise ValueError('BASELINE_ORIGINAL_CONTRACT_HASH_CHANGED')
    parent=json.loads(path.read_text());_,functions=_compatible(cfg,parent,code_root)
    if functions!=record['compatible_functions_sha256'] or parent['output_dir']!=record['parent_output']:
        raise ValueError('BASELINE_PROVENANCE_CHANGED')
    parent['_sha']=record['parent_contract_sha256']
    return parent


def evaluate_frozen_baseline(cfg,code_root,rows,stock,protected,reference,tokenizer,evaluator):
    # Caller owns every old rank lease. Never replace the receipt's real
    # original contract/behavior IDs with the new TRAIN controller's identity.
    parent=validate_deferred_baseline(cfg,code_root)
    return evaluator(rows,stock,protected,reference,tokenizer,parent,
                     Path(parent['output_dir']),0,cfg['parent_sha256'],[0.])


def baseline_for_selection(cfg,summary):
    record=cfg['initial_baseline']
    expected=(cfg['_sha'] if record.get('kind')=='fresh_frozen_initial_actor'
              else record['parent_contract_sha256'])
    if (not summary['complete'] or summary['denominator']!=cfg['validation_targets']
            or summary['checkpoint_round']!=0 or summary['actor_sha256']!=cfg['parent_sha256']
            or summary['binding']['contract_sha256']!=expected):
        raise ValueError('INCOMPLETE_OR_WRONG_BASELINE')
    result=dict(summary,source_binding=dict(summary['binding']),
        binding=dict(summary['binding'],contract_sha256=cfg['_sha']),
        evaluation_timing='after_training_using_frozen_initial_actor')
    return result
