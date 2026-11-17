"""State-preserving checkpoint migration into an isolated accelerated condition.

No platform API is used. The controller's normal resume path loads the copied
adapter, optimizer and per-rank RNG bytes, then continues at the saved next_round.
The sampler change is explicit; this never claims the old random draws persist.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

from .external_route_data import file_sha,signature
from .route_training_contract import atomic_json

REQUIRED_PAYLOADS={'adapter_model.safetensors','adapter_config.json','optimizer.pt','rng.pt'}
# Every other field is immutable, including unrecognized future semantic fields.
MUTABLE_FIELDS={'output_dir','code_root','code_files_sha256','files_sha256','entry_archive',
    'entry_archive_sha256','source_commit','authorization','experimental_condition',
    'implementation','acceleration','loss_batch_size','initial_baseline','acceleration_continuation'}
COMPUTE_FILES={
    'src/mechet/route_acceleration.py','src/mechet/route_policy_sampling.py',
    'src/mechet/route_counter_sampling.py','src/mechet/route_execution_cache.py',
    'src/mechet/route_global_sampler.py','src/mechet/route_precision.py',
    'src/mechet/external_route_io.py','src/mechet/route_training_batches.py',
    'src/mechet/route_work_broker.py','src/mechet/route_distributed_collection.py',
    'src/mechet/route_async_collection.py','src/mechet/route_policy_episode.py',
    'src/mechet/route_refit_runtime.py',
    'src/mechet/route_refit_credit.py','src/mechet/route_policy_objective.py',
    'src/mechet/external_route_rl.py','scripts/train_external_route_rl.py'}


def read_json(path):
    return json.loads(Path(path).read_text())


def _schedule(cfg):
    from scripts.train_external_route_rl import training_order
    rows=[json.loads(line) for line in (Path(cfg['dataset'])/'training.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=cfg['train_targets']:raise ValueError('CONTINUATION_TRAIN_DENOMINATOR_CHANGED')
    ids=training_order(rows,cfg['seed'])
    return [(epoch,ids[i:i+cfg['products_per_round']]) for epoch in range(cfg['epochs'])
            for i in range(0,len(ids),cfg['products_per_round'])]


def _last_complete(cfg,contract_sha):
    root=Path(cfg['output_dir']).resolve()
    pointer=read_json(root/'latest.json')
    candidates=sorted(p for p in (root/'checkpoints').glob('round-*/complete.json')
                      if re.fullmatch(r'round-\d{6}',p.parent.name))
    if not candidates:raise ValueError('NO_FULLY_COMPLETED_CHECKPOINT')
    complete=candidates[-1];checkpoint=complete.parent
    if (Path(pointer['path']).resolve()!=checkpoint or pointer['complete_sha256']!=file_sha(complete)):
        raise ValueError('CONTINUATION_LATEST_POINTER_INCONSISTENT')
    if checkpoint.is_symlink() or complete.is_symlink():raise ValueError('CHECKPOINT_SYMLINK_FORBIDDEN')
    state=read_json(complete)
    if state.get('contract_sha256')!=contract_sha:raise ValueError('SOURCE_CONTRACT_CHANGED')
    payloads=state.get('payload_sha256',{})
    if not REQUIRED_PAYLOADS<=payloads.keys():raise ValueError('INCOMPLETE_CHECKPOINT_PAYLOADS')
    for name,digest in payloads.items():
        if Path(name).name!=name or (checkpoint/name).is_symlink():raise ValueError('UNSAFE_CHECKPOINT_PAYLOAD:'+name)
        if not (checkpoint/name).is_file() or file_sha(checkpoint/name)!=digest:
            raise ValueError('CONTINUATION_PAYLOAD_CHANGED:'+name)
    if (type(state.get('next_round')) is not int or state['next_round']<1
            or state['next_round']!=int(checkpoint.name.split('-')[1])
            or type(state.get('optimizer_step')) is not int
            or not 0<=state['optimizer_step']<=state['next_round']):
        raise ValueError('INVALID_COMPLETE_TRAINING_PROGRESS')
    if read_json(root/'latest.json')!=pointer:raise ValueError('SOURCE_ADVANCED_RECHECK')
    return checkpoint,state,pointer


def audit_source(contract_path,expected_sha256,*,verify_all_inputs=False):
    """Authenticate complete payloads, complete dataset and actual training order."""
    path=Path(contract_path).resolve()
    if file_sha(path)!=expected_sha256:raise ValueError('SOURCE_CONTRACT_CHANGED')
    cfg=read_json(path);dataset=Path(cfg['dataset'])
    if file_sha(dataset/'manifest.json')!=cfg['dataset_manifest_sha256']:
        raise ValueError('CONTINUATION_DATASET_MANIFEST_CHANGED')
    manifest=read_json(dataset/'manifest.json')
    for name,digest in manifest['artifacts'].items():
        if Path(name).name!=name or file_sha(dataset/name)!=digest:
            raise ValueError('CONTINUATION_DATASET_ARTIFACT_CHANGED:'+name)
    if (manifest['training_targets'],manifest['validation_targets'])!=(cfg['train_targets'],cfg['validation_targets']):
        raise ValueError('CONTINUATION_DATASET_DENOMINATOR_CHANGED')
    validation=[json.loads(line) for line in (dataset/'validation.jsonl').read_text().splitlines() if line.strip()]
    from scripts.train_external_route_rl import validate_public_tasks
    validate_public_tasks(validation)
    if len(validation)!=cfg['validation_targets']:raise ValueError('CONTINUATION_VALID_DENOMINATOR_CHANGED')
    if file_sha(Path(cfg['parent'])/'adapter_model.safetensors')!=cfg['parent_sha256']:
        raise ValueError('CONTINUATION_FIXED_REFERENCE_CHANGED')
    if verify_all_inputs:
        for name,digest in cfg['files_sha256'].items():
            if file_sha(name)!=digest:raise ValueError('SOURCE_FROZEN_INPUT_CHANGED:'+name)
    schedule=_schedule(cfg)
    checkpoint,state,pointer=_last_complete(cfg,expected_sha256)
    if state['training_order_sha256']!=signature(schedule):raise ValueError('CONTINUATION_TRAINING_ORDER_CHANGED')
    n=state['next_round']
    if n>len(schedule) or state['visited_target_episodes']!=sum(len(chunk) for _,chunk in schedule[:n]):
        raise ValueError('CONTINUATION_VISITED_PROGRESS_CHANGED')
    if state['epoch']!=schedule[n-1][0]:raise ValueError('CONTINUATION_EPOCH_CHANGED')
    if len(state.get('round_metrics',[]))!=cfg['world_size']:raise ValueError('INCOMPLETE_RANK_CHECKPOINT_RECEIPT')
    if any(m.get('optimizer_updated') is not True for m in state['round_metrics']):
        raise ValueError('INCOMPLETE_COLLECTIVE_OPTIMIZER_RECEIPT')
    return dict(source_contract=str(path),source_contract_sha256=expected_sha256,
        source_output=cfg['output_dir'],source_checkpoint=str(checkpoint),
        source_complete_sha256=file_sha(checkpoint/'complete.json'),source_latest=pointer,
        source_payload_sha256=state['payload_sha256'],source_state=state,
        next_round=n,optimizer_step=state['optimizer_step'],training_order_sha256=signature(schedule),
        total_rounds=len(schedule),train_targets=cfg['train_targets'],validation_targets=cfg['validation_targets'],
        dataset_manifest_sha256=cfg['dataset_manifest_sha256'],parent_sha256=cfg['parent_sha256'])


def assert_source_unchanged(audit):
    """Check publication race before and after copying, before new checkpoint publish."""
    if (file_sha(audit['source_contract'])!=audit['source_contract_sha256']
            or read_json(Path(audit['source_output'])/'latest.json')!=audit['source_latest']):
        raise ValueError('SOURCE_ADVANCED_RECHECK')
    try:
        checkpoint,state,pointer=_last_complete(read_json(audit['source_contract']),audit['source_contract_sha256'])
    except (ValueError,OSError,KeyError) as exc:
        raise ValueError('SOURCE_ADVANCED_RECHECK') from exc
    if (str(checkpoint)!=audit['source_checkpoint'] or file_sha(checkpoint/'complete.json')!=audit['source_complete_sha256']
            or state!=audit['source_state']):raise ValueError('SOURCE_ADVANCED_RECHECK')


def validate_conditions(original,new):
    for key in original.keys()|new.keys():
        if key not in MUTABLE_FIELDS and (original.get(key)!=new.get(key) or (key in original)!=(key in new)):
            raise ValueError('CONTINUATION_CONDITION_CHANGED:'+key)
    for path,digest in original['files_sha256'].items():
        if not Path(path).is_relative_to(Path(original['code_root'])) and path!=original.get('entry_archive'):
            if new['files_sha256'].get(path)!=digest:
                raise ValueError('CONTINUATION_ORIGINAL_INPUT_BINDING_CHANGED:'+path)
    baseline=deepcopy(original.get('initial_baseline'))
    if not baseline or baseline.get('kind')!='fresh_frozen_initial_actor':
        raise ValueError('CONTINUATION_REQUIRES_FRESH_FROZEN_BASELINE')
    baseline['parent_output']=new['output_dir']
    if new.get('initial_baseline')!=baseline:raise ValueError('CONTINUATION_CONDITION_CHANGED:initial_baseline')
    if Path(new['output_dir']).resolve()==Path(original['output_dir']).resolve():
        raise ValueError('CONTINUATION_OUTPUT_COLLISION')
    acceleration=new.get('acceleration',{})
    expected='skip_extra_duplicates_v1' if original.get('refit') else 'disabled'
    if acceleration.get('duplicate_suffix_policy')!=expected:
        raise ValueError('CONTINUATION_D1_D3_DUPLICATE_POLICY_CHANGED')
    # Never manufacture old VALID results under the changed sampler identity.
    # Current live D1/D3 handoffs are before their first25% full validation.
    schedule=_schedule(original)
    total=sum(len(c) for _,c in schedule);visited=0;prior=[]
    fractions=original['validation_protocol']['fractions'][1:]
    for index,(_,chunk) in enumerate(schedule,1):
        visited+=len(chunk)
        if any(visited>=f*total and visited-len(chunk)<f*total for f in fractions):prior.append(index)
    return prior


def seed_continuation(audit,new_cfg,new_contract_sha256,*,authorization):
    """Copy immutable bytes, preserve all state, then publish under new identity."""
    if not isinstance(authorization,str) or not authorization.strip():raise ValueError('EXPLICIT_USER_AUTHORIZATION_REQUIRED')
    original=read_json(audit['source_contract']);prior=validate_conditions(original,new_cfg)
    if any(n<audit['next_round'] for n in prior):
        raise ValueError('PRIOR_VALIDATION_TRANSFER_REQUIRES_EXPLICIT_PROVENANCE_SUPPORT')
    root=Path(new_cfg['output_dir'])
    if root.exists():raise FileExistsError(root)
    assert_source_unchanged(audit)
    from scripts.train_pr98_route_policy import publish_checkpoint
    checkpoint=Path(audit['source_checkpoint']);record=dict(
        version='pr111_stateful_acceleration_continuation_v1',
        source_contract=str(audit['source_contract']),source_contract_sha256=audit['source_contract_sha256'],
        source_checkpoint=str(checkpoint),source_complete_sha256=audit['source_complete_sha256'],
        authorization=authorization,changed_sampler_is_new_condition=True,
        rng='original per-rank saved bytes; changed sampler does not promise identical future draws',
        incomplete_collection='old pending collection retained at original output; recollect full next round')
    state=dict(audit['source_state'],contract_sha256=new_contract_sha256,acceleration_continuation=record)
    state.pop('payload_sha256')
    def writer(destination):
        for name,digest in audit['source_payload_sha256'].items():
            shutil.copyfile(checkpoint/name,destination/name)
            if file_sha(destination/name)!=digest:raise ValueError('CONTINUATION_COPY_HASH_CHANGED:'+name)
        assert_source_unchanged(audit)
    saved=publish_checkpoint(root,audit['next_round'],state,writer)
    # Keep exact original receipts alongside the new complete receipt; no relabeling.
    provenance=root/'continuation';provenance.mkdir()
    shutil.copyfile(checkpoint/'complete.json',provenance/'source_complete.json')
    shutil.copyfile(audit['source_contract'],provenance/'source_contract.json')
    atomic_json(provenance/'audit.json',audit)
    return saved


def validate_stop_receipt(path,instance,contract_sha256):
    record=read_json(path)
    if (record.get('instance_id')!=instance or record.get('source_contract_sha256')!=contract_sha256
            or record.get('status')!='END' or record.get('observed_at') is None):
        raise ValueError('VERIFIED_SOURCE_TASK_END_RECEIPT_REQUIRED')
    return record


def validate_admission(path,cfg,code_root,*,condition):
    """Bind T4 implementation checks to code/settings and authenticated old actor."""
    import math
    gate=read_json(path)
    if (gate.get('status')!='PASS' or 'T4' not in str(gate.get('hardware',''))
            or gate.get('precision')!='legacy_fp32_lora'
            or not gate.get('legal_checks') or any(v!='PASS' for v in gate['legal_checks'].values())):
        raise ValueError('T4_LEGALITY_ADMISSION_REQUIRED')
    compute=gate.get('validated_compute_files_sha256',{})
    if not COMPUTE_FILES<=compute.keys():raise ValueError('T4_COMPUTE_HASH_COVERAGE_INCOMPLETE')
    for name,digest in compute.items():
        if not Path(name).is_absolute() and '..' not in Path(name).parts:
            if file_sha(Path(code_root)/name)!=digest:raise ValueError('T4_ADMITTED_COMPUTE_CHANGED:'+name)
        else:raise ValueError('INVALID_T4_COMPUTE_PATH')
    benchmarks=gate.get('paired_benchmarks',{})
    if isinstance(benchmarks,list):
        selected=[item for item in benchmarks if item.get('task')==condition
            and item.get('candidate_kernel',cfg['acceleration']['sampling_kernel'])==cfg['acceleration']['sampling_kernel']]
    else:selected=[benchmarks.get(condition,{})]
    for item in selected:
        if 'generation' in item.get('stage',''):
            ratio=item.get('speedup')
            if not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or ratio<1/1.05:
                raise ValueError('T4_SELECTED_GENERATION_REGRESSION:'+condition)
    if not any(isinstance(item.get('speedup'),(int,float))
            and math.isfinite(item['speedup']) and item['speedup']>1 for item in selected):
        raise ValueError('T4_PAIRED_SPEEDUP_ADMISSION_REQUIRED:'+condition)
    admitted=gate.get('admitted_settings',{})
    acceleration=admitted.get('acceleration',admitted)
    for name,value in cfg['acceleration'].items():
        if name in acceleration and acceleration[name]!=value:raise ValueError('T4_ADMITTED_SETTINGS_CHANGED:'+name)
    if (acceleration.get('sampling_kernel')!=cfg['acceleration']['sampling_kernel']
            or admitted.get('loss_batch_size')!=cfg['loss_batch_size']):
        raise ValueError('T4_ADMITTED_SETTINGS_CHANGED')
    binding=gate.get('source_task_bindings',{}).get(condition,{})
    return gate,binding
