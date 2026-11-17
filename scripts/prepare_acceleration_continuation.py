"""Audit a running source or prepare a stopped-source stateful continuation.

Never stops or submits jobs. `audit` is safe while the source runs. `prepare`
requires an END receipt, released rank leases and an admitted T4 implementation;
run it from a NEW dedicated code snapshot under ROOT/artifacts. Source snapshots,
old contracts and incomplete collection journals remain unchanged.
"""
import argparse
from contextlib import ExitStack,contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from mechet.external_route_data import file_sha
from mechet.route_acceleration_continuation import (audit_source,assert_source_unchanged,
    seed_continuation,validate_admission,validate_stop_receipt,read_json)
from mechet.route_training_contract import atomic_json
from scripts.prepare_external_route_rl import ROOT,source_files,verify_contract


def shared_new_path(value):
    path=Path(value).resolve()
    if not path.is_relative_to(Path('/aaa/fionafyang/buddy1')):raise ValueError('SHARED_CEPH_REQUIRED:'+str(path))
    if path.exists():raise FileExistsError(path)
    return path


@contextmanager
def stopped_source_leases(cfg):
    """Nonblocking read-only lock ownership prevents a late old worker/copy race."""
    import fcntl
    with ExitStack() as stack:
        for rank in range(cfg['world_size']):
            path=Path(cfg['output_dir'])/f'.worker.rank{rank:02d}.lock'
            handle=stack.enter_context(path.open('r'))
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise ValueError('SOURCE_RANK_STILL_RUNNING:'+str(rank)) from exc
        yield


def _verified_benchmark_actor(binding,audit):
    if binding.get('contract_sha256')!=audit['source_contract_sha256']:
        raise ValueError('T4_SOURCE_CONTRACT_BINDING_CHANGED')
    adapter=binding.get('adapter_sha256')
    root=Path(audit['source_output'])/'checkpoints'
    for receipt in sorted(root.glob('round-*/complete.json'),reverse=True):
        if '.building-' in receipt.parent.name:continue
        state=read_json(receipt)
        if (state.get('contract_sha256')==audit['source_contract_sha256']
                and state.get('payload_sha256',{}).get('adapter_model.safetensors')==adapter
                and (not binding.get('complete_sha256') or file_sha(receipt)==binding['complete_sha256'])
                and (not binding.get('checkpoint') or receipt.parent.resolve()==Path(binding['checkpoint']).resolve())):
            if file_sha(receipt.parent/'adapter_model.safetensors')==adapter:
                return dict(checkpoint=str(receipt.parent),complete_sha256=file_sha(receipt),adapter_sha256=adapter)
    raise ValueError('T4_BENCHMARK_ACTOR_NOT_AUTHENTICATED_SOURCE_CHECKPOINT')


def prepare(*,source_contract,source_contract_sha256,source_instance,source_stopped_receipt,
            admission,submission,output,source_commit,authorization,task_flag):
    if not authorization.strip():raise ValueError('EXPLICIT_USER_AUTHORIZATION_REQUIRED')
    code=Path(__file__).resolve().parents[1]
    if not code.is_relative_to(ROOT/'artifacts'):raise ValueError('DEDICATED_FROZEN_CODE_SNAPSHOT_REQUIRED')
    source_contract=Path(source_contract).resolve();source_stopped_receipt=Path(source_stopped_receipt).resolve()
    admission=Path(admission).resolve();submission=shared_new_path(submission);output=shared_new_path(output)
    if submission==output or submission.is_relative_to(output) or output.is_relative_to(submission):
        raise ValueError('CONTINUATION_SUBMISSION_OUTPUT_OVERLAP')
    archive=code.parent/'entry.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    original=read_json(source_contract)
    from mechet.external_route_actor import FLOWER_STAGE2_PIN
    if (original.get('parent_sha256')!=FLOWER_STAGE2_PIN.adapter_sha256
            or original.get('starting_actor_condition')!='corrected_flower_stage2_history'):
        raise ValueError('ORIGINAL_HISTORY89442_FIXED_REFERENCE_REQUIRED')
    stop=validate_stop_receipt(source_stopped_receipt,source_instance,source_contract_sha256)
    with stopped_source_leases(original):
        audit=audit_source(source_contract,source_contract_sha256,verify_all_inputs=True)
        condition='d3' if original.get('refit') else 'd1'
        cfg=deepcopy(original)
        from mechet.route_acceleration import DEFAULT_ACCELERATION,validate_acceleration
        gate=read_json(admission);settings=gate.get('admitted_settings',{})
        accel_settings=settings.get('acceleration',settings)
        acceleration=dict(DEFAULT_ACCELERATION)
        acceleration['sampling_kernel']=accel_settings.get('sampling_kernel')
        acceleration['duplicate_suffix_policy']='skip_extra_duplicates_v1' if original.get('refit') else 'disabled'
        cfg.update(output_dir=str(output),source_commit=source_commit,acceleration=acceleration,
            loss_batch_size=settings.get('loss_batch_size'),authorization=authorization,
            experimental_condition=f'pr111_{condition}_stateful_acceleration_continuation_v1',
            implementation=original['implementation']+'; explicit accelerated continuation')
        cfg['initial_baseline']['parent_output']=str(output)
        validate_acceleration(cfg)
        gate,binding=validate_admission(admission,cfg,code,condition=condition)
        benchmark=_verified_benchmark_actor(binding,audit)
        latest_legality=gate.get('latest_checkpoint_legality',{}).get(condition)
        if latest_legality is not None and (latest_legality.get('status')!='PASS'
                or latest_legality.get('adapter_sha256')!=audit['source_payload_sha256']['adapter_model.safetensors']):
            raise ValueError('LATEST_CHECKPOINT_LEGALITY_BINDING_CHANGED')
        cfg['acceleration_continuation']=dict(version='pr111_stateful_acceleration_continuation_v1',
            source_contract=str(source_contract),source_contract_sha256=source_contract_sha256,
            source_output=audit['source_output'],source_checkpoint=audit['source_checkpoint'],
            source_complete_sha256=audit['source_complete_sha256'],source_payload_sha256=audit['source_payload_sha256'],
            next_round=audit['next_round'],optimizer_step=audit['optimizer_step'],
            training_order_sha256=audit['training_order_sha256'],source_instance_id=source_instance,
            source_stopped_receipt=str(source_stopped_receipt),source_stopped_receipt_sha256=file_sha(source_stopped_receipt),
            admission=str(admission),admission_sha256=file_sha(admission),benchmark_actor=benchmark,
            authorization=authorization,original_condition=original.get('experimental_condition'),
            original_source_commit=original.get('source_commit'),sampler_equivalence='not_claimed_new_condition',
            runtime_precision=original.get('policy_precision'),t4_admission_precision=gate['precision'])
        old_code=Path(original['code_root']);old_archive=original.get('entry_archive')
        cfg['files_sha256']={path:digest for path,digest in original['files_sha256'].items()
            if not Path(path).is_relative_to(old_code) and path!=old_archive}
        cfg['code_root']=str(code)
        cfg['code_files_sha256']={name:file_sha(code/name) for name in source_files(code)}
        from mechet.route_runtime_efficiency import CONFIG_FILES
        names=source_files(code)+list(CONFIG_FILES)+['scripts/taiji_external_route_bootstrap.sh',
            'scripts/taiji_run_with_heartbeat.sh','configs/runtime/shared_host_storage.env']
        frozen={name:file_sha(code/name) for name in names}
        with tarfile.open(archive,'x:gz') as sink:
            for name in names:sink.add(code/name,arcname=name,recursive=False)
        for name,digest in frozen.items():
            if file_sha(code/name)!=digest:raise ValueError('SOURCE_CHANGED_DURING_ARCHIVE:'+name)
        cfg.update(entry_archive=str(archive),entry_archive_sha256=file_sha(archive))
        cfg['files_sha256'].update({str(code/name):digest for name,digest in frozen.items()})
        source_complete=Path(audit['source_checkpoint'])/'complete.json'
        for path in (source_contract,source_complete,source_stopped_receipt,admission,archive):
            cfg['files_sha256'][str(path)]=file_sha(path)
        # These immutable old paths authenticate original optimizer/RNG/actor.
        cfg['files_sha256'].update({str(Path(audit['source_checkpoint'])/name):digest
                                   for name,digest in audit['source_payload_sha256'].items()})
        assert_source_unchanged(audit)
        submission.mkdir(parents=True,mode=0o700)
        contract=submission/'contract.json';atomic_json(contract,cfg)
        verify_contract(contract)
        saved=seed_continuation(audit,cfg,file_sha(contract),authorization=authorization)
        from scripts.prepare_external_route_submission import task_payload
        payload=task_payload(cfg,code,contract,'20261117_'+condition+'_continuation')
        old_flag=payload['task_flag'];payload['task_flag']=task_flag
        payload['start_cmd']=payload['start_cmd'].replace(old_flag,task_flag)
        payload['readable_name']=f'meteor MechET {condition.upper()} accelerated stateful continuation (4H20)'
        payload['task_description']=(f'User-authorized {condition.upper()} acceleration continuation from fully completed '
            f'round{audit["next_round"]}, optimizer_step{audit["optimizer_step"]}; original complete TRAIN{cfg["train_targets"]}'
            f'/VALID{cfg["validation_targets"]}, reward/recovery/refit settings, seed17/full order, actor/optimizer/per-rank RNG'
            ' retained. Explicit changed sampler/compute admission, original outputs preserved. Native full VALID and final'
            ' TRAIN-only distillation remain required; aggregate maximum8 H20. No source pending partial round is credited.')
        # Reuse only the proven old private initializer/image; no secret printed.
        private_source=source_contract.parent/'task_config.private.json'
        working=read_json(private_source)
        if (working['host_num'],working['host_gpu_num'],working['GPUName'])!=(1,4,'H20'):
            raise ValueError('SOURCE_WORKING_PRIVATE_ALLOCATION_CHANGED')
        checked=subprocess.run(['bash','-n'],input=working['init_cmd'],text=True,capture_output=True)
        if checked.returncode:raise ValueError('SOURCE_PRIVATE_INITIALIZER_SHELL_SYNTAX_FAILED')
        payload.update(init_cmd=working['init_cmd'],image_full_name=working['image_full_name'])
        private=submission/'task_config.private.json';atomic_json(private,payload);private.chmod(0o600)
        atomic_json(submission/'task_config.json',dict(payload,init_cmd='<PRIVATE_INIT_REDACTED_NOT_SUBMITTABLE>'))
        result=dict(status='STATIC_PASS_NOT_SUBMITTED',condition=condition,task_flag=task_flag,
            source_instance_id=source_instance,source_contract_sha256=source_contract_sha256,
            contract_sha256=file_sha(contract),entry_archive_sha256=cfg['entry_archive_sha256'],
            source_complete_sha256=audit['source_complete_sha256'],seeded_checkpoint=str(saved),
            seeded_complete_sha256=file_sha(saved/'complete.json'),actor_sha256=audit['source_payload_sha256']['adapter_model.safetensors'],
            optimizer_step=audit['optimizer_step'],next_round=audit['next_round'],train_targets=cfg['train_targets'],
            validation_targets=cfg['validation_targets'],working_initializer_sha256=hashlib.sha256(payload['init_cmd'].encode()).hexdigest(),
            admission_sha256=file_sha(admission),source_stop_sha256=file_sha(source_stopped_receipt),
            output=str(output),runtime_precision=cfg.get('policy_precision'),benchmark_hardware=gate['hardware'],
            benchmark_precision=gate['precision'],experimental_results=False)
        atomic_json(submission/'static_verification.json',result)
        atomic_json(submission/'source_checkpoint_audit.json',audit)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    audit_parser=modes.add_parser('audit',help='Read-only checkpoint/data/order audit while source may run')
    prepare_parser=modes.add_parser('prepare',help='Prepare isolated continuation only after admission and source END')
    for mode in (audit_parser,prepare_parser):
        mode.add_argument('--source-contract',required=True,type=Path)
        mode.add_argument('--source-contract-sha256',required=True)
    audit_parser.add_argument('--audit-output',required=True,type=Path)
    audit_parser.add_argument('--verify-all-inputs',action='store_true')
    for name in ('source-stopped-receipt','admission','submission','output'):
        prepare_parser.add_argument('--'+name,required=True,type=Path)
    for name in ('source-instance','source-commit','authorization','task-flag'):
        prepare_parser.add_argument('--'+name,required=True)
    args=vars(parser.parse_args());mode=args.pop('mode')
    if mode=='audit':
        output=shared_new_path(args.pop('audit_output'))
        result=audit_source(args.pop('source_contract'),args.pop('source_contract_sha256'),**args)
        atomic_json(output,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_state','source_payload_sha256')}),flush=True)
    else:print(json.dumps(prepare(**args)),flush=True)


if __name__=='__main__':main()
