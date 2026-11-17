"""Freeze a full Standard B/C/D condition; does not submit or allocate GPUs.

Run from a dedicated immutable code snapshot, overlaid with reviewed changes.
All conditions start at the same corrected History-89442 and fixed reference.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tarfile
import subprocess

from mechet.external_route_data import file_sha
from mechet.external_route_actor import verify_actor_binding,FLOWER_STAGE2_PIN
from mechet.route_standard_contract import dataset_bindings
from mechet.route_process_reward import ALGORITHM,CONFIG,VERSION
from mechet.route_failure_policy import configure_first_rejection
from mechet.route_refit_runtime import DEFAULTS
from mechet.route_training_contract import atomic_json
from scripts.prepare_external_route_rl import source_files,verify_contract,ROOT


def prepare(*,template,template_sha256,dataset,dataset_manifest_sha256,data_verification,
            submission,output,condition,source_commit):
    if condition not in ('B','C','D'):raise ValueError('STANDARD_CONDITION_REQUIRED')
    template=Path(template).resolve();submission=Path(submission).resolve();output=Path(output).resolve()
    code=Path(__file__).resolve().parents[1]
    if not code.is_relative_to(ROOT/'artifacts'):raise ValueError('DEDICATED_FROZEN_CODE_SNAPSHOT_REQUIRED')
    for path in (submission,output):
        if not path.is_relative_to(Path('/aaa/fionafyang/buddy1')):raise ValueError('CEPH_REQUIRED')
        if path.exists():raise FileExistsError(path)
    if file_sha(template)!=template_sha256:raise ValueError('STANDARD_TEMPLATE_HASH_CHANGED')
    cfg=json.loads(template.read_text());verify_actor_binding(cfg)
    if cfg['parent_sha256']!=FLOWER_STAGE2_PIN.adapter_sha256:raise ValueError('STANDARD_STARTING_ACTOR_CHANGED')
    bindings=dataset_bindings(dataset,dataset_manifest_sha256)
    verification=Path(data_verification).resolve();audit=json.loads(verification.read_text())
    evidence=audit.get(Path(dataset).name,{})
    if evidence.get('status')!='PASS' or evidence.get('manifest_sha256')!=dataset_manifest_sha256:
        raise ValueError('COMPLETE_INDEPENDENT_SOURCE_VERIFICATION_REQUIRED')
    old_data=Path(cfg['dataset']);old_code=Path(cfg['code_root']);old_archive=cfg.get('entry_archive')
    retained={p:h for p,h in cfg['files_sha256'].items()
              if not Path(p).is_relative_to(old_data) and not Path(p).is_relative_to(old_code)
              and p!=old_archive}
    cfg.update(bindings);cfg['files_sha256'].update(retained)
    for name in ('preflight_software_evidence','refit','flow_credit_alpha','route_memory','vllm_runtime'):
        cfg.pop(name,None)
    cfg.update(algorithm=ALGORITHM,process_reward=deepcopy(CONFIG),reward_contract=VERSION,
        public_observation_contract='route_sft_aligned_feedback_v9',max_consecutive_recoveries=8,
        advantage_estimator='centered_action_return_to_go_no_std',inference_backend='hf',
        output_dir=str(output),source_commit=source_commit,world_size=4,gpu_name='H20',
        experimental_condition=f'pr109_standard_D3_or_D2_{condition}_complete_source_rows',
        authorization='2026-11-16 user requests execution of PR109; complete Standard protocol, aggregate max8 H20',
        protected_evaluation_records=0,training_examples_unit='source_route_row_target_occurrence',
        standard_external_teacher=False,standard_extra_forward_model=False)
    cfg['stock_material_contract']='exact_material_stock_v1'
    cfg.pop('route_failure_policy',None)
    if condition in ('C','D'):
        configure_first_rejection(cfg);cfg['flow_credit_alpha']=.5
    if condition=='D':cfg['refit']=dict(DEFAULTS)
    if condition=='D':
        cfg.update(censoring='exclude_whole_original_G4_and_whole_conditional_node_independently',
            gradient_estimand='mean_source_row_G4_fixedC1024_unique_receipt_branch_substitution')
    cfg['implementation']='native_process_DrGRPO_with_optional_exact_state_refit_v1'
    cfg['standard_data']['independent_verification_sha256']=file_sha(verification)
    cfg['files_sha256'][str(verification)]=file_sha(verification)
    cfg['initial_baseline']=dict(timing='after_training',kind='fresh_frozen_initial_actor',
        parent_output=str(output),reference_actor_sha256=cfg['parent_sha256'])
    cfg['code_root']=str(code);cfg['code_files_sha256']={n:file_sha(code/n) for n in source_files(code)}
    names=source_files(code)+['scripts/taiji_external_route_bootstrap.sh',
        'scripts/taiji_run_with_heartbeat.sh','configs/runtime/shared_host_storage.env']
    from mechet.route_runtime_efficiency import CONFIG_FILES
    names+=list(CONFIG_FILES)
    archive=code.parent/'entry.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz') as sink:
        for name in names:sink.add(code/name,arcname=name,recursive=False)
    cfg.update(entry_archive=str(archive),entry_archive_sha256=file_sha(archive))
    cfg['files_sha256'][str(archive)]=cfg['entry_archive_sha256']
    for name in names:cfg['files_sha256'][str(code/name)]=file_sha(code/name)
    submission.mkdir(parents=True,mode=0o700)
    path=submission/'contract.json';atomic_json(path,cfg);verify_contract(path)
    from scripts.prepare_external_route_submission import task_payload
    payload=task_payload(cfg,code,path,'20261116_pr109_'+condition.lower())
    old_flag=payload['task_flag']
    payload.update(task_flag='meteor_mechet_pr109_standard_'+condition.lower()+'_4h20_20261116',
        readable_name=f'MechET PR109 Standard {condition} complete source TRAIN/VALID (4H20)',
        task_description=f'PR109 teacher-free Standard {condition}; complete source-row TRAIN {cfg["train_targets"]}, VALID {cfg["validation_targets"]}; corrected History89442 start and fixed reference; declared source overlap and project URSA training-stock condition; native DrGRPO C1024; first rejection/rollback and conserved FLOW credit only C/D; exact-state branching only D. No external teacher, no hidden route action labels, no aggregate episode cap. Full frozen independent VALID G4 selection. Aggregate allocation ceiling8 H20; existing frozen jobs unchanged.')
    payload['start_cmd']=payload['start_cmd'].replace(old_flag,payload['task_flag'])
    # Reuse the known working mount/image initializer, never expose it in the
    # public packet or logs. Preparation does not execute that initializer.
    launch=template.parent/'task_config.private.json'
    working=json.loads(launch.read_text())
    if (working['host_num']!=1 or working['host_gpu_num']!=4 or working['GPUName']!='H20'):
        raise ValueError('WORKING_LAUNCH_ALLOCATION_CHANGED')
    payload.update(init_cmd=working['init_cmd'],image_full_name=working['image_full_name'])
    checked=subprocess.run(['bash','-n'],input=payload['init_cmd'],text=True,capture_output=True)
    if checked.returncode:raise ValueError('PRIVATE_INIT_SHELL_SYNTAX_FAILED')
    private=submission/'task_config.private.json';atomic_json(private,payload);private.chmod(0o600)
    atomic_json(submission/'task_config.json',dict(payload,init_cmd='<PRIVATE_INIT_REDACTED_NOT_SUBMITTABLE>'))
    result=dict(status='STATIC_PASS_NOT_SUBMITTED',condition=condition,contract_sha256=file_sha(path),
        train_targets=cfg['train_targets'],validation_targets=cfg['validation_targets'],
        parent_sha256=cfg['parent_sha256'],world_size=4,output=str(output))
    import hashlib
    result['working_initializer_sha256']=hashlib.sha256(payload['init_cmd'].encode()).hexdigest()
    atomic_json(submission/'static_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('template','dataset','data-verification','submission','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    for name in ('template-sha256','dataset-manifest-sha256','source-commit'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--condition',required=True,choices=('B','C','D'))
    print(json.dumps(prepare(**vars(parser.parse_args()))),flush=True)
