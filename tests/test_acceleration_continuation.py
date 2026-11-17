"""Checkpoint handoff preserves training state and refuses semantic/race changes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mechet.external_route_data import file_sha,signature
from mechet.route_training_contract import atomic_json
from scripts.train_pr98_route_policy import publish_checkpoint,verify_checkpoint


def fixture_source(tmp_path,*,refit=False):
    from scripts.train_external_route_rl import training_order
    dataset=tmp_path/'dataset';dataset.mkdir()
    rows=[dict(task_id=f't{i}',target='CC') for i in range(8)]
    for name,values in [('training.jsonl',rows),('validation.jsonl',rows[:2])]:
        (dataset/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
    for name in ['stock.txt','references.sqlite','training_protected_identities.json','validation_protected_identities.json']:
        (dataset/name).write_text('fixture '+name)
    artifacts={p.name:file_sha(p) for p in dataset.iterdir()}
    atomic_json(dataset/'manifest.json',dict(training_targets=8,validation_targets=2,artifacts=artifacts))
    output=tmp_path/'old';parent=tmp_path/'parent';parent.mkdir()
    (parent/'adapter_model.safetensors').write_bytes(b'initial')
    cfg=dict(dataset=str(dataset),dataset_manifest_sha256=file_sha(dataset/'manifest.json'),
        train_targets=8,validation_targets=2,seed=17,epochs=1,products_per_round=2,world_size=4,candidates=4,
        algorithm='external_target_only_route_process_grpo_v1',reward_contract='original',
        process_reward={'original':True},max_consecutive_recoveries=1 if refit else 8,
        parent=str(parent),parent_sha256=file_sha(parent/'adapter_model.safetensors'),
        output_dir=str(output),loss_batch_size=1,learning_rate=2e-6,
        initial_baseline=dict(timing='after_training',kind='fresh_frozen_initial_actor',
            parent_output=str(output),reference_actor_sha256=file_sha(parent/'adapter_model.safetensors')),
        validation_protocol=dict(fractions=[0.,1.]),files_sha256={str(dataset/'manifest.json'):file_sha(dataset/'manifest.json')},
        code_files_sha256={},code_root=str(tmp_path/'old_code'))
    if refit:cfg['refit']={'max_nodes_per_tree':8}
    contract=tmp_path/'original.json';atomic_json(contract,cfg)
    ids=training_order(rows,17);schedule=[(0,ids[i:i+2]) for i in range(0,8,2)]
    state=dict(contract_sha256=file_sha(contract),next_round=2,optimizer_step=2,
        training_order_sha256=signature(schedule),epoch=0,visited_target_episodes=4,
        round_metrics=[dict(optimizer_updated=True)]*4)
    def writer(path):
        for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','rng.pt'):
            (path/name).write_bytes(('saved '+name).encode())
    cp=publish_checkpoint(output,2,state,writer)
    return contract,cfg,cp


def new_config(cfg,tmp_path):
    new=deepcopy(cfg);new['output_dir']=str(tmp_path/'new');new['loss_batch_size']=4
    new['acceleration']={'sampling_kernel':'counter_categorical_v1',
        'duplicate_suffix_policy':'skip_extra_duplicates_v1' if cfg.get('refit') else 'disabled'}
    new['initial_baseline']['parent_output']=new['output_dir']
    return new


@pytest.mark.parametrize('refit',[False,True],ids=['D1','D3'])
def test_preserves_actor_optimizer_rng_step_and_training_order(tmp_path,refit):
    from mechet.route_acceleration_continuation import audit_source,seed_continuation
    contract,cfg,cp=fixture_source(tmp_path,refit=refit)
    audit=audit_source(contract,file_sha(contract));new=new_config(cfg,tmp_path)
    destination=tmp_path/'continuation.json';atomic_json(destination,new)
    seeded=seed_continuation(audit,new,file_sha(destination),authorization='explicit user request')
    state=verify_checkpoint(seeded,file_sha(destination));original=verify_checkpoint(cp,file_sha(contract))
    for key in ('next_round','optimizer_step','training_order_sha256','epoch','visited_target_episodes','round_metrics'):
        assert state[key]==original[key]
    assert state['payload_sha256']==original['payload_sha256']
    for name in state['payload_sha256']:
        assert (seeded/name).read_bytes()==(cp/name).read_bytes()
        assert (seeded/name).stat().st_ino!=(cp/name).stat().st_ino
    assert state['acceleration_continuation']['source_complete_sha256']==file_sha(cp/'complete.json')
    assert file_sha(contract)==audit['source_contract_sha256']


@pytest.mark.parametrize('field,value',[
    ('dataset','/changed'),('dataset_manifest_sha256','bad'),('reward_contract','different'),
    ('process_reward',{'different':True}),('seed',18),('products_per_round',4),
    ('max_consecutive_recoveries',1),('algorithm','standard_d'),('refit',{'added':True}),
    ('learning_rate',1e-5),('validation_targets',1),('parent_sha256','wrong')])
def test_rejects_changed_dataset_reward_recovery_order_or_rng_seed(tmp_path,field,value):
    from mechet.route_acceleration_continuation import audit_source,seed_continuation
    contract,cfg,cp=fixture_source(tmp_path);audit=audit_source(contract,file_sha(contract))
    new=new_config(cfg,tmp_path);new[field]=value
    with pytest.raises(ValueError,match='CONTINUATION_CONDITION_CHANGED:'+field):
        seed_continuation(audit,new,'new-contract',authorization='explicit')
    assert not Path(new['output_dir']).exists()


@pytest.mark.parametrize('name',['rng.pt','optimizer.pt','adapter_model.safetensors'])
def test_rejects_changed_complete_payload(tmp_path,name):
    from mechet.route_acceleration_continuation import audit_source
    contract,cfg,cp=fixture_source(tmp_path);(cp/name).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='PAYLOAD_CHANGED:'+name):audit_source(contract,file_sha(contract))


def test_incomplete_round_never_becomes_last_complete(tmp_path):
    from mechet.route_acceleration_continuation import audit_source
    contract,cfg,cp=fixture_source(tmp_path)
    incomplete=cp.parent/'round-000003.building-test';incomplete.mkdir()
    (incomplete/'rng.pt').write_bytes(b'incomplete')
    assert audit_source(contract,file_sha(contract))['next_round']==2


def test_rejects_stale_latest_pointer(tmp_path):
    from mechet.route_acceleration_continuation import audit_source
    contract,cfg,cp=fixture_source(tmp_path)
    atomic_json(Path(cfg['output_dir'])/'latest.json',dict(path=str(cp),complete_sha256='bad'))
    with pytest.raises(ValueError,match='LATEST_POINTER'):audit_source(contract,file_sha(contract))


def test_rejects_dataset_artifact_tamper(tmp_path):
    from mechet.route_acceleration_continuation import audit_source
    contract,cfg,cp=fixture_source(tmp_path)
    (Path(cfg['dataset'])/'validation.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='DATASET_ARTIFACT_CHANGED'):audit_source(contract,file_sha(contract))


def test_rejects_schedule_hash_change_even_when_receipt_hash_is_republished(tmp_path):
    from mechet.route_acceleration_continuation import audit_source
    contract,cfg,cp=fixture_source(tmp_path);s=json.loads((cp/'complete.json').read_text())
    s['training_order_sha256']='bad';atomic_json(cp/'complete.json',s)
    atomic_json(Path(cfg['output_dir'])/'latest.json',dict(path=str(cp),complete_sha256=file_sha(cp/'complete.json')))
    with pytest.raises(ValueError,match='TRAINING_ORDER_CHANGED'):audit_source(contract,file_sha(contract))


def test_rejects_source_advance_after_audit_before_copy(tmp_path):
    from mechet.route_acceleration_continuation import audit_source,seed_continuation
    contract,cfg,cp=fixture_source(tmp_path);audit=audit_source(contract,file_sha(contract))
    state=json.loads((cp/'complete.json').read_text());state.update(next_round=3,optimizer_step=3,visited_target_episodes=6)
    publish_checkpoint(Path(cfg['output_dir']),3,state,lambda p:[(p/n).write_bytes((cp/n).read_bytes()) for n in state['payload_sha256']])
    new=new_config(cfg,tmp_path)
    with pytest.raises(ValueError,match='SOURCE_ADVANCED_RECHECK'):
        seed_continuation(audit,new,'new',authorization='explicit')
    assert not Path(new['output_dir']).exists()


def test_rejects_source_advance_during_checkpoint_copy(tmp_path,monkeypatch):
    import mechet.route_acceleration_continuation as continuation
    contract,cfg,cp=fixture_source(tmp_path);audit=continuation.audit_source(contract,file_sha(contract))
    actual_copy=continuation.shutil.copyfile;changed=False
    def copy_and_advance(source,destination):
        nonlocal changed
        result=actual_copy(source,destination)
        if not changed:
            changed=True
            atomic_json(Path(cfg['output_dir'])/'latest.json',dict(path='changed',complete_sha256='bad'))
        return result
    monkeypatch.setattr(continuation.shutil,'copyfile',copy_and_advance)
    with pytest.raises(ValueError,match='SOURCE_ADVANCED_RECHECK'):
        continuation.seed_continuation(audit,new_config(cfg,tmp_path),'new',authorization='explicit')
    assert not (tmp_path/'new/latest.json').exists()


def test_missing_authorization_and_output_collision_refused(tmp_path):
    from mechet.route_acceleration_continuation import audit_source,seed_continuation
    contract,cfg,cp=fixture_source(tmp_path);audit=audit_source(contract,file_sha(contract));new=new_config(cfg,tmp_path)
    with pytest.raises(ValueError,match='EXPLICIT_USER_AUTHORIZATION_REQUIRED'):
        seed_continuation(audit,new,'new',authorization='')
    new['output_dir']=cfg['output_dir'];new['initial_baseline']['parent_output']=cfg['output_dir']
    with pytest.raises(ValueError,match='OUTPUT_COLLISION'):
        seed_continuation(audit,new,'new',authorization='explicit')


def test_stop_receipt_requires_matching_ended_source(tmp_path):
    from mechet.route_acceleration_continuation import validate_stop_receipt
    path=tmp_path/'stopped.json'
    atomic_json(path,dict(instance_id='instance',source_contract_sha256='source',status='RUNNING',observed_at='now'))
    with pytest.raises(ValueError,match='SOURCE_TASK_END'):validate_stop_receipt(path,'instance','source')
    atomic_json(path,dict(instance_id='instance',source_contract_sha256='source',status='END',observed_at='now'))
    assert validate_stop_receipt(path,'instance','source')['status']=='END'
    with pytest.raises(ValueError,match='SOURCE_TASK_END'):validate_stop_receipt(path,'other','source')


def test_t4_admission_rejects_changed_compute_or_sampler(tmp_path):
    from mechet.route_acceleration_continuation import COMPUTE_FILES,validate_admission
    for name in COMPUTE_FILES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    cfg=dict(acceleration=dict(sampling_kernel='private_pinned_argmax_v1'),loss_batch_size=4)
    gate=dict(status='PASS',hardware='Tesla T4',precision='legacy_fp32_lora',legal_checks={'probability':'PASS'},
        validated_compute_files_sha256={name:file_sha(tmp_path/name) for name in COMPUTE_FILES},
        paired_benchmarks={'d1':dict(speedup=1.2,baseline_seconds=12.,accelerated_seconds=10.)},
        admitted_settings=dict(sampling_kernel='private_pinned_argmax_v1',loss_batch_size=4),
        source_task_bindings={'d1':{'contract_sha256':'source','adapter_sha256':'actor'}})
    path=tmp_path/'admission.json';atomic_json(path,gate)
    assert validate_admission(path,cfg,tmp_path,condition='d1')[1]['adapter_sha256']=='actor'
    changed=deepcopy(cfg);changed['acceleration']['sampling_kernel']='counter_categorical_v1'
    with pytest.raises(ValueError,match='ADMITTED_SETTINGS_CHANGED'):validate_admission(path,changed,tmp_path,condition='d1')
    (tmp_path/next(iter(COMPUTE_FILES))).write_text('changed')
    with pytest.raises(ValueError,match='ADMITTED_COMPUTE_CHANGED'):validate_admission(path,cfg,tmp_path,condition='d1')


def test_t4_admission_accepts_stage_record_list_and_selected_kernel(tmp_path):
    from mechet.route_acceleration_continuation import COMPUTE_FILES,validate_admission
    for name in COMPUTE_FILES:
        file=tmp_path/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(name)
    cfg=dict(acceleration=dict(sampling_kernel='private_pinned_argmax_v1'),loss_batch_size=4)
    gate=dict(status='PASS',hardware='Tesla T4',precision='legacy_fp32_lora',legal_checks={'probability':'PASS'},
        validated_compute_files_sha256={name:file_sha(tmp_path/name) for name in COMPUTE_FILES},
        paired_benchmarks=[dict(task='d1',stage='generation',candidate_kernel='counter_categorical_v1',speedup=.8),
            dict(task='d1',stage='generation',candidate_kernel='private_pinned_argmax_v1',speedup=1.3,raw_seconds={'old':[13.],'new':[10.]})],
        admitted_settings=dict(sampling_kernel='private_pinned_argmax_v1',loss_batch_size=4),
        source_task_bindings={'d1':{'contract_sha256':'source','adapter_sha256':'actor'}})
    path=tmp_path/'admission.json';atomic_json(path,gate)
    assert validate_admission(path,cfg,tmp_path,condition='d1')[0]['status']=='PASS'


def test_new_contract_cannot_drop_original_dataset_hash_binding(tmp_path):
    from mechet.route_acceleration_continuation import audit_source,seed_continuation
    contract,cfg,cp=fixture_source(tmp_path);audit=audit_source(contract,file_sha(contract));new=new_config(cfg,tmp_path)
    new['files_sha256']={}
    with pytest.raises(ValueError,match='ORIGINAL_INPUT_BINDING_CHANGED'):
        seed_continuation(audit,new,'new',authorization='explicit')


def test_stopped_source_requires_all_original_rank_locks_are_released(tmp_path):
    import fcntl
    from scripts.prepare_acceleration_continuation import stopped_source_leases
    root=tmp_path/'old';root.mkdir()
    for rank in range(4):(root/f'.worker.rank{rank:02d}.lock').touch()
    cfg=dict(output_dir=str(root),world_size=4)
    with (root/'.worker.rank02.lock').open('r') as running:
        fcntl.flock(running,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='SOURCE_RANK_STILL_RUNNING:2'):
            with stopped_source_leases(cfg):raise AssertionError('must not reach migration')
    with stopped_source_leases(cfg):pass


def test_benchmark_actor_requires_bound_complete_receipt(tmp_path):
    from scripts.prepare_acceleration_continuation import _verified_benchmark_actor
    from mechet.route_acceleration_continuation import audit_source
    contract,cfg,cp=fixture_source(tmp_path);audit=audit_source(contract,file_sha(contract))
    binding=dict(contract_sha256=file_sha(contract),adapter_sha256=file_sha(cp/'adapter_model.safetensors'),
        checkpoint=str(cp),complete_sha256=file_sha(cp/'complete.json'))
    assert _verified_benchmark_actor(binding,audit)['checkpoint']==str(cp)
    binding['complete_sha256']='bad'
    with pytest.raises(ValueError,match='BENCHMARK_ACTOR_NOT_AUTHENTICATED'):_verified_benchmark_actor(binding,audit)


@pytest.mark.parametrize('generation_speedup,accepted',[(.996,True),(.90,False)])
def test_learner_gain_can_admit_flat_generation_but_rejects_regression(tmp_path,generation_speedup,accepted):
    from mechet.route_acceleration_continuation import COMPUTE_FILES,validate_admission
    for name in COMPUTE_FILES:
        file=tmp_path/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(name)
    cfg=dict(acceleration=dict(sampling_kernel='private_pinned_argmax_v1'),loss_batch_size=4)
    gate=dict(status='PASS',hardware='Tesla T4',precision='legacy_fp32_lora',legal_checks={'probability':'PASS'},
        validated_compute_files_sha256={name:file_sha(tmp_path/name) for name in COMPUTE_FILES},
        paired_benchmarks=[dict(task='d1',stage='same_prompt_same_complete_response_fixed_replay_generation',
            candidate_kernel='private_pinned_argmax_v1',speedup=generation_speedup,raw_seconds={'old':[10.],'new':[10./generation_speedup]}),
            dict(task='d1',stage='real_adapter_learner_forward_backward',candidate_kernel='private_pinned_argmax_v1',speedup=1.2,
                raw_seconds={'old':[12.],'new':[10.]})],
        admitted_settings=dict(sampling_kernel='private_pinned_argmax_v1',loss_batch_size=4),
        source_task_bindings={'d1':{'contract_sha256':'source','adapter_sha256':'actor'}})
    path=tmp_path/'admission.json';atomic_json(path,gate)
    if accepted:assert validate_admission(path,cfg,tmp_path,condition='d1')[0]['status']=='PASS'
    else:
        with pytest.raises(ValueError,match='GENERATION_REGRESSION'):validate_admission(path,cfg,tmp_path,condition='d1')
