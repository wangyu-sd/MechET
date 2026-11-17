from copy import deepcopy
import json

import pytest


def test_native_training_refuses_missing_failed_or_stale_hardware_gate(tmp_path):
    from mechet.route_native_admission import require_native_admission
    cfg=dict(output_dir=str(tmp_path),inference_backend='vllm',policy_precision='bf16_autocast',
        acceleration={'version':'pr111_full_acceleration_v2'},
        native_admission=dict(required=True,actor_sha256='actor',source_contract_sha256='old'))
    with pytest.raises(ValueError,match='NATIVE_HARDWARE_ADMISSION_REQUIRED'):
        require_native_admission(cfg,'contract')
    p=tmp_path/'runtime_validation/native_admission.json';p.parent.mkdir()
    report=tmp_path/'report.json'
    value=dict(status='PASS',hardware='NVIDIA H20',hardware_profile='h20_bf16',precision='bf16_autocast',
        candidate_contract_sha256='contract',source_bindings={'first':dict(adapter_sha256='actor',contract_sha256='old')},
        legal_checks={'native_probability':'PASS'},validated_compute_files_sha256={})
    from mechet.external_route_data import file_sha
    for changed in [dict(status='FAIL'),dict(hardware_profile='t4_fp16'),dict(candidate_contract_sha256='wrong'),
                    dict(precision='legacy_fp32_lora'),dict(legal_checks={})]:
        report.write_text(json.dumps(dict(value,**changed)))
        p.write_text(json.dumps(dict(report=str(report),sha256=file_sha(report))))
        with pytest.raises(ValueError):require_native_admission(cfg,'contract')


def test_full_native_continuation_keeps_semantics_and_explicit_backend_change(tmp_path):
    from test_acceleration_continuation import fixture_source,new_config
    from mechet.route_acceleration_continuation import validate_conditions
    from mechet.route_acceleration import full_acceleration
    _,old,_=fixture_source(tmp_path,refit=True)
    new=new_config(old,tmp_path)
    new.update(acceleration=full_acceleration(refit=True),inference_backend='vllm',vllm_runtime={'frozen':True},
        native_admission={'required':True})
    validate_conditions(old,new)
    bad=deepcopy(new);bad['max_consecutive_recoveries']=8
    with pytest.raises(ValueError,match='CONDITION_CHANGED:max_consecutive_recoveries'):validate_conditions(old,bad)
    bad=deepcopy(new);bad['acceleration']['version']='old'
    with pytest.raises(ValueError):validate_conditions(old,bad)


def test_real_hardware_receipt_is_required_with_exact_compute_actor_and_contract(tmp_path):
    from mechet.route_native_admission import require_native_admission,NATIVE_FILES,REQUIRED_CHECKS
    from mechet.route_acceleration_continuation import COMPUTE_FILES
    from mechet.external_route_data import file_sha
    code=tmp_path/'code'
    for name in COMPUTE_FILES|NATIVE_FILES:
        p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    cfg=dict(output_dir=str(tmp_path),policy_precision='bf16_autocast',
        acceleration={'version':'pr111_full_acceleration_v2'},native_admission=dict(required=True,
            actor_sha256='actor',source_contract_sha256='old',source_complete_sha256='complete'))
    report=tmp_path/'report.json'
    value=dict(status='PASS',hardware='NVIDIA H20',hardware_profile='h20_bf16',precision='bf16_autocast',
        candidate_contract_sha256='contract',source_bindings={'first':dict(adapter_sha256='actor',
            contract_sha256='old',complete_sha256='complete')},legal_checks={k:'PASS' for k in REQUIRED_CHECKS},
        validated_compute_files_sha256={name:file_sha(code/name) for name in COMPUTE_FILES|NATIVE_FILES})
    report.write_text(json.dumps(value))
    pointer=tmp_path/'runtime_validation/native_admission.json';pointer.parent.mkdir()
    pointer.write_text(json.dumps(dict(report=str(report),sha256=file_sha(report))))
    assert require_native_admission(cfg,'contract',code=code)['status']=='PASS'
    (code/'src/mechet/route_work_broker.py').write_text('changed')
    with pytest.raises(ValueError,match='ADMITTED_COMPUTE_CHANGED'):
        require_native_admission(cfg,'contract',code=code)


def test_native_preparation_carries_forward_the_bound_nvml_runtime_leaf(tmp_path):
    from scripts.prepare_acceleration_continuation import preserve_runtime_leaves
    from mechet.external_route_data import file_sha
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir()
    relative='scripts/_nvml_runtime/pynvml.py';source=old/relative;source.parent.mkdir(parents=True)
    source.write_text('bound NVIDIA dependency')
    cfg=dict(code_root=str(old),files_sha256={str(source):file_sha(source)})
    preserve_runtime_leaves(cfg,new)
    assert (new/relative).read_bytes()==source.read_bytes()
    source.write_text('changed')
    with pytest.raises(ValueError,match='RUNTIME_LEAF_CHANGED'):preserve_runtime_leaves(cfg,new)
