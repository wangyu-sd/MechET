import json

import pytest


def failure_files(tmp_path):
    report=tmp_path/'failure.json';values=tmp_path/'native_outputs.json'
    report.write_text(json.dumps(dict(status='FAIL',source_bindings={'first':{'adapter_sha256':'actor'}})))
    rows=[dict(request_id=str(i),prompt_token_ids=[1,i+2],generated_token_ids=[4,9],
        behavior_logprobs=[-.1,-.2]) for i in range(5)]
    values.write_text(json.dumps(rows));return report,values


def test_every_failed_receipt_is_rechecked_at_b1_and_b4(tmp_path):
    from mechet.route_probability_admission import replay_failed_probability_receipts
    report,values=failure_files(tmp_path);seen=[]
    def score(model,pairs,pad,**kwargs):
        seen.extend(p[0][1] for p in pairs);return [[-.11,-.19] for _ in pairs]
    result=replay_failed_probability_receipts(report,values,None,0,'bf16_autocast',score,expected_actor='actor')
    assert seen==[2,3,4,5,6]*2
    assert result['status']=='PASS' and result['receipts']==5 and result['positions_per_pass']==10
    assert result['max_error_b1']==pytest.approx(.01)


@pytest.mark.parametrize('changed_actor,bad_score',[(True,False),(False,True)])
def test_replay_cannot_admit_wrong_actor_or_bad_old_receipt(tmp_path,changed_actor,bad_score):
    from mechet.route_probability_admission import replay_failed_probability_receipts
    report,values=failure_files(tmp_path)
    def score(model,pairs,pad,**kwargs):return [[-.1,-.6 if bad_score and p[0][1]==4 else -.2] for p in pairs]
    with pytest.raises(ValueError):
        replay_failed_probability_receipts(report,values,None,0,'bf16_autocast',score,
            expected_actor='wrong' if changed_actor else 'actor')


def test_teacher_witness_preserves_return_identity_and_restores_after_failure(tmp_path):
    from types import SimpleNamespace
    from mechet.route_probability_admission import ScoreWitnesses
    output=[[-.125,-.25]];actor=object()
    def original(*args,**kwargs):return output
    module=SimpleNamespace(token_logps_batch=original)
    with pytest.raises(RuntimeError):
        with ScoreWitnesses(module,tmp_path/'teacher.jsonl') as trace:
            assert module.token_logps_batch(actor,[([1],[2,9])],pad_token_id=77,precision='bf16_autocast') is output
            assert trace.primary_model is actor and trace.pad_token_id==77
            raise RuntimeError('diagnostic failure')
    assert module.token_logps_batch is original
    row=json.loads((tmp_path/'teacher.jsonl').read_text())
    assert row['teacher_logprobs']==output[0]


def test_hardware_failure_persists_scores_and_blocks_unchanged_automatic_retry(tmp_path,monkeypatch):
    import scripts.run_route_native_admission as entry
    import scripts.validate_pr111_native_runtime as native
    from scripts import pr96_earho_contract as probability
    from mechet.external_route_data import file_sha
    source=tmp_path/'source.json';source.write_text('{}')
    cfg=dict(acceleration={'version':'pr111_full_acceleration_v2'},output_dir=str(tmp_path/'out'),
        native_admission=dict(second_source_contract=str(source),second_source_contract_sha256=file_sha(source),
            source_contract_sha256=file_sha(source)),acceleration_continuation={'source_contract':str(source)})
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(cfg));calls=[]
    def original(*args,**kwargs):return [[-.125]]
    monkeypatch.setattr(probability,'token_logps_batch',original)
    def failed_run(args):
        calls.append(1);args.output.mkdir(parents=True)
        probability.token_logps_batch(object(),[([1],[2])],99,precision='bf16_autocast')
        (args.output/'admission.json').write_text(json.dumps({'status':'FAIL'}))
        raise ValueError('NATIVE_PROBABILITY_PARITY_UNCHANGED_025_GATE')
    monkeypatch.setattr(native,'run',failed_run)
    with pytest.raises(ValueError,match='PARITY'):entry.main(contract)
    validation=tmp_path/'out/runtime_validation'
    assert (validation/'native_admission_failure.json').exists()
    assert not (validation/'native_admission.json').exists()
    assert list(validation.glob('native-attempt-*/teacher_scores.jsonl'))
    assert probability.token_logps_batch is original
    with pytest.raises(ValueError,match='FROZEN_NATIVE_FAILURE_REQUIRES_REPAIR'):entry.main(contract)
    assert len(calls)==1


def test_failed_prior_replay_keeps_teacher_vectors(tmp_path,monkeypatch):
    import scripts.run_route_native_admission as entry
    import scripts.validate_pr111_native_runtime as native
    from scripts import pr96_earho_contract as probability
    from mechet.external_route_data import file_sha
    report,outputs=failure_files(tmp_path)
    source=tmp_path/'source.json';source.write_text('{}')
    previous={k:str(p) for k,p in [('report',report),('outputs',outputs)]}
    previous.update({k+'_sha256':file_sha(p) for k,p in [('report',report),('outputs',outputs)]})
    cfg=dict(acceleration={'version':'pr111_full_acceleration_v2'},output_dir=str(tmp_path/'out'),
        policy_precision='bf16_autocast',acceleration_continuation={'source_contract':str(source)},
        native_admission=dict(second_source_contract=str(source),second_source_contract_sha256=file_sha(source),
            source_contract_sha256=file_sha(source),actor_sha256='actor',previous_probability_failure=previous))
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(cfg))
    monkeypatch.setattr(probability,'token_logps_batch',lambda model,pairs,pad,**kw:[[-.1,-.6] for _ in pairs])
    def run(args):
        args.output.mkdir(parents=True)
        probability.token_logps_batch(object(),[([1],[2,9])],77,precision='bf16_autocast')
        (args.output/'admission.json').write_text(json.dumps({'status':'PASS'}))
    monkeypatch.setattr(native,'run',run)
    with pytest.raises(ValueError,match='PREVIOUS_FAILED_RECEIPT_PARITY'):entry.main(contract)
    receipts=list((tmp_path/'out/runtime_validation').glob('native-attempt-*/previous_teacher_scores.jsonl'))
    assert len(receipts)==1
    assert json.loads(receipts[0].read_text().splitlines()[0])['teacher_logprobs']==[-.1,-.6]
