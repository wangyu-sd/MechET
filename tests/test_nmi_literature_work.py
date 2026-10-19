"""Measured run-ledger work is distinct from per-report/cache accounting."""
import hashlib
import importlib
import json

import pytest


def ledger(path,rows):
    path.write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows))


def wave(i,tokens,solved,added):
    return {'wave':i,'output_tokens':tokens,'model_forward_calls':i*2,'prompt_tokens':i*20,
        'policy_request_batches':i,'cpu_executor_thread_seconds':i*.5,
        'shared_policy_cache_hits':i,'reused_output_tokens':i*100,
        'inference_elapsed_seconds':i*3.,'new_stock_solved_ids':added,
        'credited_stock_solved_targets':solved,'boundary':'post_executor_whole_generation_wave_v1'}


def implementation():
    try:return importlib.import_module('mechet.analysis.literature_work')
    except ModuleNotFoundError:pytest.fail('frozen-generation work auditor missing')


def test_work_prefix_uses_exact_frozen_boundary_and_does_not_charge_cache_tokens(tmp_path):
    path=tmp_path/'waves.jsonl';prefix=[wave(0,0,0,[]),wave(1,10,1,['a'])]
    # Actual runners publish wave zero after setup: its timer is already >0,
    # while all generation/executor counters are still exactly zero.
    prefix[0]['inference_elapsed_seconds']=.25
    ledger(path,prefix+[wave(2,30,2,['b'])])
    result=implementation().snapshot_wave_prefix(path,output_tokens=10,solved_ids={'a'})
    assert result['actual_generated_output_tokens']==10
    assert result['reused_output_tokens_not_charged']==100
    assert result['unique_scheduled_model_forward_calls']==2
    assert result['logical_prompt_tokens_not_uncached_prefill']==20
    assert result['frozen_stock_solved_ids']==['a']
    assert result['actual_gpu_active_seconds'] is None
    expected=b''.join((json.dumps(r)+'\n').encode() for r in prefix)
    assert result['source_prefix_sha256']==hashlib.sha256(expected).hexdigest()
    assert result['source_prefix_bytes']==len(expected)
    ledger(path,prefix+[wave(2,40,2,['c'])])
    again=implementation().snapshot_wave_prefix(path,output_tokens=10,solved_ids={'a'})
    assert again==result


def test_work_prefix_unchanged_by_future_cache_only_solution(tmp_path):
    path=tmp_path/'waves.jsonl'
    prefix=[wave(0,0,0,[]),wave(1,10,1,['a'])]
    ledger(path,prefix)
    before=implementation().snapshot_wave_prefix(path,output_tokens=10,solved_ids={'a'})
    later=wave(2,10,2,['b'])
    later['model_forward_calls']=2
    later['prompt_tokens']=20
    ledger(path,prefix+[later])
    after=implementation().snapshot_wave_prefix(path,output_tokens=10,solved_ids={'a'})
    assert after==before
    assert after['last_whole_wave']==1
    assert after['source_prefix_bytes']==len(b''.join((json.dumps(r)+'\n').encode() for r in prefix))


@pytest.mark.parametrize('mutation',['missing_boundary','phantom_solved','duplicate_credit',
    'regressing_counter','ambiguous_boundary','truncated_boundary','wrong_contract'])
def test_work_prefix_refuses_invented_or_ambiguous_cost_alignment(tmp_path,mutation):
    path=tmp_path/'waves.jsonl';data=[wave(0,0,0,[]),wave(1,10,1,['a']),wave(2,30,2,['b'])]
    if mutation=='missing_boundary':data[1]['output_tokens']=11
    if mutation=='phantom_solved':data[1]['new_stock_solved_ids']=['x']
    if mutation=='duplicate_credit':data[1]['new_stock_solved_ids']=['a','a']
    if mutation=='regressing_counter':data[1]['prompt_tokens']=-1
    if mutation=='ambiguous_boundary':
        data[2]['output_tokens']=10
        data[2]['new_stock_solved_ids']=[]
        data[2]['credited_stock_solved_targets']=1
    if mutation=='wrong_contract':data[1]['boundary']='unknown'
    ledger(path,data)
    if mutation=='truncated_boundary':path.write_bytes(path.read_bytes().splitlines(keepends=True)[0]+b'{"wave":1')
    with pytest.raises(ValueError):implementation().snapshot_wave_prefix(path,output_tokens=10,solved_ids={'a'})


def frozen_work_files(tmp_path):
    import gzip
    from mechet.analysis.literature_sources import file_sha256
    rank=tmp_path/'rank00';rank.mkdir()
    ledger(rank/'work_waves.jsonl',[wave(0,0,0,[]),wave(1,10,1,['a']),wave(2,30,2,['b'])])
    contract=rank/'manifest.json';contract.write_text(json.dumps({'adapter_sha':'b'*64,'rank':0,'world':1,'shard_rows':2}))
    snapshot=tmp_path/'snapshot.json';snapshot.write_text(json.dumps({'r1':{
        'full_denominator':2,'saved_reports':1,'nonzero_stock_connected':1,'unfinished':1},
        'sources':{'r1_progress':[{'rank':0,'path':str(rank/'progress.json'),'output_tokens':10,'stock_solved_targets':1}]}}))
    freeze=tmp_path/'freeze';freeze.mkdir()
    for name,records in [('frozen_routes',[{'run_id':'r1_mcts_pr88','target_id':'a','reaction_keys':['reaction'],
            'stock_verified':True,'already_in_stock':False}]),('target_population',[
            {'run_id':'r1_mcts_pr88','target_id':'a','report_status':'observed'},
            {'run_id':'r1_mcts_pr88','target_id':'b','report_status':'pending'}])]:
        with gzip.open(freeze/(name+'.jsonl.gz'),'wt') as f:
            for r in records:f.write(json.dumps(r)+'\n')
    value={'complete':True,'actor_sha256':'b'*64,'progress_snapshot_sha256':file_sha256(snapshot),
        'source_reports':[{'run_id':'r1_mcts_pr88','target_id':'a','report_path':str(rank/'evaluations/a.json'),
            'manifest_path':str(contract),'manifest_sha256':file_sha256(contract),'actor_sha256':'b'*64}],
        'output_files':{p.name:{'sha256':file_sha256(p)} for p in freeze.glob('*.gz')}}
    (freeze/'prediction_freeze_manifest.json').write_text(json.dumps(value))
    return snapshot,freeze


def test_work_audit_binds_frozen_routes_and_original_population_before_costing(tmp_path):
    snapshot,freeze=frozen_work_files(tmp_path)
    value=implementation().audit_r1_snapshot(snapshot,freeze)
    assert value['original_target_denominator']==2
    assert value['observed_reports']==1
    assert value['pending_reports']==1
    assert value['nonzero_stock_connected_targets']==1
    assert value['totals']['actual_generated_output_tokens']==10
    assert value['totals']['reused_output_tokens_not_charged']==100
    assert value['new_model_inference'] is False
    snapshot.write_text('{}')
    with pytest.raises(ValueError):implementation().audit_r1_snapshot(snapshot,freeze)
