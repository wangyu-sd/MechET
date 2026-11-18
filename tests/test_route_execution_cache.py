from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
import time

import pytest


def receipt(arguments, uid='r1', lp=-.3):
    return dict(request_id=uid,raw='<tool_call>'+json.dumps(dict(name='apply_electron_flow',arguments=arguments))+'</tool_call>',
                generated_token_ids=[2],behavior_logprobs=[lp])


def test_singleflight_rebuilds_each_proposals_metadata_and_scores():
    from mechet.route_execution_cache import CachedChemistry, ExecutionCache
    from mechet.route_policy_episode import NativeChemistry
    from test_route_native_observation import episode, flow
    ep=episode();ep.next_decision();args=flow(ep);base=NativeChemistry();original=base.step
    calls=[]
    def step(node, output):
        calls.append(1);time.sleep(.05);return original(node,output)
    base.step=step;cache=ExecutionCache();chem=CachedChemistry(base,cache,{'executor':'frozen'})
    a=receipt(args);b=receipt(args,'r2',-.7)
    bargs=deepcopy(args);bargs['electron_flow'][0]['instruction']='another explanation'
    b=receipt(bargs,'r2',-.7)
    with ThreadPoolExecutor(2) as pool:
        f=pool.submit(chem.step,deepcopy(ep.node),a);g=pool.submit(chem.step,deepcopy(ep.node),b)
        x,e=f.result();y,e2=g.result()
    assert e==e2=='' and x.state==y.state and len(calls)==1
    assert x.logprob==pytest.approx(ep.node.logprob-.3)
    assert y.logprob==pytest.approx(ep.node.logprob-.7)
    assert x.actions[-1]['arguments']==args and y.actions[-1]['arguments']==bargs
    assert cache.metrics()['accepted_execution_cache_hits']==1
    assert cache.metrics()['inflight_waiters']==1


def test_cached_flow_still_checks_each_nodes_visited_states():
    from mechet.route_execution_cache import CachedChemistry,ExecutionCache
    from mechet.route_policy_episode import NativeChemistry
    from scripts.run_natural_language_value_search import visible
    from test_route_native_observation import episode,flow
    ep=episode();ep.next_decision();chem=CachedChemistry(NativeChemistry(),ExecutionCache(),{'executor':'frozen'})
    output=receipt(flow(ep));child,error=chem.step(ep.node,output);assert not error
    node=deepcopy(ep.node);node.visited.add(visible(child.state))
    child,error=chem.step(node,dict(output,request_id='r2'))
    assert child is None and error=='ValueError:STATE_CYCLE'


def test_cache_never_memoizes_resource_errors_or_infrastructure_exceptions():
    from mechet.route_execution_cache import ExecutionCache
    cache=ExecutionCache();calls=[]
    def resource():
        calls.append(1);return {'error':'RESOURCE_CENSORED:MEMORY'}
    cache.resolve('k','r1',resource);cache.resolve('k','r2',resource)
    assert len(calls)==2
    with pytest.raises(RuntimeError):cache.resolve('e','r1',lambda:(_ for _ in ()).throw(RuntimeError('transport')))
    assert cache.resolve('e','r2',lambda:{'state':'ok'})['state']=='ok'


def test_returned_infrastructure_error_is_not_a_cached_chemical_rejection():
    from mechet.route_execution_cache import ExecutionCache
    cache=ExecutionCache();calls=[]
    def memory():
        calls.append(1);return {'error':'MemoryError:allocation failed'}
    cache.resolve('k','r1',memory);cache.resolve('k','r2',memory)
    assert len(calls)==2


def test_history_and_finish_are_not_cloned_from_the_cache():
    from mechet.route_execution_cache import CachedChemistry,ExecutionCache
    from mechet.route_policy_episode import NativeChemistry
    from test_route_native_observation import episode,flow
    ep=episode();ep.next_decision();base=NativeChemistry();chem=CachedChemistry(base,ExecutionCache(),{'executor':'frozen'})
    r=receipt(flow(ep));x,_=chem.step(ep.node,r)
    n=deepcopy(ep.node);n.actions=[{'name':'old','arguments':{},'result':{}}];n.tokens=9;n.logprob=-4
    y,_=chem.step(n,dict(r,request_id='r2'))
    assert y.actions[:-1]==n.actions and y.tokens==10 and y.logprob==pytest.approx(-4.3)
    assert x.actions[:-1]==ep.node.actions
    finish=dict(r,raw='<tool_call>{"name":"finish_trace","arguments":{}}</tool_call>')
    actual,error=chem.step(y,finish);expected,e=base.step(y,finish)
    assert actual==expected and error==e


def test_cache_exposes_producer_and_bound_pure_witness_without_changing_receipt():
    from mechet.route_execution_cache import CachedChemistry,ExecutionCache
    from mechet.route_policy_episode import NativeChemistry
    from mechet.external_route_data import signature
    from test_route_native_observation import episode,flow
    ep=episode();ep.next_decision();cache=ExecutionCache();base=NativeChemistry()
    chem=CachedChemistry(base,cache,{'executor':'frozen'});a=receipt(flow(ep))
    original=deepcopy(a);x,error=chem.step(ep.node,a)
    provenance=cache.provenance('r1')
    assert a==original and not error
    b=dict(a,request_id='r2',behavior_logprobs=[-.9]);y,error=chem.step(ep.node,b)
    second=cache.provenance('r2')
    assert provenance['producer_receipt_id']==second['producer_receipt_id']=='r1'
    assert provenance['request_id']=='r1' and second['request_id']=='r2'
    assert provenance['witness_sha']==signature(dict(execution_key=provenance['execution_key'],
        producer_receipt_id='r1',transition=dict(error='',state=x.state,next_map=x.next_map,
        imported=x.imported,result=x.actions[-1]['result'])))
    assert second['witness_sha']==provenance['witness_sha']
    assert x.actions==y.actions and y.logprob==pytest.approx(ep.node.logprob-.9)
    assert cache.metrics()['cache_reconstruction_failures']==0


def test_corrupted_cached_transition_fails_witness_and_counts_reconstruction_failure():
    from mechet.route_execution_cache import CachedChemistry,ExecutionCache
    from mechet.route_policy_episode import NativeChemistry
    from test_route_native_observation import episode,flow
    ep=episode();ep.next_decision();cache=ExecutionCache()
    chem=CachedChemistry(NativeChemistry(),cache,{'executor':'frozen'})
    output=receipt(flow(ep));chem.step(ep.node,output)
    # Corrupt a durable pure successor at the cache boundary; consuming it must
    # propagate rather than return a foreign/forged chemical state.
    next(iter(cache.entries.values())).result()['transition']['state']='forged'
    with pytest.raises(ValueError,match='CACHE.*WITNESS'):
        chem.step(ep.node,dict(output,request_id='r2'))
    assert cache.metrics()['cache_reconstruction_failures']==1


def test_reconstruction_exception_is_counted_and_propagated(monkeypatch):
    from mechet.route_execution_cache import CachedChemistry,ExecutionCache
    from mechet.route_policy_episode import NativeChemistry
    from test_route_native_observation import episode,flow
    ep=episode();ep.next_decision();cache=ExecutionCache()
    chem=CachedChemistry(NativeChemistry(),cache,{'executor':'frozen'})
    output=receipt(flow(ep));chem.step(ep.node,output)
    import scripts.run_natural_language_value_search as chemistry
    def broken(*args,**kwargs):raise RuntimeError('reconstruction failed')
    monkeypatch.setattr(chemistry,'Node',broken)
    with pytest.raises(RuntimeError,match='reconstruction failed'):
        chem.step(ep.node,dict(output,request_id='r2'))
    assert cache.metrics()['cache_reconstruction_failures']==1


def test_missing_cached_witness_field_is_counted_and_propagated():
    from mechet.route_execution_cache import CachedChemistry,ExecutionCache
    from mechet.route_policy_episode import NativeChemistry
    from test_route_native_observation import episode,flow
    ep=episode();ep.next_decision();cache=ExecutionCache()
    chem=CachedChemistry(NativeChemistry(),cache,{'executor':'frozen'})
    output=receipt(flow(ep));chem.step(ep.node,output)
    del next(iter(cache.entries.values())).result()['transition']
    with pytest.raises((KeyError,ValueError)):
        chem.step(ep.node,dict(output,request_id='r2'))
    assert cache.metrics()['cache_reconstruction_failures']==1
def test_cache_producer_witness_survives_runtime_shutdown(tmp_path):
    import json
    from mechet.route_execution_cache import ExecutionCache
    from mechet.external_route_data import file_sha
    cache=ExecutionCache()
    cache.resolve('key','producer',lambda:dict(error='',state='state'))
    cache.resolve('key','consumer',lambda:(_ for _ in ()).throw(AssertionError('reexecuted')))
    result=cache.persist_provenance(tmp_path/'cache-witnesses.json')
    rows=json.loads((tmp_path/'cache-witnesses.json').read_text())
    assert result['sha256']==file_sha(tmp_path/'cache-witnesses.json')
    assert rows['consumer']['request_id']=='consumer'
    assert rows['consumer']['producer_receipt_id']=='producer'
    assert rows['consumer']['witness_sha']==rows['producer']['witness_sha']
