"""Executed CPU chemistry journals and tiny CPU gradients; no neural inference."""
from copy import deepcopy
import importlib.util
import json
import sqlite3
from types import SimpleNamespace

import pytest

ACTOR='a'*64


def api():
    assert importlib.util.find_spec('mechet.route_self_distill_runtime') is not None, 'distill runtime missing'
    from mechet import route_self_distill_runtime
    return route_self_distill_runtime


@pytest.fixture
def scored(tmp_path):
    from test_route_first_rejection import episode
    from test_route_native_observation import flow,send
    from mechet.route_refit import RefitRouteEpisode
    from mechet.route_refit_runtime import RefitCollection
    from mechet.route_policy_sampling import collect_route_groups
    from mechet.route_process_reward import ProcessRouteReward
    from mechet.external_route_data import edge_signature
    from scripts.train_external_route_rl import score_groups
    base=episode();stock={'[CH3+]','[OH-]'}
    controller=RefitCollection([RefitRouteEpisode(base.cfg,'CO',stock,'tree',17)],
                              dict(max_nodes_per_tree=4,max_expansions=2,max_duplicates=2,
                                   max_forks_per_tree=4,matched_continuations=1))
    class Sampler:
        cfg=dict(generation_batch_size=8,executor_coalesce_seconds=0.)
        def generate(self, requests):
            out=[]
            for request in requests:
                owner=controller.owner(request.request_id)
                if owner.ep.node.actions and '/fork-' in request.request_id:
                    name,args='finish_trace',{}
                else:
                    name,args='apply_electron_flow',flow(owner.ep)
                    if owner.ep.node.actions or '/fork-' in request.request_id:
                        args['electron_flow'][0]['source']='the bond between atoms A99 and A98'
                out.append(dict(request_id=request.request_id,behavior_version=request.behavior_version,
                                prompt_token_ids=[1,2],generated_token_ids=[3,4],behavior_logprobs=[-.1,-.2],
                                raw='<tool_call>'+json.dumps(dict(name=name,arguments=args))+'</tool_call>'))
            return out
    journal=[]
    reports=collect_route_groups(controller.initial,Sampler(),workers=2,journal=journal.append,
                                on_completed_action=controller.completed)
    packed=controller.pack(reports)
    positive=next(r for r in packed[0]['refit']['suffix_reports'] if r['status']=='stock_solved')
    db=tmp_path/'reference.sqlite'
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE tasks(id TEXT,target TEXT,split TEXT,reference_steps INTEGER); CREATE TABLE edges(task_id TEXT,signature TEXT);')
        conn.execute('INSERT INTO tasks VALUES(?,?,?,?)',('t','CO','training',1))
        edge=positive['reactions'][0]
        conn.execute('INSERT INTO edges VALUES(?,?)',('t',edge_signature(edge['product'],edge['precursors'])))
    other=episode('other');send(other,'finish_trace',{})
    with ProcessRouteReward(db,stock,failure_policy=base.cfg['route_failure_policy']) as oracle:
        groups=score_groups([dict(task_id='t',target='CO')],[packed[0],other.finish_report()],2,oracle)
        yield groups,oracle,journal


def export(groups, oracle, **changes):
    cfg=dict(actual_actor_sha=ACTOR,training_task_ids={'t':'CO'},behavior_version='first-rejection',
             tokenizer_sha='1'*64,tool_schema_sha='2'*64,executor_sha='3'*64,
             stock_contract_sha='4'*64,budget_sha='5'*64,reward_contract_sha='6'*64,reward_oracle=oracle)
    cfg.update(changes)
    return api().export_search_states(groups,**cfg)


def test_export_real_scored_refit_preserves_task_id_preimage_and_unique_prefix(scored):
    from mechet.route_self_distill import prepare_state
    groups,oracle,journal=scored
    result=export(groups,oracle)
    assert result['states']
    all_ids=[c['receipt_id'] for s in result['states'] for c in s['candidates']]
    assert len(all_ids)==len(set(all_ids))
    assert len(all_ids)<=len(journal)
    for state in result['states']:
        assert state['conditioning']['target_id']=='t'
        assert state['conditioning']['refit_preimage']['target_id']=='CO'
        assert state['conditioning']['refit_preimage']['mapped_state']
    prepared=[prepare_state(s,result['receipt_manifest']['receipts'],expected_actor_sha=ACTOR,
                            training_target_ids={'t'},eos_token_id=4) for s in result['states']]
    assert any(s['status']=='prepared' and len(s['targets'])==1 for s in prepared)
    assert result['receipt_manifest']['scope']=='executed_journal_and_frozen_training_reward_oracle_not_independent_replay'
    assert all(r['independent_executor_replay_verified'] is False for r in result['receipt_manifest']['receipts'].values())


@pytest.mark.parametrize('change,code',[('digest','CREDIT'),('status','REWARD'),('conditioning','CONDITIONING'),
                                       ('task_id','TRAIN'),('proposal','PROPOSAL'),('record','CONDITIONING')])
def test_export_refuses_corrupt_scored_provenance(scored,change,code):
    groups,oracle,_=scored;groups=deepcopy(groups);report=groups[0][0]
    if change=='digest':report['reward_evidence']['process_credit']['sha256']='0'*64
    elif change=='status':report['status']='stock_solved'
    elif change=='conditioning':next(iter(report['refit']['conditioning'].values()))['prompt_token_ids']=[99]
    elif change=='task_id':report['task_id']='CO'
    elif change=='proposal':report['refit_path'][0]['proposal']='unknown'
    else:report['records'][0]['prompt_token_ids']=[8]
    with pytest.raises(ValueError,match=code):export(groups,oracle)


def test_censored_matched_node_masks_all_siblings_including_survivor(scored):
    from mechet.route_self_distill import prepare_state
    groups,oracle,_=scored;groups=deepcopy(groups);root=groups[0][0]
    positive=next(r for r in root['refit']['suffix_reports'] if r['status']=='stock_solved')
    final=positive['refit_path'][-1];node=final['node_id']
    root['refit'].setdefault('node_censored_receipts',{})[node]=['interrupted-new-request']
    result=export(groups,oracle)
    state=next(s for s in result['states'] if s['node_id']==node)
    assert state['matched_group_censored'] is True
    prepared=prepare_state(state,result['receipt_manifest']['receipts'],expected_actor_sha=ACTOR,
                           training_target_ids={'t'},eos_token_id=4)
    assert prepared['status']=='skipped' and prepared['reason']=='all_ineligible'
    assert all(result['receipt_manifest']['receipts'][c['receipt_id']]['censored'] for c in state['candidates'])


def test_export_rechecks_oracle_and_does_not_accept_supplied_verified_flag(scored):
    groups,oracle,_=scored;groups=deepcopy(groups)
    groups[0][0]['reward_evidence']['verified_stock_closed']=True
    with pytest.raises(ValueError,match='REWARD'):export(groups,oracle)
    with pytest.raises(ValueError,match='ORACLE'):export(scored[0],None)


def examples():
    from mechet.route_self_distill import OBJECTIVE
    return [dict(state_key='s1',kind='self_search',status='prepared',objective=OBJECTIVE,split='training',
                 external_teacher_calls=0,source_actor_sha=ACTOR,mixture_weight=.7,prompt_token_ids=[1,2],
                 targets=[dict(generated_token_ids=[3,4],probability=.25),dict(generated_token_ids=[5,6,4],probability=.75)]),
            dict(state_key='s2',kind='original_sft',status='prepared',objective=OBJECTIVE,split='training',
                 external_teacher_calls=0,source_actor_sha=None,mixture_weight=.3,prompt_token_ids=[7],
                 targets=[dict(generated_token_ids=[8,4],probability=1.)])]


def tiny():
    import torch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(12,5);self.head=torch.nn.Linear(5,12)
        def forward(self,input_ids,attention_mask=None,logits_to_keep=0,**kwargs):
            states=(self.emb(input_ids)*attention_mask[...,None]).cumsum(1)
            return SimpleNamespace(logits=self.head(states[:,logits_to_keep]))
    return Tiny()


def update_cfg():
    return dict(actual_actor_sha=ACTOR,source_actor_sha=ACTOR,eos_token_id=4,pad_token_id=0,
                loss_batch_size=2,training_max_padded_tokens=100,max_grad_norm=1e6,
                policy_precision='legacy_fp32_lora')


def test_update_matches_independent_dense_summed_native_gradient_with_source_mass():
    import torch
    from scripts.pr96_earho_contract import token_logps_batch
    torch.manual_seed(17);actor=tiny();dense=deepcopy(actor);rows=examples()
    for row in rows:
        for target in row['targets']:
            p,r=row['prompt_token_ids'],target['generated_token_ids']
            logps=token_logps_batch(dense,[(p,r)])[0]
            (-row['mixture_weight']*target['probability']*logps.sum()).backward()
    torch.optim.SGD(dense.parameters(),lr=.1).step()
    result=api().distillation_update(actor,torch.optim.SGD(actor.parameters(),lr=.1),rows,update_cfg())
    assert result['optimizer_updated'] is True
    assert result['objective']=='offline_search_selected_native_ce_not_grpo'
    assert result['policy_tokens']==7 and result['states']==2
    for actual,wanted in zip(actor.parameters(),dense.parameters()):
        assert torch.allclose(actual,wanted,atol=2e-7,rtol=1e-6)


def test_update_batch_partition_does_not_change_gradient_or_drop_eos():
    import torch
    torch.manual_seed(8);a=tiny();b=deepcopy(a)
    first=api().distillation_update(a,torch.optim.SGD(a.parameters(),lr=.02),examples(),update_cfg())
    cfg=dict(update_cfg(),loss_batch_size=1,training_max_padded_tokens=5)
    second=api().distillation_update(b,torch.optim.SGD(b.parameters(),lr=.02),examples(),cfg)
    assert first['loss']==pytest.approx(second['loss'],abs=1e-6)
    assert second['batches']==3
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p,q,atol=2e-7,rtol=1e-6)


@pytest.mark.parametrize('change,code',[('split','TRAINING'),('actor','ACTOR'),('eos','EOS'),
                                       ('mass','MASS'),('probability','PROBABILITY'),('duplicate','DUPLICATE')])
def test_update_validates_every_example_before_parameter_update(change,code):
    import torch
    actor=tiny();before=deepcopy(actor.state_dict());rows=examples()
    if change=='split':rows[0]['split']='test'
    elif change=='actor':rows[0]['source_actor_sha']='0'*64
    elif change=='eos':rows[0]['targets'][0]['generated_token_ids']=[3]
    elif change=='mass':rows[0]['mixture_weight']=-1
    elif change=='probability':rows[0]['targets'][0]['probability']=.5
    else:rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError,match=code):api().distillation_update(actor,torch.optim.SGD(actor.parameters(),lr=.1),rows,update_cfg())
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in before.items())
