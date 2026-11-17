from copy import deepcopy
import json
import pytest


def fixture():
    from mechet.route_refit import decision_identity,grounded_action_key
    from mechet.route_process_reward import digest,returns_to_go
    from mechet.route_failure_policy import FIRST_REJECTION,REWARD_VERSION
    def rec(uid,name,args):
        return dict(request_id=uid,behavior_version='b0',occurrence='0',accepted=True,
            state_before='[CH4:1]',state_after='[CH4:1]',prompt_token_ids=[1,2],
            generated_token_ids=[3],behavior_logprobs=[-.2],
            raw='<tool_call>'+json.dumps(dict(name=name,arguments=args))+'</tool_call>')
    prefix=rec('p','import_fragments',{'fragments':[]})
    a=rec('a','apply_electron_flow',{'choice':'a'})
    b=rec('b','apply_electron_flow',{'choice':'b'})
    other=rec('other','finish_trace',{})
    ctx=dict(target_id='C',tree_id='tree',occurrence='0',prompt_token_ids=[1,2],
        tools=[],accepted_history=[{k:prefix[k] for k in ['raw','state_before','state_after']}],
        route_context={},behavior_version='b0',executor_contract='v1',mapped_state='[CH4:1]')
    node=decision_identity(**ctx)
    def report(records,events,path):
        credit=dict(version=REWARD_VERSION,request_ids=[r['request_id'] for r in records],
                    event_rewards=events,returns=returns_to_go(events))
        credit['sha256']=digest(credit)
        return dict(status='stock_solved',route_failure_policy=FIRST_REJECTION,reward=sum(events),
                    records=records,reward_evidence={'process_credit':credit},refit_path=path)
    def path(record):
        return [dict(node_id=node,index=1,request_id=record['request_id'],proposal='original_policy',
            name='apply_electron_flow',action=grounded_action_key(record['state_before'],record['raw']))]
    suffix=report([deepcopy(prefix),b],[0.,-.02],path(b))
    first=report([prefix,a],[0.,.2],path(a))
    first['refit']=dict(conditioning={node:ctx},config={'matched_continuations':1},suffix_reports=[suffix])
    return [first,report([other],[-.01],[])]


def test_branch_replaces_g4_and_prefix_is_trained_once():
    from mechet.route_refit_credit import training_records
    group=fixture();records,stats=training_records(group)
    values={r['request_id']:a for r,a in records}
    assert values==pytest.approx({'p':.105,'a':.11,'b':-.11,'other':-.105})
    assert stats['conditional_groups']==1 and stats['unique_receipts']==4


def test_censored_suffix_masks_comparison_and_preserves_original_g4():
    from mechet.route_refit_credit import training_records
    group=fixture();suffix=group[0]['refit']['suffix_reports'][0]
    suffix['reward']=None;suffix['reward_evidence']['process_credit']=None
    suffix['status']='resource_censored'
    records,stats=training_records(group)
    assert {r['request_id']:a for r,a in records}==pytest.approx({'p':.105,'a':.1,'other':-.105})
    assert stats['conditional_groups']==0


def test_censored_unparsed_proposal_masks_whole_conditional_node():
    from mechet.route_refit_credit import training_records
    group=fixture();node=next(iter(group[0]['refit']['conditioning']))
    group[0]['refit']['node_censored_receipts']={node:['unparsed-resource-draw']}
    records,stats=training_records(group)
    assert stats['conditional_groups']==0 and len(records)==3


def test_changed_prompt_or_unknown_proposal_cannot_claim_conditional_credit():
    from mechet.route_refit_credit import training_records
    group=fixture();group[0]['refit']['suffix_reports'][0]['records'][1]['prompt_token_ids']=[99]
    with pytest.raises(ValueError,match='CONDITIONING'):training_records(group)
    group=fixture();group[0]['refit']['suffix_reports'][0]['refit_path'][0]['proposal']='deterministic_puct'
    records,stats=training_records(group)
    assert stats['conditional_groups']==0 and len(records)==3


def test_unique_branch_optimizer_matches_dense_independent_gradient():
    import torch
    from types import SimpleNamespace
    from mechet.external_route_rl import optimize_routes
    from mechet.route_process_reward import ALGORITHM
    from scripts.pr96_earho_contract import token_logps_batch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(13,5);self.head=torch.nn.Linear(5,13)
        def forward(self,input_ids,attention_mask=None,logits_to_keep=0,**kwargs):
            h=(self.emb(input_ids)*attention_mask[...,None]).cumsum(1)
            return SimpleNamespace(logits=self.head(h[:,logits_to_keep]))
    torch.manual_seed(31);actor=Tiny();dense=deepcopy(actor);reference=deepcopy(actor)
    group=fixture();records=group[0]['records']+group[1]['records']+[group[0]['refit']['suffix_reports'][0]['records'][1]]
    for report in [*group,*group[0]['refit']['suffix_reports']]:
        for record in report['records']:
            record['behavior_logprobs']=token_logps_batch(actor,[(record['prompt_token_ids'],record['generated_token_ids'])])[0].detach().tolist()
    for record,advantage in zip(records,[.105,.11,-.105,-.11]):
        pair=(record['prompt_token_ids'],record['generated_token_ids'])
        current=token_logps_batch(dense,[pair])[0]
        old=torch.tensor(record['behavior_logprobs']);ref=token_logps_batch(reference,[pair])[0].detach()
        ratio=(current-old).exp();delta=ref-current
        loss=(.02*ratio*(delta.exp()-delta-1)-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)).sum()/(2*1024)
        loss.backward()
    torch.optim.SGD(dense.parameters(),lr=.1).step()
    cfg=dict(algorithm=ALGORITHM,refit={},candidates=2,loss_batch_size=2,pad_token_id=0,
        loss_token_normalizer=1024.,behavior_logprob_tolerance=.25,clip_epsilon=.2,kl_beta=.02,max_grad_norm=1e6)
    result=optimize_routes(actor,reference,torch.optim.SGD(actor.parameters(),lr=.1),[group],cfg)
    assert result['sampled_action_records']==4 and result['conditional_groups']==1
    assert result['policy_receipt_scope']=='unique_actual_sampled_receipts_shared_prefix_once'
    for actual,wanted in zip(actor.parameters(),dense.parameters()):
        assert torch.allclose(actual,wanted,atol=2e-7,rtol=1e-6)
