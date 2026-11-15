from copy import deepcopy
from types import SimpleNamespace
import math

import pytest


def test_optimizer_uses_each_actions_future_advantage_and_matches_independent_gradient():
    import torch
    from mechet.external_route_rl import optimize_routes
    from mechet.route_process_reward import ALGORITHM, VERSION, digest, returns_to_go
    from scripts.pr96_earho_contract import token_logps_batch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=torch.nn.Embedding(13,5);self.head=torch.nn.Linear(5,13)
            self.config=SimpleNamespace(use_cache=True)
        def forward(self,input_ids,attention_mask=None,use_cache=False,logits_to_keep=0,**kwargs):
            h=(self.emb(input_ids)*attention_mask[...,None]).cumsum(1)
            return SimpleNamespace(logits=self.head(h[:,logits_to_keep]))
    torch.manual_seed(31)
    actor=Tiny();dense=deepcopy(actor);ref=deepcopy(actor)
    with torch.no_grad():ref.head.bias.add_(torch.linspace(-.1,.1,13))
    pairs=[([1,2],[3,4]),([5],[6]),([7,8],[9,10])]
    old=[v.detach().tolist() for v in token_logps_batch(actor,pairs)]
    rewards=[[0.,.2],[-.01],[.1],[0.,0.]]
    group=[]
    for i,events in enumerate(rewards):
        records=[]
        for t in range(len(events)):
            n=(i+t)%len(pairs);p,r=pairs[n]
            records.append(dict(request_id=f'e{i}/a{t}',prompt_token_ids=p,
                                generated_token_ids=r,behavior_logprobs=old[n]))
        credit=dict(version=VERSION,request_ids=[r['request_id'] for r in records],
                    event_rewards=events,returns=returns_to_go(events))
        credit['sha256']=digest(credit)
        group.append(dict(reward=math.fsum(events),records=records,
                          reward_evidence=dict(process_credit=credit)))
    censored=[dict(reward=None,records=[],reward_evidence=dict(process_credit=None))]*4
    terms=[]
    for i,report in enumerate(group):
        for t,record in enumerate(report['records']):
            own=sum(rewards[i][t:])
            baseline=sum(sum(other[t:]) for other in rewards)/4
            advantage=own-baseline
            pair=(record['prompt_token_ids'],record['generated_token_ids'])
            current=token_logps_batch(dense,[pair])[0]
            with torch.no_grad():reference=token_logps_batch(ref,[pair])[0]
            ratio=(current-torch.tensor(record['behavior_logprobs'])).exp()
            delta=reference-current
            terms.append((.02*ratio*(delta.exp()-delta-1)-
                torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)).sum()/(4*1024))
    sum(terms).backward();torch.optim.SGD(dense.parameters(),lr=.1).step()
    cfg=dict(algorithm=ALGORITHM,candidates=4,loss_batch_size=2,pad_token_id=0,
             behavior_logprob_tolerance=1e-5,loss_token_normalizer=1024.,
             clip_epsilon=.2,kl_beta=.02,max_grad_norm=1e6)
    result=optimize_routes(actor,ref,torch.optim.SGD(actor.parameters(),lr=.1),[group,censored],cfg)
    assert result['policy_groups']==1 and result['censored_groups']==1
    assert result['advantage_estimator']=='centered_action_return_to_go_no_std'
    assert result['sampled_action_records']==6
    for actual,wanted in zip(actor.parameters(),dense.parameters()):
        assert torch.allclose(actual,wanted,atol=2e-7,rtol=1e-6)
    assert all(p.grad is None for p in ref.parameters())


def test_process_protocol_is_explicit_and_cannot_be_silently_enabled_on_legacy():
    from mechet.route_process_reward import ALGORITHM, CONFIG, VERSION, validate_process_protocol
    cfg=dict(algorithm=ALGORITHM,process_reward=deepcopy(CONFIG),reward_contract=VERSION,
             advantage_estimator='centered_action_return_to_go_no_std',
             public_observation_contract='route_sft_aligned_feedback_v9',partial_reward_weight=.2)
    validate_process_protocol(cfg)
    for key,value in [('reward_contract','legacy'),('partial_reward_weight',.5),
                      ('public_observation_contract','route_action_feedback_v8')]:
        with pytest.raises(ValueError,match='PROCESS_'):
            validate_process_protocol(dict(cfg,**{key:value}))
    bad=deepcopy(cfg);bad['process_reward']['validator_weight']=10
    with pytest.raises(ValueError,match='PROCESS_'):validate_process_protocol(bad)
    with pytest.raises(ValueError,match='PROCESS_'):
        validate_process_protocol(dict(cfg,algorithm='external_target_only_route_dr_grpo_v2'))
