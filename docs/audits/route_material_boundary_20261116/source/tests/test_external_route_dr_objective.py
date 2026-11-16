"""Tiny CPU tensor checks of the approved fixed-scale full-route objective.

These are software gradient checks, not model or chemistry experiments.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest


def test_centered_advantages_are_not_rloo_or_standard_deviation_scaled():
    from mechet.external_route_rl import route_group_advantages
    rewards=[0.,.1,.3,1.]
    centered=route_group_advantages(rewards,'external_target_only_route_dr_grpo_v2')
    assert centered==pytest.approx([-.35,-.25,-.05,.65])
    rloo=route_group_advantages(rewards,'external_target_only_route_rloo_v1')
    assert rloo==pytest.approx([v*4/3 for v in centered])
    assert route_group_advantages([1.,1.,1.,1.],
        'external_target_only_route_dr_grpo_v2')==[0.,0.,0.,0.]
    assert route_group_advantages([0.,None,1.,0.],
        'external_target_only_route_dr_grpo_v2') is None
    with pytest.raises(ValueError,match='ALGORITHM'):
        route_group_advantages(rewards,'pretend_gspo')


@pytest.mark.parametrize('advantage',[-.6,0.,.4])
def test_token_kernel_value_and_gradient_match_pinned_trl_dr_grpo_math(advantage):
    import torch
    from mechet.route_policy_objective import action_loss

    # This independent expression is TRL v1.17.1's token clipping / fixed
    # normalizer with its importance-sampling KL correction enabled, using our
    # declared N*G*C as the total normalizer. It does not call action_loss.
    old=torch.tensor([-2.,-3.,-4.,-5.,-6.],dtype=torch.float64,requires_grad=True)
    ref=torch.tensor([-2.4,-2.7,-4.2,-4.5,-6.3],dtype=torch.float64,requires_grad=True)
    current=(old.detach()+torch.tensor([-.6,-.1,0.,.15,.6],dtype=torch.float64)).requires_grad_()
    independent=current.detach().clone().requires_grad_()
    normalization=2*4*1024
    actual=action_loss(current,old,ref,advantage,clip=.2,beta=.02,normalizer=normalization)
    ratio=torch.exp(independent-old.detach())
    unclipped=ratio*advantage
    clipped=torch.clamp(ratio,min=.8,max=1.2)*advantage
    delta=ref.detach()-independent
    kl=torch.exp(delta)-1-delta
    expected=(.02*ratio*kl-torch.minimum(unclipped,clipped)).sum()/normalization
    assert torch.allclose(actual['total'],expected,atol=1e-15,rtol=1e-12)
    grad=torch.autograd.grad(actual['total'],current,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected,independent,retain_graph=True)[0]
    assert torch.allclose(grad,expected_grad,atol=1e-15,rtol=1e-12)
    actual['total'].backward()
    assert old.grad is None and ref.grad is None
    # A detached importance ratio would produce a different KL gradient here.
    detached_ratio=ratio.detach()
    wrong=(.02*detached_ratio*kl-torch.minimum(unclipped,clipped)).sum()/normalization
    wrong_grad=torch.autograd.grad(wrong,independent)[0]
    assert not torch.allclose(grad,wrong_grad,atol=1e-10,rtol=1e-7)


def test_saved_assistant_eos_and_rejected_tokens_are_policy_tokens_only():
    from mechet.route_policy_objective import policy_sequences
    report=dict(status='rejected_action',records=[dict(
        prompt_token_ids=[1,2,3],generated_token_ids=[7,16],
        behavior_logprobs=[-.4,-.8],raw='malformed XML',
        state_after='private-looking tool result must not become policy tokens')])
    assert policy_sequences(report)==[([1,2,3],[7,16],[-.4,-.8])]


def test_reward_evidence_labels_training_and_validation_namespaces():
    from mechet.external_route_data import edge_signature
    from mechet.external_route_rl import HiddenRouteReward

    class FrozenDatabase:
        def __init__(self,split):
            self.split=split

        def execute(self,query,parameters):
            if query.startswith('SELECT target'):
                return SimpleNamespace(fetchone=lambda:('CCO',self.split,2))
            return [(edge_signature('CCO',['CC','O']),)]

    report=dict(root='CCO',status='stock_solved',stock_leaves=['CC','O'],
                reactions=[dict(product='CCO',precursors=['CC','O'],occurrence='0')])
    for split,label in [('training','TRAIN'),('validation','VALID')]:
        oracle=object.__new__(HiddenRouteReward)
        oracle.db=FrozenDatabase(split);oracle.split=split
        oracle.stock={'CC','O'};oracle.partial_weight=.2
        from mechet.route_materials import EXACT_MATERIAL
        oracle.material_contract=EXACT_MATERIAL
        result=oracle.score('fixture',report)
        assert result['chemical_validation']=='hidden_'+label+'_reference_support_only'
        assert result['reward_status']=='verified_closed'


def test_dr_grpo_compaction_matches_dense_variable_length_groups_with_censoring():
    import torch
    from mechet.external_route_rl import optimize_routes
    from scripts.pr96_earho_contract import token_logps_batch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=torch.nn.Embedding(17,5)
            self.head=torch.nn.Linear(5,17)

        def forward(self,input_ids,attention_mask,logits_to_keep,use_cache,position_ids=None):
            hidden=(self.embedding(input_ids)*attention_mask[...,None]).cumsum(1)
            return SimpleNamespace(logits=self.head(hidden[:,logits_to_keep]))

    torch.manual_seed(23)
    actor=Tiny();dense=deepcopy(actor);reference=deepcopy(actor)
    with torch.no_grad():
        reference.head.bias.add_(torch.linspace(-.2,.2,17))
    pairs=[([1,2],[3,4,16]),([5],[6,16]),([7,8,9],[10,16])]
    old=[v.detach() for v in token_logps_batch(actor,pairs)]
    records=[dict(prompt_token_ids=p,generated_token_ids=r,behavior_logprobs=b.tolist())
             for (p,r),b in zip(pairs,old)]
    groups=[[
        dict(reward=reward,records=[records[i%3]]*(1+i%2))
        for i,reward in enumerate(rewards)]
        for rewards in ([0.,1.,.2,0.],[0.,.1,.1,.2],[None,1.,0.,0.])]
    norm=2*4*1024
    expected=[]
    for group in groups[:2]:
        mean=sum(row['reward'] for row in group)/4
        for row in group:
            for record in row['records']:
                pair=(record['prompt_token_ids'],record['generated_token_ids'])
                logp=token_logps_batch(dense,[pair])[0]
                with torch.no_grad():
                    ref=token_logps_batch(reference,[pair])[0]
                behavior=torch.tensor(record['behavior_logprobs'])
                ratio=(logp-behavior).exp()
                advantage=row['reward']-mean
                surrogate=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
                delta=ref-logp
                expected.append((.02*ratio*(delta.exp()-delta-1)-surrogate).sum()/norm)
    sum(expected).backward()
    torch.optim.SGD(dense.parameters(),lr=.1).step()
    cfg=dict(algorithm='external_target_only_route_dr_grpo_v2',candidates=4,
             loss_batch_size=2,pad_token_id=0,behavior_logprob_tolerance=1e-5,
             loss_token_normalizer=1024.,clip_epsilon=.2,kl_beta=.02,max_grad_norm=1e6)
    stats=optimize_routes(actor,reference,torch.optim.SGD(actor.parameters(),lr=.1),groups,cfg)
    assert stats['policy_groups']==2 and stats['censored_groups']==1
    assert stats['sampled_action_records']==12 and stats['optimized_unique_records']<12
    assert stats['algorithm']==cfg['algorithm']
    for actual,wanted in zip(actor.parameters(),dense.parameters()):
        assert torch.allclose(actual,wanted,atol=2e-7,rtol=1e-6)
    assert all(p.grad is None for p in reference.parameters())
