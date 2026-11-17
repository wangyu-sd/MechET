from copy import deepcopy
from types import SimpleNamespace


def test_microbatch_four_matches_one_real_forward_gradients_and_parameter_update():
    import torch
    from mechet.external_route_rl import optimize_routes,DR_GRPO_ALGORITHM
    from scripts.pr96_earho_contract import token_logps_batch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(19,5);self.head=torch.nn.Linear(5,19)
        def forward(self,input_ids,attention_mask,logits_to_keep,use_cache,position_ids=None):
            hidden=(self.emb(input_ids)*attention_mask[...,None]).cumsum(1)
            return SimpleNamespace(logits=self.head(hidden[:,logits_to_keep]))
    torch.manual_seed(21);initial=Tiny();reference=deepcopy(initial)
    with torch.no_grad():reference.head.bias.add_(torch.linspace(-.3,.3,19))
    pairs=[([1,2],[3,4,18]),([5],[6,18]),([7,8,9],[10,18]),([11,12],[13,18])]
    old=token_logps_batch(initial,pairs)
    records=[dict(prompt_token_ids=p,generated_token_ids=r,behavior_logprobs=b.detach().tolist())
             for (p,r),b in zip(pairs,old)]
    groups=[[dict(reward=y,records=[records[i]]) for i,y in enumerate([0.,1.,.2,0.])],
            [dict(reward=y,records=[records[i]]) for i,y in enumerate([1.,0.,0.,.2])]]
    results=[]
    for size in [1,4]:
        actor=deepcopy(initial);optimizer=torch.optim.SGD(actor.parameters(),lr=.01)
        stats=optimize_routes(actor,reference,optimizer,groups,dict(algorithm=DR_GRPO_ALGORITHM,
            candidates=4,loss_batch_size=size,pad_token_id=0,clip_epsilon=.2,kl_beta=.02,
            loss_token_normalizer=1024.,max_grad_norm=100.,behavior_logprob_tolerance=.25))
        results.append((actor,stats))
    for x,y in zip(results[0][0].parameters(),results[1][0].parameters()):
        torch.testing.assert_close(x.grad,y.grad,rtol=2e-5,atol=1e-8)
        torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-8)
    assert results[0][1]['global_policy_groups']==results[1][1]['global_policy_groups']==2
