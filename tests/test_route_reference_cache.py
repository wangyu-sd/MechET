"""Only immutable reference scores may reuse a forward; actor gradients stay real."""
from copy import deepcopy
from types import SimpleNamespace

import pytest


def test_reference_cache_deduplicates_exact_pairs_and_isolates_identity(tmp_path):
    import torch
    from mechet.route_reference_cache import ReferenceLogprobCache
    cache=ReferenceLogprobCache(2);calls=[]
    def compute(pairs):
        calls.extend(pairs);return [torch.tensor([-float(len(p))]*len(r)) for p,r in pairs]
    a=([1,2],[3]);b=([1],[4,5])
    values=cache.lookup([a,a,b],identity=('ref','tok','bf16','mask'),compute=compute,device='cpu')
    assert len(calls)==2 and values[0].tolist()==[-2.]
    values[0].add_(99.)
    assert cache.lookup([a],identity=('ref','tok','bf16','mask'),compute=compute,device='cpu')[0].tolist()==[-2.]
    cache.lookup([a],identity=('other-ref','tok','bf16','mask'),compute=compute,device='cpu')
    cache.lookup([a],identity=('other-ref','tok','fp32','mask'),compute=compute,device='cpu')
    assert len(calls)==4
    assert cache.snapshot()['coalesced_pairs']==1 and cache.snapshot()['cache_hits']==1


def test_reference_lru_eviction_never_loses_current_batch_outputs():
    import torch
    from mechet.route_reference_cache import ReferenceLogprobCache
    cache=ReferenceLogprobCache(1)
    values=cache.lookup([([1],[2]),([3],[4,5])],identity='frozen',
        compute=lambda pairs:[torch.tensor([-.1]*len(r)) for p,r in pairs],device='cpu')
    assert [len(v) for v in values]==[1,2]
    assert cache.snapshot()['entries']==1 and cache.snapshot()['evictions']==1
    assert all(not v.requires_grad for v in values)


def test_cached_reference_preserves_opposing_advantages_kl_and_parameter_update():
    import torch
    from scripts.pr96_earho_contract import token_logps_batch
    from mechet.external_route_rl import optimize_routes,DR_GRPO_ALGORITHM
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(7,4);self.head=torch.nn.Linear(4,7);self.observed_rows=0
        def forward(self,input_ids,attention_mask,logits_to_keep,use_cache,position_ids=None):
            self.observed_rows+=len(input_ids)
            hidden=(self.emb(input_ids)*attention_mask[...,None]).cumsum(1)
            return SimpleNamespace(logits=self.head(hidden[:,logits_to_keep]))
    torch.manual_seed(31);initial=Tiny();reference=deepcopy(initial)
    with torch.no_grad():reference.head.bias[0]+=.2
    pairs=[([1],[2,3]),([1],[2,3]),([4],[5]),([4],[5])]
    old=[v.detach().tolist() for v in token_logps_batch(initial,pairs)]
    reports=[dict(reward=reward,records=[dict(prompt_token_ids=p,generated_token_ids=r,behavior_logprobs=lp)])
             for (p,r),lp,reward in zip(pairs,old,[1.,0.,0.,0.])]
    cfg=dict(algorithm=DR_GRPO_ALGORITHM,candidates=4,loss_batch_size=4,pad_token_id=0,
        clip_epsilon=.2,kl_beta=.02,loss_token_normalizer=1024.,max_grad_norm=1.,behavior_logprob_tolerance=.25,
        parent_sha256='frozen-reference',base_manifest='frozen-tokenizer')
    plain=deepcopy(initial);cached=deepcopy(initial);ref1=deepcopy(reference);ref2=deepcopy(reference)
    ref1.observed_rows=ref2.observed_rows=0
    a=optimize_routes(plain,ref1,torch.optim.SGD(plain.parameters(),lr=.2),[deepcopy(reports)],cfg)
    b=optimize_routes(cached,ref2,torch.optim.SGD(cached.parameters(),lr=.2),[deepcopy(reports)],
        dict(cfg,acceleration={'reference_cache_capacity':8}))
    assert ref2.observed_rows==2<ref1.observed_rows
    for p,q in zip(plain.parameters(),cached.parameters()):torch.testing.assert_close(p,q,rtol=2e-6,atol=2e-7)
    assert a['reference_kl']==pytest.approx(b['reference_kl'],rel=2e-6,abs=2e-7)
    assert b['reference_cache']['physical_sequences']==2
