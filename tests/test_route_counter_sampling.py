import pytest


def test_counter_sampler_preserves_distribution_and_selected_original_logprob():
    import torch
    from mechet.route_counter_sampling import CounterCategorical
    from mechet.route_policy_sampling import TokenLogprobs
    logits=torch.tensor([[0.,1.,-2.],[2.,-.5,0.]])
    capture=TokenLogprobs(99);sample=CounterCategorical(['a','b'],17,'cpu',capture)
    ids=torch.tensor([[1],[1]])
    capture(ids,logits);pinned=sample(ids,logits);chosen=pinned.argmax(-1)
    capture.finish(chosen)
    values=capture.values()
    expected=logits.log_softmax(-1).gather(1,chosen[:,None]).flatten()
    torch.testing.assert_close(torch.tensor([v[0] for v in values]),expected)
    assert torch.isfinite(pinned).sum().item()==2
    reordered=CounterCategorical(['b','a'],17,'cpu',TokenLogprobs(99))
    assert reordered(ids,logits.flip(0)).argmax(-1).flip(0).tolist()==chosen.tolist()


def test_counter_categorical_frequency_and_zero_mass():
    import torch
    from mechet.route_counter_sampling import CounterCategorical
    from mechet.route_policy_sampling import TokenLogprobs
    n=12000;uids=[str(i) for i in range(n)]
    scores=torch.tensor([0.,1.,-float('inf')]).repeat(n,1)
    sampler=CounterCategorical(uids,17,'cpu',TokenLogprobs(99))
    result=sampler(torch.ones(n,1,dtype=torch.long),scores).argmax(-1)
    observed=(result==1).float().mean().item()
    assert abs(observed-torch.softmax(scores[0],-1)[1].item())<.02
    assert (result==2).sum().item()==0


def test_counter_sampler_rejects_invalid_logits():
    import torch
    from mechet.route_counter_sampling import CounterCategorical
    from mechet.route_policy_sampling import TokenLogprobs
    sampler=CounterCategorical(['a'],17,'cpu',TokenLogprobs(99))
    with pytest.raises(ValueError,match='DISTRIBUTION'):
        sampler(torch.tensor([[1]]),torch.tensor([[float('nan'),0.]]))


def test_invalid_distribution_diagnostics_keep_request_and_decode_binding():
    import torch
    from mechet.route_counter_sampling import CounterCategorical
    from mechet.route_policy_sampling import TokenLogprobs
    events=[]
    sampler=CounterCategorical(['real-request'],17,'cpu',TokenLogprobs(99),on_failure=events.append,prompt_width=1)
    with pytest.raises(ValueError):sampler(torch.tensor([[1,2]]),torch.tensor([[float('nan'),0.]]))
    assert events[0]['event']=='policy-distribution-failure'
    assert events[0]['rows'][0]['request_id']=='real-request' and events[0]['decode_step']==0
