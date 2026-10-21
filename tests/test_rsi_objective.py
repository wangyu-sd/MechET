import torch


def test_dpo_update_improves_preferred_logprob_without_reference_gradient():
    from mechet.rsi_objective import dpo_loss
    theta=torch.nn.Parameter(torch.tensor(0.));ref=torch.tensor(0.,requires_grad=True)
    optimizer=torch.optim.SGD([theta],lr=1.)
    loss=dpo_loss(theta,-theta,ref,beta=.1)
    loss.backward();assert ref.grad is None
    optimizer.step();assert theta.item()>0
    assert torch.allclose(loss.detach(),torch.tensor(.6931472))


def test_action_logp_mask_includes_eos_but_not_prompt():
    from mechet.rsi_objective import action_inputs
    ids,mask=action_inputs([4,5],[6,7,151645],device='cpu')
    assert ids.tolist()==[[4,5,6,7,151645]]
    assert mask.tolist()==[[0,0,1,1,1]]
