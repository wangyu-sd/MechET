import pytest
import torch

from mechet.group_policy_objective import clipped_group_policy_loss


def test_sequence_ratio_differs_from_token_ratio_and_has_gradient():
    new = torch.tensor([[0.2, -0.1, 0.0]], requires_grad=True)
    old = torch.zeros_like(new)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    advantage = torch.tensor([1.0])
    token = clipped_group_policy_loss(new, old, mask, advantage, mode="token")
    sequence = clipped_group_policy_loss(new, old, mask, advantage, mode="sequence")
    assert not torch.allclose(token, sequence)
    sequence.backward()
    assert new.grad is not None and new.grad[0, 2] == 0
    assert torch.isfinite(new.grad).all()


def test_all_negative_zero_advantage_has_no_policy_gradient():
    new = torch.tensor([[0.1, -0.2]], requires_grad=True)
    old = torch.zeros_like(new)
    loss = clipped_group_policy_loss(new, old, torch.ones_like(new), torch.tensor([0.0]), mode="sequence")
    loss.backward()
    assert torch.equal(new.grad, torch.zeros_like(new))


def test_bad_mask_and_mode_fail_closed():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="align"):
        clipped_group_policy_loss(x, x[:, :1], x, torch.ones(1), mode="token")
    with pytest.raises(ValueError, match="mode"):
        clipped_group_policy_loss(x, x, x, torch.ones(1), mode="unknown")


def test_token_mode_matches_historical_grpo_formula_exactly():
    new = torch.tensor([[0.2, -0.3, 0.1]], requires_grad=True)
    old = torch.tensor([[0.0, -0.1, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    advantage = torch.tensor([0.7])
    ref = torch.tensor([[0.1, -0.25, 0.0]])
    observed = clipped_group_policy_loss(
        new, old, mask, advantage, mode="token",
        reference_logps=ref, reference_weight=0.01,
    )
    ratio = torch.exp(new - old)
    objective = torch.minimum(
        ratio * advantage[:, None],
        ratio.clamp(0.8, 1.2) * advantage[:, None],
    )
    historical = ((-objective + 0.01 * (ref - new).square()) * mask).sum() / mask.sum()
    assert torch.allclose(observed, historical)
