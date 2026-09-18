from types import SimpleNamespace

import torch

from mechet.rlvr import (
    EventPolicySample,
    discounted_event_returns,
    event_level_policy_loss,
    event_rloo_advantages,
)


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size=16, width=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.output = torch.nn.Linear(width, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def test_discounted_returns_assign_terminal_credit_backward():
    returns = discounted_event_returns([0.1, 0.0, 4.0], gamma=0.95)
    assert returns[-1] == 4.0
    assert returns[0] > 3.0
    rejected = discounted_event_returns([-0.25, -0.75], gamma=0.95)
    assert all(value < 0 for value in rejected)


def test_event_rloo_is_within_start_group_and_keeps_rejected_spans():
    returns, advantages = event_rloo_advantages(
        [[0.1, 4.0], [-0.25, -0.75]], gamma=0.95
    )
    assert len(returns) == len(advantages) == 2
    assert len(advantages[0]) == len(advantages[1]) == 2
    assert advantages[0][0] > 0
    assert advantages[1][0] < 0
    assert abs(advantages[0][0] + advantages[1][0]) < 1e-8


def test_event_policy_loss_is_finite_updates_parameters_and_counts_rejections():
    torch.manual_seed(17)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    samples = [
        EventPolicySample((1, 2, 3), (4, 5), 1.0, -3.0, rejected=False),
        EventPolicySample((1, 2, 3), (6, 7), -1.0, -3.0, rejected=True),
    ]
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss, stats = event_level_policy_loss(model, samples, beta_kl=0.01)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert stats["event_terms"] == 2
    assert stats["rejected_event_terms"] == 1
    assert stats["mean_kl"] >= 0
