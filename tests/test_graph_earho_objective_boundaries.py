"""Analytical objective checks on tiny tensors; no model sampling/training."""
import torch

from mechet.graph_earho import VerifiedOutcome, successor_group_advantages
from mechet.graph_endpoint_rl import successor_class_policy_loss, truncated_slate_reference_kl


def test_local_support_surrogate_can_reduce_actual_endpoint_probability():
    # Action 0 matches a local label but fails later; action 1 reaches the
    # endpoint beyond H; action 2 is invalid. True endpoint probability=.5.
    logits = torch.tensor([0.4, 0.5, 0.1], dtype=torch.float64).log().requires_grad_()
    outcomes = [VerifiedOutcome(True, reference_successor=True),
                VerifiedOutcome(True, exhausted=True), VerifiedOutcome(False)]
    adv = successor_group_advantages(outcomes)
    loss = successor_class_policy_loss(list(logits.log_softmax(0)),
                                       [(0,), (1,), (2,)], adv, None)
    gradient, = torch.autograd.grad(loss, logits)
    new_probability = (logits.detach() - 0.01 * gradient).softmax(0)[1]
    assert new_probability < 0.5  # concrete counterexample to endpoint improvement
    assert successor_group_advantages(outcomes, endpoint_only=True) == (0.0, 0.0, 0.0)


def test_slate_kl_does_not_bound_probability_leaving_the_sampled_slate():
    old = torch.tensor([0.4, 0.4], dtype=torch.float64).log()
    changed = torch.tensor([0.1, 0.1], dtype=torch.float64).log()
    assert abs(truncated_slate_reference_kl(list(changed), list(old)).item()) < 1e-12
    # Unsampled probability moved .2 -> .8 even though conditional slate KL=0.
    full_old = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float64)
    full_new = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float64)
    assert (full_old * (full_old.log() - full_new.log())).sum() > 0.8
