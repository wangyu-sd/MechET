"""Exact finite-probability/autograd checks, not model experiments."""
import itertools
import math
import random

import pytest
import torch

from mechet import graph_endpoint_rl as objective


def require(name):
    value = getattr(objective, name, None)
    if value is None:
        pytest.fail(f"missing conditional endpoint primitive: {name}")
    return value


def test_normalization_uses_the_whole_fixed_support_and_temperature():
    normalize = require("slate_log_probabilities")
    scores = torch.tensor([math.log(.2), math.log(.3), math.log(.5)], dtype=torch.float64)
    torch.testing.assert_close(normalize(scores).exp(), torch.tensor([.2, .3, .5], dtype=torch.float64))
    weights = torch.tensor([math.sqrt(.2), math.sqrt(.3), math.sqrt(.5)], dtype=torch.float64)
    torch.testing.assert_close(normalize(scores, temperature=2).exp(), weights / weights.sum())
    torch.testing.assert_close(normalize(scores + 1000), normalize(scores))


def test_iid_draws_allow_duplicates_even_when_draw_count_exceeds_support():
    draw = require("iid_slate_indices")
    class FixedRandom:
        def __init__(self):
            self.values = iter([0.0, .01, .75, .99])
        def random(self):
            return next(self.values)
    assert draw(torch.tensor([math.log(.5), math.log(.5)]), 4, FixedRandom()) == (0, 0, 1, 1)


def test_leave_one_out_does_not_include_own_return():
    loo = require("leave_one_out_advantages")
    torch.testing.assert_close(loo([1., 0., 0.]), torch.tensor([1., -.5, -.5]))
    torch.testing.assert_close(loo([1., 1.]), torch.zeros(2))
    torch.testing.assert_close(loo([0., 0.]), torch.zeros(2))
    with pytest.raises(ValueError):
        loo([1.])


def test_expected_sample_gradient_equals_exact_conditional_endpoint_gradient():
    loss_fn = require("conditional_endpoint_policy_loss")
    logits = torch.tensor([math.log(.2), math.log(.3), math.log(.5)], dtype=torch.float64,
                          requires_grad=True)
    expected_gradient = torch.zeros(3, dtype=torch.float64)
    probabilities = [.2, .3, .5]
    returns = [0., 1., 0.]
    # Exhaust every possible independent two-draw batch, including duplicates.
    for left, right in itertools.product(range(3), repeat=2):
        loss = loss_fn(logits, [left, right], [returns[left], returns[right]])
        gradient, = torch.autograd.grad(loss, logits)
        expected_gradient += probabilities[left] * probabilities[right] * gradient
    torch.testing.assert_close(expected_gradient,
                               torch.tensor([.06, -.21, .15], dtype=torch.float64))


def test_duplicate_samples_keep_their_sampling_multiplicity():
    loss_fn = require("conditional_endpoint_policy_loss")
    logits = torch.tensor([0., 0.], dtype=torch.float64, requires_grad=True)
    loss = loss_fn(logits, [0, 0, 1], [1., 1., 0.])
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, torch.tensor([-1/3, 1/3], dtype=torch.float64))


def test_short_horizon_never_becomes_endpoint_actor_update_even_with_a_hit():
    loss_fn = require("conditional_endpoint_policy_loss")
    logits = torch.tensor([0., 0.], requires_grad=True)
    loss = loss_fn(logits, [0, 1], [1., 0.], actor_enabled=False)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, torch.zeros(2))
    assert loss.item() == 0.


def test_sampler_rejects_unnormalized_or_nonfinite_behavior_probabilities():
    draw = require("iid_slate_indices")
    with pytest.raises(ValueError):
        draw(torch.tensor([0., 0.]), 2, random.Random(17))
    with pytest.raises(ValueError):
        draw(torch.tensor([float('nan')]), 2, random.Random(17))
