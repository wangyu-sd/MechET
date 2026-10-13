from types import SimpleNamespace
import itertools

import pytest
import torch

from mechet.graph_endpoint_rl import conditional_endpoint_policy_loss


def test_import_first_error_is_eligible_and_preserves_actor_state():
    from scripts.train_crn_qwen_joint_earho import select_joint_frontier
    node = SimpleNamespace(state='actual-state')
    episode = [(node, {'accepted': True, 'successor': 'different-context', 'terminal': False})]
    reference = [SimpleNamespace(successor_key='recorded-context', terminal=False)]
    selected, index = select_joint_frontier(episode, reference)
    assert selected is node and index == 0


def test_chain_rule_microbackward_matches_full_joint_support_gradient():
    from mechet.joint_endpoint_policy import conditional_score_gradients
    values = torch.tensor([-1.1, -0.8, -2.3], requires_grad=True)
    expected = torch.autograd.grad(conditional_endpoint_policy_loss(
        list(values), (0, 1, 1), (1., 0., 0.), temperature=0.8,
        actor_enabled=True), values)[0]
    actual, loss = conditional_score_gradients(values.detach(), (0, 1, 1),
        (1., 0., 0.), temperature=0.8, actor_enabled=True, kl_weight=0.)
    assert actual.tolist() == pytest.approx(expected.tolist())
    short, _ = conditional_score_gradients(values.detach(), (0, 1),
        (1., 0.), temperature=1., actor_enabled=False, kl_weight=0.)
    assert short.tolist() == [0., 0., 0.]


def test_joint_expected_gradient_equals_exact_conditional_endpoint_objective():
    from mechet.joint_endpoint_policy import conditional_score_gradients
    # Three delegated actions: import, FLOW, stop. Both policies share route.
    route = torch.tensor([-0.2, -0.7, -0.4], requires_grad=True)
    graph = torch.tensor(-0.3, requires_grad=True)
    scores = torch.stack([route[0]-0.4, route[1]+graph, route[2]])
    q = scores.softmax(0)
    reward = torch.tensor([1., 0., 0.])
    exact = torch.autograd.grad(-(q*reward).sum(), (route, graph), retain_graph=True)
    expected_route = torch.zeros(3)
    expected_graph = 0.
    for draws in itertools.product(range(3), repeat=2):
        coefficients, _ = conditional_score_gradients(scores.detach(), draws,
            tuple(float(reward[i]) for i in draws), temperature=1.,
            actor_enabled=True, kl_weight=0.)
        weight = float(q[draws[0]].detach()*q[draws[1]].detach())
        expected_route += weight*coefficients
        expected_graph += weight*float(coefficients[1])
    assert expected_route.tolist() == pytest.approx(exact[0].tolist(), abs=1e-6)
    assert expected_graph == pytest.approx(exact[1].item(), abs=1e-6)


def test_full_follower_failure_is_zero_not_dropped_and_imports_interleave():
    from scripts.train_crn_qwen_joint_earho import endpoint_return, import_participation
    assert endpoint_return(None, {}, '[C:1]') == 0
    from scripts.run_natural_language_value_search import Node
    node = Node('C', '[CH3:1][OH:2]', 3, actions=[
        {'name': 'import_fragments', 'state_before': '[CH4:1]',
         'arguments': {'fragments':[{'smiles':'O', 'count':1, 'purpose':'context'}]},
         'result': {}},
        {'name': 'apply_electron_flow', 'state_before':'[CH4:1].[OH2:2]',
         'arguments': {}, 'result': {}},
    ])
    # Advisory context labels do not prevent the imported atom participating.
    audit = import_participation(node, [{2}])
    assert audit[0]['declared_purposes'] == ['context']
    assert audit[0]['used_in_flow'] is True
