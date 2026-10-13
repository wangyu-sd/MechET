import math
from types import SimpleNamespace

import pytest
import torch


def test_suffix_probability_includes_eos_and_sums_not_means():
    from mechet.joint_endpoint_policy import qwen_continuation_log_probability
    class UniformLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3))
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.logits.expand(1, input_ids.shape[1], 3))
    model = UniformLM()
    score = qwen_continuation_log_probability(model, (0, 1), (2, 0))
    assert score.item() == pytest.approx(-2 * math.log(3))
    score.backward()
    assert model.logits.grad.tolist() == pytest.approx([1/3, -2/3, 1/3])


def test_delegated_flow_score_reaches_both_policies():
    from mechet.joint_endpoint_policy import JointProposal, joint_action_log_probability
    route = torch.tensor(-0.3, requires_grad=True)
    event = torch.tensor(-1.2, requires_grad=True)
    proposal = JointProposal('apply_electron_flow', {}, (1, 2), '<route>')
    result = joint_action_log_probability(proposal, route, event)
    assert result.item() == pytest.approx(-1.5)
    result.backward()
    assert route.grad.item() == event.grad.item() == 1
    with pytest.raises(ValueError, match='graph'):
        joint_action_log_probability(proposal, route)


def test_invalid_import_remains_scoreable_and_dedup_is_token_path_specific():
    from mechet.joint_endpoint_policy import JointProposal, freeze_joint_support
    a = JointProposal('import_fragments', {}, (1, 2), '{bad', error='TRUNCATED')
    b = JointProposal('import_fragments', {}, (1, 3), '{bad', error='TRUNCATED')
    assert freeze_joint_support([a, a, b]) == (a, b)


def test_tiny_qwen_generation_teacher_score_parity():
    pytest.importorskip('transformers')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from mechet.joint_endpoint_policy import qwen_continuation_log_probability
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, pad_token_id=0, eos_token_id=31,
    )).eval()
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=3, do_sample=False,
            return_dict_in_generate=True, output_scores=True)
        expected = model.compute_transition_scores(generated.sequences,
            generated.scores, normalize_logits=True).sum()
    actual = qwen_continuation_log_probability(model, (1, 2, 3),
        tuple(generated.sequences[0, 3:].tolist()))
    assert actual.item() == pytest.approx(expected.item(), abs=1e-5)


def test_forced_import_prefix_counts_toward_same_decoder_budget():
    from mechet.joint_endpoint_policy import import_generation_budget
    assert import_generation_budget(3700, 15, max_context=4096, max_tokens=384) == 369
    with pytest.raises(ValueError, match='CONTEXT'):
        import_generation_budget(3720, 15, max_context=4096, max_tokens=384)
    with pytest.raises(ValueError, match='prefix'):
        import_generation_budget(100, 385, max_context=4096, max_tokens=384)
