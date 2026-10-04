import torch

from mechet.system_one_decision import ElectronFlowDecisionHead, OptionPointerHead, multi_target_nll


def test_option_pointer_shape():
    head = OptionPointerHead(16, 8)
    logits = head(torch.randn(16), torch.randn(7, 16))
    assert logits.shape == (7,)


def test_electron_flow_decision_shapes():
    head = ElectronFlowDecisionHead(16, 8)
    atom = torch.randn(4, 16)
    src_pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    sink_pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long)
    out = head(torch.randn(16), atom, src_pairs, sink_pairs)
    assert out.source_logits.shape == (6,)
    assert out.sink_logits.shape == (10,)
    assert out.pair_logits.shape == (6, 10)


def test_multi_target_mass_beats_single_wrong_target():
    logits = torch.tensor([0.0, 4.0, 4.0, -2.0])
    good = multi_target_nll(logits, [1, 2])
    wrong = multi_target_nll(logits, [0])
    assert good < wrong
