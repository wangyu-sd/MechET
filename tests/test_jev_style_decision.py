import torch
from types import SimpleNamespace

from mechet.electron_pointer import candidate_keys
from mechet.jev_style_decision import (
    FactorizedTypedElectronFlowHead,
    JevEncoding,
    TypedElectronFlowHead,
    TypedQuestion,
    block_causal_option_mask,
    encode_typed_record,
    required_target_nll,
)
from scripts.train_jev_style_electron_flow import prepare_typed


class FakeTokenizer:
    unk_token_id = -1

    def __init__(self):
        self.controls = {
            "<|fim_prefix|>": 1001,
            "<|fim_middle|>": 1002,
            "<|box_start|>": 1003,
            "<|box_end|>": 1004,
            "<|fim_suffix|>": 1005,
        }

    def convert_tokens_to_ids(self, token):
        return self.controls.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        if text in self.controls:
            return {"input_ids": [self.controls[text]]}
        return {"input_ids": [10 + (ord(ch) % 200) for ch in text]}


def test_typed_encoding_has_shared_state_and_two_questions():
    enc = encode_typed_record(
        FakeTokenizer(),
        state="CURRENT STATE",
        questions=(
            TypedQuestion("SOURCE", ("A01", "A02")),
            TypedQuestion("SINK", ("A03", "A04")),
        ),
    )
    assert len(enc.decide_indices) == 2
    assert [len(x) for x in enc.option_indices] == [2, 2]
    assert set(enc.segment_ids) == {0, 1, 2}


def test_option_isolation_and_question_isolation_mask():
    enc = encode_typed_record(
        FakeTokenizer(),
        state="STATE",
        questions=(
            TypedQuestion("SOURCE", ("A01", "A02")),
            TypedQuestion("SINK", ("A03", "A04")),
        ),
    )
    mask = block_causal_option_mask(enc, device=torch.device("cpu"), dtype=torch.float32)[0, 0]
    source_first, source_second = enc.option_indices[0]
    source_decide, sink_decide = enc.decide_indices

    # Later source option cannot read the earlier sibling option.
    assert mask[source_second, source_first] < -1e20
    # Decide can aggregate all options in its own question.
    assert mask[source_decide, source_first] == 0
    assert mask[source_decide, source_second] == 0
    # Sink question cannot read the source question branch.
    assert mask[sink_decide, source_decide] < -1e20
    # Every question can read the shared state.
    assert mask[sink_decide, 0] == 0


def test_typed_pair_head_shapes():
    head = TypedElectronFlowHead(16, 8)
    out = head(
        torch.randn(16),
        torch.randn(5, 16),
        torch.randn(16),
        torch.randn(7, 16),
    )
    assert out.source_logits.shape == (5,)
    assert out.sink_logits.shape == (7,)
    assert out.pair_logits.shape == (5, 7)


def test_factorized_sink_retains_all_atom_pairs_with_linear_text_options():
    n_atoms = 85
    example = SimpleNamespace(
        row_id="test::event",
        atom_names=tuple(f"A{i + 1:02d}" for i in range(n_atoms)),
        bonds=(),
        source_targets=(("atom", 0, 0),),
        sink_targets=(("bond", 0, n_atoms - 1),),
        messages=[{"role": "user", "content": "CURRENT STATE"}],
    )
    prepared = prepare_typed(example, FakeTokenizer())
    assert prepared.explicit_sink_options == n_atoms
    assert len(prepared.encoding.option_indices[1]) == n_atoms
    assert prepared.sink_count == n_atoms + n_atoms * (n_atoms - 1) // 2
    assert prepared.pair_targets == (2 * n_atoms - 2,)
    assert len(prepared.encoding.input_ids) < 3000


def test_factorized_head_scores_full_pair_space_and_backpropagates():
    head = FactorizedTypedElectronFlowHead(16, 8)
    source = torch.randn(5, 16, requires_grad=True)
    sink_atoms = torch.randn(4, 16, requires_grad=True)
    output = head(torch.randn(16), source, torch.randn(16), sink_atoms)
    assert output.source_logits.shape == (5,)
    assert output.sink_logits.shape == (4 + 6,)
    assert output.pair_logits.shape == (5, 10)
    required_target_nll(output.pair_logits, [5, 7]).backward()
    assert sink_atoms.grad is not None and torch.isfinite(sink_atoms.grad).all()


def test_factorized_pair_order_matches_executor_candidate_indices():
    pairs = torch.triu_indices(4, 4, offset=1).T.tolist()
    expected = [(left, right) for _, left, right in
                candidate_keys(4, (), source=False)[4:]]
    assert [tuple(pair) for pair in pairs] == expected


def test_required_target_nll_penalizes_missing_required_flow():
    balanced = torch.tensor([0.0, 4.0, 4.0])
    concentrated = torch.tensor([0.0, 8.0, -8.0])
    assert required_target_nll(balanced, [1, 2]) < required_target_nll(concentrated, [1, 2])
