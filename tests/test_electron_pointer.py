import pytest
import torch

from mechet.electron_pointer import (
    UnsupportedPointerEvent,
    candidate_keys,
    parse_pointer_example,
    parse_pointer_observation,
    target_indices,
)
from mechet.electron_pointer_model import (
    CoupledPointerHead, PointerHead, coupled_pair_recall_metrics,
    pair_recall_metrics, pairs_for_observation,
)
from mechet.electron_pointer_runtime import action_pointer_log_likelihood_ratio


def test_multimove_pointer_targets_and_candidate_universe():
    row = {
        "id": "example::event", "tools": [],
        "messages": [
            {"role": "user", "content": "ANNOTATED CURRENT STATE: <A01>C<A02>O.<A03>[Cl-]"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "apply_electron_flow", "arguments": {
                "electron_flow": [
                    {"source": "a lone pair on atom A03", "destination": "the bond to form between atoms A01 and A03"},
                    {"source": "the bond between atoms A01 and A02", "destination": "atom A02"},
                ]
            }}}]},
        ],
    }
    example = parse_pointer_example(row)
    assert example is not None
    assert example.assistant_message["role"] == "assistant"
    assert len(example.atom_names) == 3
    assert example.bonds == ((0, 1),)
    source = candidate_keys(3, example.bonds, source=True)
    sink = candidate_keys(3, example.bonds, source=False)
    assert [source[i] for i in target_indices(example, source=True)] == [
        ("atom", 2, 2), ("bond", 0, 1)
    ]
    assert [sink[i] for i in target_indices(example, source=False)] == [
        ("atom", 1, 1), ("bond", 0, 2)
    ]


def test_import_has_no_pointer_target():
    row = {"id": "example::import", "messages": [
        {"role": "assistant", "tool_calls": [{"function": {"name": "import_fragments", "arguments": {"fragments": []}}}]}
    ]}
    assert parse_pointer_example(row) is None


def test_runtime_pointer_uses_only_visible_inventory_and_rejects_bad_handle():
    obs = parse_pointer_observation("ANNOTATED CURRENT STATE: <A01>C<A02>O.<A03>[Cl-]")
    source = torch.tensor([0.0, 0.0, 4.0, 0.0])
    sink = torch.tensor([0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
    correct = {"electron_flow": [{"source": "a lone pair on atom A03",
                                 "destination": "the bond to form between atoms A01 and A03"}]}
    assert action_pointer_log_likelihood_ratio(source, sink, obs, "apply_electron_flow", correct) > 0
    assert action_pointer_log_likelihood_ratio(source, sink, obs, "finish_trace", {}) == 0
    wrong = {"electron_flow": [{"source": "a lone pair on atom A99", "destination": "atom A02"}]}
    assert action_pointer_log_likelihood_ratio(source, sink, obs, "apply_electron_flow", wrong) == float("-inf")


def test_pointer_head_backward_and_candidate_order():
    obs = parse_pointer_observation("ANNOTATED CURRENT STATE: <A01>C<A02>O.<A03>[Cl-]")
    src, sink = pairs_for_observation(obs, torch.device("cpu"))
    assert src.tolist() == [[0, 1]]
    assert sink.tolist() == [[0, 1], [0, 2], [1, 2]]
    head = PointerHead(12, width=8)
    source_logits, sink_logits = head(torch.randn(12), torch.randn(3, 12), src, sink)
    assert source_logits.shape == (4,) and sink_logits.shape == (6,)
    (source_logits[0] + sink_logits[1]).backward()
    assert head.atom.weight.grad is not None


def test_pair_recall_does_not_confuse_independent_site_recall():
    source = torch.tensor([3., 2.])
    sink = torch.tensor([3., 2.])
    # Both marginal gold sites occur in their top two, but coupled gold is
    # ranked fourth by an independent pointer head.
    metrics = pair_recall_metrics(source, sink, [1], [1])
    assert metrics["pair_r1"] == 0
    assert metrics["pair_r4"] == 1
    assert metrics["pair_all_r4"] == 1


def test_conditional_pair_head_trains_and_scores_coupled_moves():
    obs = parse_pointer_observation("ANNOTATED CURRENT STATE: <A01>C<A02>O.<A03>[Cl-]")
    src, sink = pairs_for_observation(obs, torch.device("cpu"))
    head = CoupledPointerHead(12, width=8)
    source_logits, sink_logits, pair_logits = head(
        torch.randn(12), torch.randn(3, 12), src, sink
    )
    assert pair_logits.shape == (source_logits.numel(), sink_logits.numel())
    pair_logits[2, 4].backward()
    assert head.pair_source.weight.grad is not None
    assert coupled_pair_recall_metrics(torch.tensor([[0., 1.], [3., 2.]]), [1], [0])["pair_r1"] == 1


def test_pointer_pilot_excludes_valid_delta_only_event_explicitly():
    row = {
        "id": "example::delta", "tools": [],
        "messages": [
            {"role": "user", "content": "ANNOTATED CURRENT STATE: <A01>C<A02>O"},
            {"role": "assistant", "tool_calls": [{"function": {
                "name": "apply_electron_flow",
                "arguments": {"electron_flow": [], "bond_order_changes": [{"atoms": ["A01", "A02"], "delta": -1}]},
            }}]},
        ],
    }
    with pytest.raises(UnsupportedPointerEvent, match="no source/sink"):
        parse_pointer_example(row)


def test_pointer_pilot_excludes_radical_pair_explicitly():
    row = {
        "id": "example::radical", "tools": [],
        "messages": [
            {"role": "user", "content": "ANNOTATED CURRENT STATE: <A01>C<A02>O"},
            {"role": "assistant", "tool_calls": [{"function": {
                "name": "apply_electron_flow",
                "arguments": {"electron_flow": [{
                    "source": "a radical pair on atoms A01 and A02",
                    "destination": "atom A02",
                }]},
            }}]},
        ],
    }
    with pytest.raises(UnsupportedPointerEvent, match="radical-pair"):
        parse_pointer_example(row)
