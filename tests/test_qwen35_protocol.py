"""Actual pinned tokenizer parity plus exact assistant-only loss math."""
import os

import pytest
import torch


@pytest.fixture
def tokenizer():
    path = os.environ.get("MECHET_QWEN35_SNAPSHOT")
    if not path:
        pytest.skip("set MECHET_QWEN35_SNAPSHOT to the pinned official tokenizer snapshot")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize("name,args", [
    ("import_fragments", {"fragments": [{"smiles": "[Br-]", "count": 1, "purpose": "electron_participant"}]}),
    ("finish_trace", {}),
])
def test_official_tool_template_prefix_is_token_identical_and_only_xml_is_supervised(tokenizer, name, args):
    from mechet.qwen35_protocol import encode_tool_decision, tool_prefix
    from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
    from mechet.agent_inference import parse_tool_calls
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "CURRENT STATE SMILES: CO"}]
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": args}}]}
    row = {"messages": messages + [assistant, {"role": "tool", "name": name, "content": "PRIVATE_RESULT_NOT_TO_PREDICT"}], "tools": TOOLS}
    encoded, audit = encode_tool_decision(tokenizer, row, max_length=8192)
    supervised = tokenizer.decode([t for t,l in zip(encoded["input_ids"], encoded["labels"]) if l != -100])
    assert supervised.startswith("<tool_call>")
    assert "<think>" not in supervised and "PRIVATE_RESULT" not in supervised
    assert "PRIVATE_RESULT" not in tokenizer.decode(encoded["input_ids"])
    calls = parse_tool_calls(supervised)
    assert [(c.name, c.arguments) for c in calls] == [(name, args)]
    prefix_ids = tokenizer(tool_prefix(tokenizer, messages, TOOLS), add_special_tokens=False)["input_ids"]
    assert encoded["input_ids"][:len(prefix_ids)] == prefix_ids
    assert all(v == -100 for v in encoded["labels"][:len(prefix_ids)])
    assert audit["supervised_tokens"] > 0 and not audit["exceeds_max_length"]


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 7)
        self.head = torch.nn.Linear(7, 17)

    def forward(self, input_ids, logits_to_keep, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)[:, logits_to_keep, :]))


def test_runtime_rejects_triton_that_breaks_pinned_torch_inductor_imports():
    from mechet.qwen35_protocol import validate_runtime_versions
    validate_runtime_versions(torch_version="2.6.0+cu118", cuda_version="11.8", triton_version="3.2.0")
    with pytest.raises(ValueError, match="TORCH_TRITON"):
        validate_runtime_versions(torch_version="2.6.0+cu118", cuda_version="11.8", triton_version="3.3.1")


def test_selective_loss_and_gradients_match_full_causal_loss_with_different_spans_per_row():
    from mechet.qwen35_protocol import supervised_loss
    torch.manual_seed(17)
    model = TinyCausalModel()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    labels = torch.tensor([[-100, -100, 3, 4, -100, -100], [-100, -100, -100, 3, 2, 1]])
    logits = model(ids, slice(None)).logits
    dense = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1))
    dense.backward()
    grads = {n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad()
    loss, _ = supervised_loss(model, {"input_ids": ids, "labels": labels})
    loss.backward()
    assert torch.allclose(loss, dense, atol=1e-7)
    for n,p in model.named_parameters():
        assert torch.allclose(p.grad, grads[n], atol=1e-7), n


def test_selective_loss_fails_closed_on_zero_supervision():
    from mechet.qwen35_protocol import supervised_loss
    with pytest.raises(ValueError, match="ZERO_SUPERVISION"):
        supervised_loss(TinyCausalModel(), {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[-100, -100]])})
