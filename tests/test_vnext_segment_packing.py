import torch

from mechet.segment_packing import block_causal_inputs, pack_encoded_rows


def test_packing_keeps_assistant_labels_and_blocks_cross_reaction_attention():
    rows = [
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
        {"input_ids": [6, 7, 8], "labels": [-100, 7, 8]},
    ]
    packs = pack_encoded_rows(rows, max_length=5)
    assert [pack["segments"] for pack in packs] == [[3, 2], [3]]
    first = block_causal_inputs(packs[0], device="cpu", dtype=torch.float32)
    assert first["labels"].tolist() == [[-100, -100, 3, -100, 5]]
    assert first["position_ids"].tolist() == [[0, 1, 2, 0, 1]]
    mask = first["attention_mask"][0, 0]
    assert mask[4, 3] == 0 and mask[4, 2] < -1e20
    assert mask[2, 0] == 0 and mask[2, 4] < -1e20


def test_qwen3_segment_packed_logits_match_independent_forward():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(5)
    config = Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64,
        attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(config).eval()
    rows = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
    ]
    pack = pack_encoded_rows(rows, max_length=8)[0]
    with torch.no_grad():
        got = model(**block_causal_inputs(pack, device="cpu", dtype=torch.float32)).logits[0]
        first = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits[0]
        second = model(input_ids=torch.tensor([[5, 6, 7]])).logits[0]
    assert torch.allclose(got[:4], first, atol=1e-5)
    assert torch.allclose(got[4:], second, atol=1e-5)
