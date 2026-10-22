"""Pinned Qwen3.5 text-only tool-decision rendering and exact selective CE.

The official template owns the non-thinking prefix. The model still owns the
entire XML tool call, function choice, parameters and terminating assistant EOS.
"""
from mechet.assistant_masking import render_chat, tokenize_text

OFFICIAL_MODEL = "Qwen/Qwen3.5-0.8B"
OFFICIAL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                "gate_proj", "up_proj", "down_proj"]


def tool_prefix(tokenizer, messages, tools):
    if not tools or not messages or messages[-1].get("role") != "user":
        raise ValueError("QWEN35_EXPECTS_CURRENT_USER_STATE_AND_TOOLS")
    # Unlike the old Qwen3 helper, the official Qwen3.5 template contains its
    # own non-thinking prefix. Never prefill tool/function names with a GT hint.
    return render_chat(tokenizer, list(messages), tools=list(tools), add_generation_prompt=True)


def encode_tool_decision(tokenizer, row, *, max_length):
    messages = list(row["messages"])
    indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(indices) != 1:
        raise ValueError("EXPECTED_ONE_ASSISTANT_TOOL_DECISION")
    index = indices[0]
    assistant = messages[index]
    if assistant.get("content") or len(assistant.get("tool_calls", [])) != 1:
        raise ValueError("EXPECTED_ONE_EMPTY_CONTENT_TOOL_CALL")
    prefix = tool_prefix(tokenizer, messages[:index], row["tools"])
    # Future tool-result tokens carry no supervised loss or causal information.
    # Omit them rather than padding compute with a private executor response.
    full = render_chat(tokenizer, messages[:index + 1], tools=row["tools"])
    if not full.startswith(prefix) or not full[len(prefix):].startswith("<tool_call>"):
        raise ValueError("QWEN35_TEMPLATE_PREFIX_PARITY_FAILED")
    ids = tokenize_text(tokenizer, full)
    prompt_ids = tokenize_text(tokenizer, prefix)
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("QWEN35_TOKEN_PREFIX_PARITY_FAILED")
    eos_ids = tokenize_text(tokenizer, "<|im_end|>")
    if len(eos_ids) != 1:
        raise ValueError("QWEN35_EOS_NOT_SINGLE_TOKEN")
    start = len(prompt_ids)
    try:
        end = ids.index(eos_ids[0], start) + 1
    except ValueError as exc:
        raise ValueError("QWEN35_TOOL_EOS_MISSING") from exc
    labels = [-100] * len(ids)
    labels[start:end] = ids[start:end]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}, {
        "raw_length": len(ids), "exceeds_max_length": len(ids) > max_length,
        "assistant_turns": 1, "assistant_spans": [[start, end]],
        "supervised_tokens": end - start, "mask_method": "qwen35_owned_xml_tool_span_v1",
        "prefix_token_parity": True, "future_tool_response_tokens": 0,
    }


def supervised_loss(model, inputs, *, num_items_in_batch=None):
    """Exact next-token CE; project vocabulary only at supervised positions.

    No state prediction, reweighting or objective change. The union of needed
    sequence positions works with differently positioned spans across a batch.
    """
    import torch
    from torch.nn import functional as F
    values = dict(inputs)
    labels = values.pop("labels")
    positions = (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=True)[0]
    if not positions.numel():
        raise ValueError("ZERO_SUPERVISION")
    values.pop("num_items_in_batch", None)
    output = model(**values, logits_to_keep=positions, use_cache=False)
    selected = labels[:, positions + 1]
    logits = output.logits
    active = selected != -100
    loss = F.cross_entropy(logits[active].float(), selected[active], reduction="sum")
    denominator = active.sum() if num_items_in_batch is None else num_items_in_batch
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.to(loss.device)
    return loss / denominator, output


def load_language_model(path, *, dtype, local_files_only=True, revision=OFFICIAL_REVISION):
    """Use the official text class and its registered checkpoint key mapping."""
    from transformers import AutoConfig, Qwen3_5ForCausalLM
    if revision != OFFICIAL_REVISION:
        raise ValueError("QWEN35_OFFICIAL_REVISION_REQUIRED")
    full = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=False, local_files_only=local_files_only)
    config = full.get_text_config()
    if config.model_type != "qwen3_5_text" or config.hidden_size != 1024 or config.num_hidden_layers != 24:
        raise ValueError("NOT_OFFICIAL_QWEN35_08B_TEXT_ARCHITECTURE")
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        path, config=config, revision=revision, dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=False, local_files_only=local_files_only, output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError(f"QWEN35_INCOMPLETE_TEXT_WEIGHT_LOAD:{info}")
    unexpected = [k for k in info.get("unexpected_keys", [])
                  if not k.startswith(("model.visual.", "mtp."))]
    if unexpected or any("visual" in n for n, _ in model.named_parameters()):
        raise ValueError(f"QWEN35_LANGUAGE_ONLY_LOAD_FAILED:{unexpected}")
    return model, info
