import pytest

from mechet.assistant_masking import render_chat, render_qwen_sft_aligned_prefix


class ThinkingTemplate:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=True, tools=None
    ):
        assert not tokenize
        history = "".join(
            f"<|im_start|>{message['role']}\n"
            + ("<think>\n\n</think>\n\n" if message["role"] == "assistant" else "")
            + message["content"] + "<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            return history + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return history


def test_generation_prefix_matches_full_sft_tool_call_boundary():
    tok = ThinkingTemplate()
    history = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    completion = {"role": "assistant", "content": "<tool_call>flow</tool_call>"}
    full = render_chat(tok, history + [completion], tools=[{"name": "flow"}])
    aligned = render_qwen_sft_aligned_prefix(tok, history, tools=[{"name": "flow"}])
    generated = render_chat(tok, history, tools=[{"name": "flow"}], add_generation_prompt=True)
    assert full.startswith(aligned)
    assert aligned == generated
    assert aligned.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_real_qwen3_prefix_matches_tool_and_value_sft_when_available():
    from pathlib import Path

    from transformers import AutoTokenizer

    snapshot = Path(
        "/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache/"
        "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )
    if not snapshot.is_dir():
        pytest.skip("local Qwen3-8B tokenizer snapshot is unavailable")
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
    history = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    tools = [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object"}}}]
    completions = [
        ({"role": "assistant", "content": "P"}, None),
        ({"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "foo", "arguments": {"a": 1}}}
        ]}, tools),
    ]
    for assistant, schemas in completions:
        prefix = render_qwen_sft_aligned_prefix(tok, history, tools=schemas)
        full = render_chat(tok, history + [assistant], tools=schemas)
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        full_ids = tok.encode(full, add_special_tokens=False)
        assert full_ids[:len(prefix_ids)] == prefix_ids
        assert len(full_ids) > len(prefix_ids)


def test_generation_prefix_rejects_non_chatml_history():
    class BrokenTemplate(ThinkingTemplate):
        def apply_chat_template(self, *args, **kwargs):
            return "not ChatML"

    with pytest.raises(ValueError, match="completed message"):
        render_qwen_sft_aligned_prefix(BrokenTemplate(), [{"role": "user", "content": "U"}])
