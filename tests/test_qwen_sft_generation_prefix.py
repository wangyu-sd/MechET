import pytest

from mechet.assistant_masking import render_chat, render_qwen_sft_aligned_prefix


class ThinkingTemplate:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=True, tools=None
    ):
        assert not tokenize
        history = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
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
    old = render_chat(tok, history, tools=[{"name": "flow"}], add_generation_prompt=True)
    assert full.startswith(aligned)
    assert not full.startswith(old)
    assert aligned.endswith("<|im_start|>assistant\n")
    assert "<think>" not in aligned


def test_generation_prefix_rejects_non_chatml_history():
    class BrokenTemplate(ThinkingTemplate):
        def apply_chat_template(self, *args, **kwargs):
            return "not ChatML"

    with pytest.raises(ValueError, match="completed message"):
        render_qwen_sft_aligned_prefix(BrokenTemplate(), [{"role": "user", "content": "U"}])
