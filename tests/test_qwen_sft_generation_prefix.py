from pathlib import Path

import pytest

from mechet.assistant_masking import (
    render_chat,
    render_qwen_sft_text_prefix,
    render_qwen_sft_tool_prefix,
)


class ThinkingTemplate:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=True, tools=None
    ):
        assert not tokenize
        history = ""
        for index, message in enumerate(messages):
            content = str(message.get("content") or "")
            if message.get("tool_calls"):
                content += "<tool_call>flow</tool_call>"
            followed_by_tool = (
                message["role"] == "assistant"
                and index + 1 < len(messages)
                and messages[index + 1]["role"] == "tool"
            )
            if message["role"] == "assistant" and not followed_by_tool:
                content = "<think>\n\n</think>\n\n" + content
            history += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            return history + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return history


def test_tool_prefix_matches_completed_sft_tool_call_not_generation_template():
    tok = ThinkingTemplate()
    history = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assistant = {"role": "assistant", "content": "", "tool_calls": [{"name": "flow"}]}
    result = {"role": "tool", "content": "OK"}
    tools = [{"name": "flow"}]
    full = render_chat(tok, history + [assistant, result], tools=tools)
    prefix = render_qwen_sft_tool_prefix(tok, history, tools=tools)
    assert full.startswith(prefix)
    assert prefix.endswith("<|im_start|>assistant\n")
    assert not full.startswith(render_chat(tok, history, tools=tools, add_generation_prompt=True))


def test_text_prefix_matches_completed_sft_answer():
    tok = ThinkingTemplate()
    history = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    full = render_chat(tok, history + [{"role": "assistant", "content": "P"}])
    prefix = render_qwen_sft_text_prefix(tok, history)
    assert full.startswith(prefix)
    assert prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_real_qwen3_tool_and_text_prefixes_match_completed_sft_when_available():
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
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "foo", "arguments": {"a": 1}}}
    ]}
    cases = [
        (render_qwen_sft_tool_prefix(tok, history, tools=tools),
         render_chat(tok, history + [assistant, {"role": "tool", "name": "foo", "content": "OK"}], tools=tools)),
        (render_qwen_sft_text_prefix(tok, history),
         render_chat(tok, history + [{"role": "assistant", "content": "P"}])),
    ]
    for prefix, full in cases:
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        full_ids = tok.encode(full, add_special_tokens=False)
        assert full_ids[:len(prefix_ids)] == prefix_ids
        assert len(full_ids) > len(prefix_ids)


def test_prefixes_reject_non_chatml_history_and_missing_tool_schema():
    class BrokenTemplate(ThinkingTemplate):
        def apply_chat_template(self, *args, **kwargs):
            return "not ChatML"

    history = [{"role": "user", "content": "U"}]
    with pytest.raises(ValueError, match="completed message"):
        render_qwen_sft_tool_prefix(BrokenTemplate(), history, tools=[{"name": "foo"}])
    with pytest.raises(ValueError, match="completed message"):
        render_qwen_sft_text_prefix(BrokenTemplate(), history)
    with pytest.raises(ValueError, match="requires tool schemas"):
        render_qwen_sft_tool_prefix(ThinkingTemplate(), history, tools=[])
