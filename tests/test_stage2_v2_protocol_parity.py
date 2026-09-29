import argparse
import json
from pathlib import Path

import pytest

from mechet.assistant_masking import render_chat, render_qwen_sft_tool_prefix
from scripts.run_natural_language_value_search import (
    validate_matched_v2_args,
    validate_v2_adapter_manifest,
)
from scripts.audit_stage2_v2_protocol_parity import (
    assert_tool_prefix_matches_sft,
    reconstruct_history_user_prompt,
)
from scripts.eval_natural_language_event_local import render_policy_prompt


class ThinkingTemplate:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt,
        enable_thinking=True, tools=None,
    ):
        assert tokenize is False
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


def matched_args(**overrides):
    values = dict(
        matched_v2=True,
        legacy_dual_prompt=False,
        max_decisions=40,
        max_imports=32,
        branching=1,
        early_beam=1,
        late_beam=1,
        value_adapter="",
        value_weight=0.0,
        compact_history=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_qwen_tool_prefix_matches_completed_sft_tool_call_without_thinking():
    tok = ThinkingTemplate()
    history = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    tools = [{"type": "function", "function": {"name": "flow", "parameters": {"type": "object"}}}]
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "x",
                "type": "function",
                "function": {"name": "flow", "arguments": {"a": 1}},
            }
        ],
    }
    result = {"role": "tool", "name": "flow", "content": "OK"}
    full = render_chat(tok, history + [assistant, result], tools=tools)
    prefix = render_qwen_sft_tool_prefix(tok, history, tools=tools)
    assert full.startswith(prefix)
    assert prefix.endswith("<|im_start|>assistant\n")
    assert "<think>" not in prefix
    assert not full.startswith(
        render_chat(tok, history, tools=tools, add_generation_prompt=True)
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"legacy_dual_prompt": True}, "legacy dual prompt"),
        ({"max_decisions": 12}, "40 decisions"),
        ({"max_decisions": 41}, "40 decisions"),
        ({"max_imports": 8}, "32 imports"),
        ({"max_imports": 33}, "32 imports"),
        ({"branching": 2}, "branching=1"),
        ({"early_beam": 2}, "beam width 1"),
        ({"late_beam": 2}, "beam width 1"),
        ({"value_adapter": "/tmp/value"}, "value critic"),
        ({"value_weight": 0.2}, "value critic"),
    ],
)
def test_matched_v2_rejects_historical_inference_contracts(override, match):
    with pytest.raises(ValueError, match=match):
        validate_matched_v2_args(matched_args(**override))


def test_matched_v2_accepts_pure_policy_40_32_contract():
    validate_matched_v2_args(matched_args())


@pytest.mark.parametrize(
    "compact_history,environment",
    [
        (False, "natural_language_electron_event_v2"),
        (True, "natural_language_electron_event_history_v2"),
    ],
)
def test_v2_adapter_manifest_accepts_only_matching_stage_contract(
    tmp_path: Path, compact_history: bool, environment: str
):
    adapter = tmp_path / ("history" if compact_history else "state")
    adapter.mkdir()
    manifest = {
        "artifact_type": "trainable_peft_adapter",
        "base_model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "environment_revision": environment,
        "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
    }
    (adapter / "adapter_manifest.json").write_text(json.dumps(manifest))
    observed = validate_v2_adapter_manifest(adapter, compact_history=compact_history)
    assert observed["environment_revision"] == environment


def test_v2_adapter_manifest_rejects_v1_checkpoint(tmp_path: Path):
    adapter = tmp_path / "legacy"
    adapter.mkdir()
    (adapter / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "trainable_peft_adapter",
                "base_model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "environment_revision": "natural_language_electron_event_history_v1",
                "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
            }
        )
    )
    with pytest.raises(ValueError, match="protocol-v2"):
        validate_v2_adapter_manifest(adapter, compact_history=True)


def test_runtime_history_prompt_is_exact_history_sft_transform():
    base = (
        "TARGET PRODUCT SMILES: CC=O\n"
        "CURRENT STATE SMILES: CC=O\n\n"
        "MOLECULAR INVENTORY\nA01 ...\n"
        "\nChoose the single next retrosynthetic action."
    )
    actions = [
        {
            "name": "import_fragments",
            "arguments": {"fragments": [{"smiles": "O", "count": 1}]},
            "result": {"ok": True, "code": "PASS", "current_state": "CC=O.O"},
        }
    ]
    prompt = reconstruct_history_user_prompt(base, actions)
    assert "TRAJECTORY HISTORY" in prompt
    assert "accepted_action_types: import_fragments" in prompt
    assert prompt.endswith("Choose the single next retrosynthetic action.")


def test_prefix_audit_detects_generation_template_drift():
    tok = ThinkingTemplate()
    row = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": "flow", "arguments": {"a": 1}},
                    }
                ],
            },
            {"role": "tool", "name": "flow", "content": "OK"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "flow", "parameters": {"type": "object"}},
            }
        ],
    }
    report = assert_tool_prefix_matches_sft(tok, row)
    assert report["prefix_match"] is True
    assert report["generation_template_match"] is False


def test_local_oracle_evaluator_can_use_sft_aligned_tool_prefix():
    tok = ThinkingTemplate()
    task = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "flow", "parameters": {"type": "object"}},
            }
        ],
    }
    aligned = render_policy_prompt(tok, task, sft_aligned=True)
    legacy = render_policy_prompt(tok, task, sft_aligned=False)
    assert aligned.endswith("<|im_start|>assistant\n")
    assert "<think>" not in aligned
    assert "<think>" in legacy
