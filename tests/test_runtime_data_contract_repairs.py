import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from mechet.assistant_masking import encode_assistant_only_conversation
from mechet.model_revision import (
    resolve_lineage_revision,
    resolve_loaded_model_revision,
)


REPO = Path(__file__).resolve().parents[1]
SHA_A = "a" * 40
SHA_B = "b" * 40


def load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeQwenTokenizer:
    name_or_path = "Qwen/Fake"
    eos_token = "<eos>"
    pad_token = None
    init_kwargs = {"_commit_hash": SHA_A}

    def apply_chat_template(
        self,
        messages=None,
        *,
        conversation=None,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        assert tokenize is False
        values = messages if messages is not None else conversation
        assert values is not None
        output = []
        if tools:
            output.append("<tools>" + json.dumps(tools, sort_keys=True) + "</tools>\n")
        for message in values:
            role = str(message.get("role") or "")
            body = str(message.get("content") or "")
            if message.get("tool_calls"):
                body += "TOOL_CALL:" + json.dumps(
                    message["tool_calls"], sort_keys=True
                )
            output.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            output.append("<|im_start|>assistant\n")
        return "".join(output)

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in str(text)]}


@pytest.fixture
def tool_row():
    return {
        "id": "r1",
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "USER_SECRET"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "inspect_state",
                            "arguments": {},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "inspect_state",
                "tool_call_id": "c1",
                "content": '{"ok": true}',
            },
            {"role": "assistant", "content": "DONE"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect_state",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def test_final_chatml_scan_masks_all_assistant_turns_without_prefix_rerender(tool_row):
    tokenizer = FakeQwenTokenizer()
    encoded, audit = encode_assistant_only_conversation(
        tokenizer, tool_row, max_length=12288
    )
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    supervised_text = "".join(
        chr(token) for token, label in zip(input_ids, labels) if label != -100
    )
    unsupervised_text = "".join(
        chr(token) for token, label in zip(input_ids, labels) if label == -100
    )
    assert audit["mask_method"] == "final_chatml_token_scan_v1"
    assert audit["assistant_turns"] == 2
    assert len(audit["assistant_spans"]) == 2
    assert "TOOL_CALL:" in supervised_text
    assert "DONE" in supervised_text
    assert "USER_SECRET" in unsupervised_text
    assert "USER_SECRET" not in supervised_text


def test_zero_truncation_audit_reports_over_budget_without_slicing(tool_row):
    tokenizer = FakeQwenTokenizer()
    encoded, audit = encode_assistant_only_conversation(
        tokenizer, tool_row, max_length=8
    )
    assert audit["exceeds_max_length"] is True
    assert len(encoded["input_ids"]) == audit["raw_length"]
    assert len(encoded["input_ids"]) > 8


def test_mutable_model_request_resolves_to_loaded_immutable_commit():
    tokenizer = FakeQwenTokenizer()
    value = resolve_loaded_model_revision(
        model_name_or_path="Qwen/Qwen3-0.6B",
        requested_revision="main",
        tokenizer=tokenizer,
    )
    assert value["requested_model_revision"] == "main"
    assert value["resolved_model_revision"] == SHA_A
    assert value["tokenizer_revision"] == SHA_A


def test_mutable_downstream_config_defers_to_adapter_commit():
    assert resolve_lineage_revision("main", SHA_A) == SHA_A
    with pytest.raises(ValueError, match="revision mismatch"):
        resolve_lineage_revision(SHA_B, SHA_A)


def test_transformers5_length_grouping_uses_sampling_strategy():
    module = load_script("train_tool_sft.py")

    class FakeTrainingArguments:
        __dataclass_fields__ = {
            "train_sampling_strategy": object(),
        }

    kwargs, report = module._training_argument_kwargs(
        FakeTrainingArguments,
        cfg={"output_dir": "out"},
        training={"group_by_length": True},
        max_steps=1,
        seed=1,
        data_seed=1,
        bf16=False,
        fp16=False,
        tf32=False,
        gradient_checkpointing=False,
    )
    assert kwargs["train_sampling_strategy"] == "group_by_length"
    assert report["api_field"] == "train_sampling_strategy"
    assert report["applied_value"] == "group_by_length"


def _h1_row(identifier="h1", calls=1):
    tool_calls = []
    for index in range(calls - 1):
        tool_calls.append(
            {
                "id": f"inspect-{index}",
                "type": "function",
                "function": {"name": "inspect_state", "arguments": {}},
            }
        )
    tool_calls.append(
        {
            "id": "finish",
            "type": "function",
            "function": {"name": "finish_trace", "arguments": {}},
        }
    )
    messages = [{"role": "assistant", "tool_calls": tool_calls}]
    for call in tool_calls:
        name = call["function"]["name"]
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": call["id"],
                "content": json.dumps(
                    {"ok": True, "endpoint_exact": True}
                    if name == "finish_trace"
                    else {"ok": True}
                ),
            }
        )
    return {
        "id": identifier,
        "target_smiles": "[CH3:1][OH:2]",
        "expected_precursor": "[CH3:1][OH:2]",
        "messages": messages,
        "metadata": {
            "endpoint_source": "environment_owned_trace",
            "executor_replayed": True,
            "trace_digest": "trace",
            "trace_plan": {"steps": []},
        },
    }


def test_h1_benchmark_validation_requires_trace_owned_finish_and_budget():
    module = load_script("build_h1_benchmark.py")
    result = module.validate_h1_row(_h1_row(), max_tool_calls=16)
    assert result["id"] == "h1"
    assert result["tool_calls"] == 1
    with pytest.raises(ValueError, match="H1_TOOL_BUDGET_EXCEEDED"):
        module.validate_h1_row(_h1_row(calls=17), max_tool_calls=16)


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_split_evidence_builder_requires_disjoint_train_valid_test_ids(tmp_path):
    rows = {
        "train": [{"id": "train-1", "target_smiles": "C", "expected_precursor": "C"}],
        "valid": [{"id": "valid-1", "target_smiles": "N", "expected_precursor": "N"}],
        "test": [{"id": "test-1", "target_smiles": "O", "expected_precursor": "O"}],
    }
    inputs = {}
    for split, values in rows.items():
        path = tmp_path / f"{split}.jsonl"
        _write_jsonl(path, values)
        inputs[split] = path
    output = tmp_path / "suite"
    config = {
        "suite_id": "split-test",
        "output_dir": str(output),
        "conditions": {
            "source": {"input": str(inputs["train"]), "knowledge": "none"},
            "derived": {"derive_from": "source", "transform": "strip_knowledge"},
        },
        "splits": {
            name: {"inputs": {"source": str(path)}}
            for name, path in inputs.items()
        },
    }
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.check_call(
        [
            sys.executable,
            str(REPO / "scripts/build_knowledge_ablation_suite.py"),
            "--config",
            str(config_path),
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["controls"]["train_valid_test_stable_ids_disjoint"] is True
    assert all(value == 0 for value in manifest["cross_split_id_overlap"].values())
    assert (output / "test" / "derived.jsonl").exists()


def _evidence_row(identifier, term, passage_id):
    text = f"Evidence text for {identifier} and a matched mechanistic topic."
    result = {
        "context": {
            "text": text,
            "n_characters": len(text),
            "passage_ids": [passage_id],
        },
        "matches": (
            [{"matched_terms": [term], "state_terms": []}] if term else []
        ),
        "direct_reward": False,
    }
    return {
        "id": identifier,
        "target_smiles": "CC",
        "expected_precursor": "C.C",
        "messages": [
            {
                "role": "tool",
                "name": "retrieve_textbook_guidance",
                "tool_call_id": f"tool-{identifier}",
                "content": json.dumps(result),
            }
        ],
        "metadata": {"textbook_context_characters": len(text)},
    }


def test_same_topic_wrong_writes_paired_eligible_reference(tmp_path):
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(
        input_path,
        [
            _evidence_row("a", "substitution", "p-a"),
            _evidence_row("b", "substitution", "p-b"),
            _evidence_row("c", "", "p-c"),
        ],
    )
    output = tmp_path / "interventions"
    subprocess.check_call(
        [
            sys.executable,
            str(REPO / "scripts/build_evidence_interventions.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output),
            "--intervention",
            "same_topic_wrong",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    spec = manifest["outputs"]["same_topic_wrong"]
    eligible = json.loads(Path(spec["eligible_ids"]).read_text())
    transformed = [json.loads(line) for line in Path(spec["path"]).read_text().splitlines()]
    reference = [
        json.loads(line)
        for line in Path(spec["paired_reference"]).read_text().splitlines()
    ]
    assert spec["n_rows"] == 2
    assert spec["n_quarantined"] == 1
    assert spec["same_ids_as_full_input"] is False
    assert eligible["stable_ids"] == [row["id"] for row in transformed]
    assert [row["id"] for row in reference] == [row["id"] for row in transformed]
    assert manifest["contract"]["full_input_id_universe_preserved"] is False
