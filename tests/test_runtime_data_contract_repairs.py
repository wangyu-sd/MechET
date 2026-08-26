import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from collections import UserDict

import pytest
import yaml

from mechet.assistant_masking import encode_assistant_only_conversation
from mechet.agent_env import AgentEnvConfig
from mechet.model_revision import (
    resolve_lineage_revision,
    resolve_loaded_model_revision,
)
from mechet.tool_schemas import trace_tool_schemas


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


def test_inference_revision_prefers_frozen_adapter_over_mutable_config():
    module = load_script("infer_mechet.py")
    assert (
        module._resolve_revision(
            {"training": {"model_revision": "main"}},
            "",
            {"base_model_revision": SHA_A},
            scripted=False,
        )
        == SHA_A
    )
    with pytest.raises(ValueError, match="immutable 40-hex model revision"):
        module._resolve_revision(
            {"training": {"model_revision": "main"}},
            "",
            {},
            scripted=False,
        )
    assert module._resolve_revision({}, "", {}, scripted=True) == "scripted"


def _reference_trace_row(max_tool_calls: int = 12) -> dict:
    observation = {
        "task": "trace_owned_inverse_electron_flow",
        "max_tool_calls": max_tool_calls,
        "faithfulness_contract": {
            "free_form_proof_submission": False,
            "endpoint_source": "environment_owned_trace",
            "final_tool": "finish_trace",
            "declared_moves_replayed_before_compilation": True,
            "observation_mode": "action_delta",
        },
    }
    return {
        "id": "matched-contract",
        "target_smiles": "[CH3:1][OH:2]",
        "messages": [
            {"role": "system", "content": "FROZEN TRAIN SYSTEM"},
            {
                "role": "user",
                "content": (
                    "TARGET: [CH3:1][OH:2]\nFrozen training instruction.\n\n"
                    "INITIAL ENVIRONMENT OBSERVATION:\n"
                    + json.dumps(observation)
                ),
            },
        ],
        "tools": trace_tool_schemas(),
    }


def test_reference_prompt_contract_reuses_training_preamble_and_budget():
    module = load_script("infer_mechet.py")
    row = _reference_trace_row()
    tools = module._reference_tools([row], trace_tool_schemas())
    env_config = AgentEnvConfig(max_tool_calls=12, observation_mode="action_delta")
    contract = module._reference_prompt_contract(
        [row], tools=tools, env_config=env_config, max_iterations=12
    )
    assert contract["source"] == "reference_training_messages_v1"
    assert contract["max_tool_calls"] == contract["max_iterations"] == 12
    runtime_observation = json.dumps(
        {
            **module._reference_initial_observation(row),
            "instructions": ["runtime-only display text is not model-visible"],
        }
    )
    assert module._trace_messages(
        row,
        "trace",
        runtime_observation,
        prompt_source="reference",
    ) == row["messages"][:2]


def test_reference_prompt_contract_rejects_budget_mismatch():
    module = load_script("infer_mechet.py")
    row = _reference_trace_row()
    with pytest.raises(ValueError, match="reference max_tool_calls=12 != runtime 40"):
        module._reference_prompt_contract(
            [row],
            tools=trace_tool_schemas(),
            env_config=AgentEnvConfig(
                max_tool_calls=40,
                observation_mode="action_delta",
            ),
            max_iterations=12,
        )


def test_inference_accepts_mapping_model_inputs(monkeypatch):
    module = load_script("infer_mechet.py")
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, **_kwargs):
            return UserDict(
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }
            )

        def decode(self, tokens, skip_special_tokens=False):
            assert skip_special_tokens is False
            return "generated:" + ",".join(str(int(x)) for x in tokens)

    class FakeModel:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2, 3]])

    text, prefix = module._generate_response(
        FakeModel(),
        FakeTokenizer(),
        [{"role": "user", "content": "x"}],
        [],
        max_new_tokens=4,
        temperature=0,
        top_p=1.0,
        seed=17,
    )
    assert text == "generated:3"
    assert prefix.tolist() == [1, 2]


def test_direct_inference_batches_sampled_candidates_in_one_generate_call():
    module = load_script("infer_mechet.py")
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, tokens, skip_special_tokens=False):
            assert skip_special_tokens is False
            return "generated:" + ",".join(str(int(x)) for x in tokens)

    class FakeModel:
        calls = 0

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            self.calls += 1
            assert kwargs["num_return_sequences"] == 3
            return torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]])

    model = FakeModel()
    texts, prefix = module._generate_responses(
        model,
        FakeTokenizer(),
        [{"role": "user", "content": "x"}],
        [],
        max_new_tokens=4,
        temperature=0.7,
        top_p=0.95,
        seeds=[17, 18, 19],
    )
    assert model.calls == 1
    assert texts == ["generated:3", "generated:4", "generated:5"]
    assert prefix.tolist() == [1, 2]


def test_direct_inference_trims_only_padding_after_first_eos():
    module = load_script("infer_mechet.py")
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, tokens, skip_special_tokens=False):
            assert skip_special_tokens is False
            return ",".join(str(int(x)) for x in tokens)

    class FakeModel:
        generation_config = type("GenerationConfig", (), {"eos_token_id": [0, 9]})()

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, **_kwargs):
            return torch.tensor(
                [
                    [1, 2, 3, 0, 0, 0],
                    [1, 2, 4, 5, 9, 0],
                    [1, 2, 6, 7, 8, 8],
                ]
            )

    texts, _ = module._generate_responses(
        FakeModel(),
        FakeTokenizer(),
        [{"role": "user", "content": "x"}],
        [],
        max_new_tokens=4,
        temperature=0.7,
        top_p=0.95,
        seeds=[17, 18, 19],
    )
    assert texts == ["3,0", "4,5,9", "6,7,8,8"]


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


def test_node_local_cache_rebinds_manifest_arrow_paths(tmp_path):
    module = load_script("train_tool_sft.py")
    local_cache = tmp_path / "local-cache"
    local_cache.mkdir()
    (local_cache / "train.rank00.arrow").write_bytes(b"arrow")

    resolved = module.resolve_cached_arrow_files(
        local_cache,
        ["data/original-cache/train.rank00.arrow"],
    )

    assert resolved == [str(local_cache / "train.rank00.arrow")]


def test_node_local_cache_rejects_missing_arrow_shard(tmp_path):
    module = load_script("train_tool_sft.py")

    with pytest.raises(FileNotFoundError, match="cached Arrow shard"):
        module.resolve_cached_arrow_files(
            tmp_path,
            ["data/original-cache/train.rank07.arrow"],
        )


def test_training_arguments_enable_validation_across_transformers_api():
    module = load_script("train_tool_sft.py")

    class FakeTrainingArguments:
        __dataclass_fields__ = {
            "evaluation_strategy": object(),
            "eval_accumulation_steps": object(),
            "gradient_checkpointing_kwargs": object(),
            "group_by_length": object(),
        }

    kwargs, _ = module._training_argument_kwargs(
        FakeTrainingArguments,
        cfg={"output_dir": "out"},
        training={"eval_strategy": "epoch", "eval_accumulation_steps": 4},
        max_steps=-1,
        seed=17,
        data_seed=17,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        has_validation=True,
    )
    assert kwargs["evaluation_strategy"] == "epoch"
    assert kwargs["per_device_eval_batch_size"] == 1
    assert kwargs["eval_accumulation_steps"] == 4
    assert kwargs["gradient_checkpointing_kwargs"] == {
        "use_reentrant": False
    }


def test_training_arguments_enable_fused_liger_loss_when_requested():
    module = load_script("train_tool_sft.py")

    class FakeTrainingArguments:
        __dataclass_fields__ = {
            "use_liger_kernel": object(),
            "liger_kernel_config": object(),
        }

    kernel_config = {
        "rope": False,
        "rms_norm": False,
        "swiglu": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
    }
    kwargs, _ = module._training_argument_kwargs(
        FakeTrainingArguments,
        cfg={"output_dir": "out"},
        training={
            "use_liger_kernel": True,
            "liger_kernel_config": kernel_config,
        },
        max_steps=1,
        seed=17,
        data_seed=17,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
    )
    assert kwargs["use_liger_kernel"] is True
    assert kwargs["liger_kernel_config"] == kernel_config


def test_training_arguments_expose_kernel_benchmark_metrics():
    module = load_script("train_tool_sft.py")

    class FakeTrainingArguments:
        __dataclass_fields__ = {
            "torch_compile": object(),
            "torch_compile_backend": object(),
            "torch_compile_mode": object(),
            "include_tokens_per_second": object(),
            "include_num_input_tokens_seen": object(),
        }

    kwargs, _ = module._training_argument_kwargs(
        FakeTrainingArguments,
        cfg={"output_dir": "out"},
        training={
            "torch_compile": True,
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "reduce-overhead",
        },
        max_steps=1,
        seed=17,
        data_seed=17,
        bf16=False,
        fp16=True,
        tf32=False,
        gradient_checkpointing=True,
    )
    assert kwargs["torch_compile"] is True
    assert kwargs["torch_compile_backend"] == "inductor"
    assert kwargs["torch_compile_mode"] == "reduce-overhead"
    assert kwargs["include_tokens_per_second"] is True
    assert kwargs["include_num_input_tokens_seen"] is True


def test_resume_checkpoint_is_explicit_and_selects_latest(tmp_path):
    module = load_script("train_tool_sft.py")
    for step in (10, 20):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    incomplete = tmp_path / "checkpoint-30"
    incomplete.mkdir()

    assert module.resolve_resume_checkpoint(None, tmp_path) is None
    assert module.resolve_resume_checkpoint("auto", tmp_path) == (
        tmp_path / "checkpoint-20"
    )
    assert module.resolve_resume_checkpoint(
        str(tmp_path / "checkpoint-10"), tmp_path
    ) == (tmp_path / "checkpoint-10")


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


def test_repaired_runner_clis_parse():
    for name in (
        "run_h1_suite.py",
        "run_h2_suite.py",
        "run_h3_suite.py",
        "run_h3_intervention.py",
    ):
        subprocess.check_call([sys.executable, str(REPO / "scripts" / name), "--help"])
