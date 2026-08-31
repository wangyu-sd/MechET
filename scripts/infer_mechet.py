#!/usr/bin/env python3
"""Run reproducible MechET rollouts, matched conditions, and H1 controls."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from agent_model_init import path_sha256
from mechet.agent_env import AgentEnvConfig
from mechet.agent_inference import (
    append_tool_exchange,
    parse_tool_calls,
    scripted_rollout,
    tool_result_pool,
)
from mechet.frozen_evidence_environments import (
    FrozenAnchorTraceOwnedTRLEnvironment,
    FrozenTextbookAnchorTraceOwnedTRLEnvironment,
    FrozenTextbookTraceOwnedTRLEnvironment,
)
from mechet.knowledge_ablation import read_jsonl, row_id
from mechet.knowledge_agent_env import KnowledgeAgentConfig
from mechet.model_revision import resolve_lineage_revision
from mechet.tool_schemas import trace_tool_schemas
from mechet.trl_environments import (
    LegacyProofTRLEnvironment,
    TraceOwnedTRLEnvironment,
)

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("install PyYAML or mechet[agent]") from exc


TRACE_SYSTEM = """You are MechET, a trace-owned inverse electron-flow agent.
Use explicit mapped electron-flow tools for every state change. The only final
proof and precursor must be produced by finish_trace. Abstain when unsupported."""
EVIDENCE_SUFFIX = """
Retrieved passages and mechanistic anchors are soft external evidence. They are
not answers, rewards, or validity oracles."""
DIRECT_SYSTEM = """Predict the atom-contributing structural precursor SMILES.
Return one line beginning with PRECURSOR:."""
CANDIDATE_SELECTOR = "sample0_direct__formal_trace_reward_failures_v1"
REFERENCE_OBSERVATION_MARKER = "INITIAL ENVIRONMENT OBSERVATION:\n"


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def _adapter_manifest(adapter: str) -> dict[str, Any]:
    return _load_json_object(Path(adapter) / "adapter_manifest.json") if adapter else {}


def _resolve_revision(
    cfg: Mapping[str, Any], cli_revision: str, manifest: Mapping[str, Any], *, scripted: bool
) -> str:
    if scripted:
        return "scripted"
    training = dict(cfg.get("training") or {})
    configured = str(
        cli_revision
        or training.get("model_revision")
        or cfg.get("model_revision")
        or ""
    ).strip()
    adapter_revision = str(
        manifest.get("base_model_revision")
        or manifest.get("model_revision")
        or ""
    ).strip()
    revision = resolve_lineage_revision(configured, adapter_revision)
    if revision:
        return revision
    raise ValueError(
        "non-scripted inference requires an immutable 40-hex model revision via "
        "--model-revision or adapter_manifest.json; mutable config aliases such "
        "as main are not frozen inference revisions"
    )


def _software_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "rdkit",
        "vllm",
    ):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "unavailable"
    return output


def _candidate_seed(base_seed: int, identifier: str, sample_index: int) -> int:
    payload = f"{base_seed}\0{identifier}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _condition_metadata(row: Mapping[str, Any]) -> tuple[Any, Any]:
    metadata = dict(row.get("metadata") or {})
    return (
        row.get("competitor_products") or metadata.get("competitor_products"),
        row.get("conditions") or metadata.get("conditions"),
    )


def _tool_result(row: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    for message in row.get("messages") or []:
        if message.get("role") == "tool" and message.get("name") == name:
            try:
                value = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                return None
            return dict(value) if isinstance(value, dict) else None
    return None


def _required_tool_calls(row: Mapping[str, Any]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in row.get("messages") or []
        if message.get("role") == "assistant"
    )


def _environment_config(
    cfg: dict[str, Any], mode: str
) -> AgentEnvConfig | KnowledgeAgentConfig:
    payload = dict(cfg.get("environment") or {})
    if mode in {"textbook", "irrelevant", "anchors", "combined"}:
        payload.setdefault(
            "textbook_corpus_path",
            str(cfg.get("textbook_corpus_path") or REPO / "knowledge/corpus/passages.jsonl"),
        )
        payload.setdefault(
            "primitive_library_path",
            str(
                cfg.get("primitive_library_path")
                or REPO / "knowledge/primitives/core_polar_primitives.yaml"
            ),
        )
        payload.setdefault(
            "primitive_source_registry_path",
            str(
                cfg.get("primitive_source_registry_path")
                or REPO / "knowledge/source_registry.yaml"
            ),
        )
        payload["enable_structured_primitives"] = mode in {"anchors", "combined"}
        payload["require_textbook_corpus"] = mode in {
            "textbook",
            "irrelevant",
            "combined",
        }
        return KnowledgeAgentConfig(**payload)
    return AgentEnvConfig(**payload)


def _environment(
    cfg: dict[str, Any],
    mode: str,
    *,
    intervention: str,
    shuffled: dict[str, Any],
):
    env_cfg = _environment_config(cfg, mode)
    common = {
        "config": env_cfg,
        "forward_checkpoint": cfg.get("forward_checkpoint") or None,
        "forward_device": str(cfg.get("forward_device") or "cpu"),
    }
    intervention_kwargs = {
        "intervention": intervention,
        "shuffled_observations": shuffled,
    }
    if mode == "trace":
        return TraceOwnedTRLEnvironment(**common, **intervention_kwargs)
    if mode in {"textbook", "irrelevant"}:
        return FrozenTextbookTraceOwnedTRLEnvironment(**common, **intervention_kwargs)
    if mode == "anchors":
        return FrozenAnchorTraceOwnedTRLEnvironment(**common, **intervention_kwargs)
    if mode == "combined":
        return FrozenTextbookAnchorTraceOwnedTRLEnvironment(
            **common, **intervention_kwargs
        )
    if mode == "legacy":
        return LegacyProofTRLEnvironment(**common)
    raise ValueError(f"mode has no environment: {mode}")


def _tools(mode: str) -> list[dict[str, Any]]:
    if mode == "trace":
        return trace_tool_schemas()
    if mode in {"textbook", "irrelevant"}:
        return trace_tool_schemas(textbook=True)
    if mode == "anchors":
        return trace_tool_schemas(anchors=True)
    if mode == "combined":
        return trace_tool_schemas(textbook=True, anchors=True)
    if mode == "legacy":
        allowed = {
            "inspect_state",
            "apply_electron_move",
            "apply_coupled_electron_moves",
            "submit_proof",
            "abstain",
        }
        return [
            item
            for item in trace_tool_schemas(legacy_submit_proof=True)
            if str((item.get("function") or {}).get("name") or "") in allowed
        ]
    return []


def _direct_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = [
        dict(item)
        for item in row.get("messages") or []
        if item.get("role") in {"system", "user"}
    ]
    system = next((item for item in messages if item.get("role") == "system"), None)
    user = next((item for item in messages if item.get("role") == "user"), None)
    if system and user:
        return [system, user]
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": f"TARGET: {str(row.get('target_smiles') or '')}"},
    ]


def _reference_trace_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact system/user preamble used to train this row."""
    system = next(
        (
            dict(message)
            for message in row.get("messages") or []
            if message.get("role") == "system"
        ),
        None,
    )
    user = next(
        (
            dict(message)
            for message in row.get("messages") or []
            if message.get("role") == "user"
        ),
        None,
    )
    if system is None or user is None:
        raise ValueError("reference row has no system/user training preamble")
    target = str(row.get("target_smiles") or "")
    if not str(user.get("content") or "").startswith(f"TARGET: {target}\n"):
        raise ValueError("reference user prompt does not match target_smiles")
    return [system, user]


def _reference_initial_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    user = _reference_trace_messages(row)[1]
    content = str(user.get("content") or "")
    if REFERENCE_OBSERVATION_MARKER not in content:
        raise ValueError("reference user prompt has no initial observation")
    payload = content.split(REFERENCE_OBSERVATION_MARKER, 1)[1].strip()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("reference initial observation is not a JSON object")
    return dict(value)


def _reference_tools(
    rows: list[Mapping[str, Any]], runtime_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    encoded = {
        json.dumps(row.get("tools") or [], sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    if len(encoded) != 1:
        raise ValueError("reference rows do not share one frozen tool schema")
    tools = json.loads(next(iter(encoded)))
    if not tools:
        raise ValueError("reference rows have no frozen tool schema")
    if tools != runtime_tools:
        raise ValueError("reference and runtime tool schemas differ")
    return tools


def _reference_prompt_contract(
    rows: list[Mapping[str, Any]],
    *,
    tools: list[dict[str, Any]],
    env_config: AgentEnvConfig | KnowledgeAgentConfig,
    max_iterations: int,
) -> dict[str, Any]:
    observations = [_reference_initial_observation(row) for row in rows]
    budgets = {int(item.get("max_tool_calls") or 0) for item in observations}
    modes = {
        str((item.get("faithfulness_contract") or {}).get("observation_mode") or "")
        for item in observations
    }
    system_prompts = {
        str(_reference_trace_messages(row)[0].get("content") or "") for row in rows
    }
    if 0 in budgets:
        raise ValueError(f"reference rows have invalid tool budgets: {sorted(budgets)}")
    max_budget = max(budgets)
    if max_budget > int(env_config.max_tool_calls):
        raise ValueError(
            f"reference max_tool_calls={max_budget} exceeds runtime cap "
            f"{env_config.max_tool_calls}"
        )
    if max_budget > max_iterations:
        raise ValueError(
            f"reference max_tool_calls={max_budget} exceeds max_iterations="
            f"{max_iterations}"
        )
    if modes != {str(env_config.observation_mode)}:
        raise ValueError(
            f"reference observation modes {sorted(modes)} != runtime "
            f"{env_config.observation_mode}"
        )
    if len(system_prompts) != 1:
        raise ValueError("reference rows do not share one frozen system prompt")
    schema_payload = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    system_prompt = next(iter(system_prompts))
    value = {
        "source": "reference_training_messages_v1",
        "max_tool_calls": (
            next(iter(budgets)) if len(budgets) == 1 else None
        ),
        "max_tool_calls_values": sorted(budgets),
        "max_tool_calls_min": min(budgets),
        "max_tool_calls_max": max_budget,
        "runtime_max_tool_calls_cap": int(env_config.max_tool_calls),
        "max_iterations": max_iterations,
        "observation_mode": str(env_config.observation_mode),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "tool_schema_sha256": hashlib.sha256(schema_payload.encode()).hexdigest(),
    }
    value["contract_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def _trace_messages(
    row: Mapping[str, Any],
    mode: str,
    observation: str,
    *,
    prompt_source: str = "runtime",
) -> list[dict[str, Any]]:
    if prompt_source == "reference":
        if mode != "trace":
            raise ValueError("reference prompt source is supported only for trace mode")
        reference_observation = _reference_initial_observation(row)
        runtime_observation = json.loads(observation)
        for key in ("task", "max_tool_calls", "faithfulness_contract"):
            if reference_observation.get(key) != runtime_observation.get(key):
                raise ValueError(f"reference/runtime initial observation mismatch: {key}")
        return _reference_trace_messages(row)
    suffix = (
        EVIDENCE_SUFFIX
        if mode in {"textbook", "irrelevant", "anchors", "combined"}
        else ""
    )
    return [
        {"role": "system", "content": TRACE_SYSTEM + suffix},
        {
            "role": "user",
            "content": (
                f"TARGET: {str(row.get('target_smiles') or '')}\n\n"
                "INITIAL ENVIRONMENT OBSERVATION:\n" + observation
            ),
        },
    ]


def _load_model(
    model_name: str,
    adapter: str,
    *,
    revision: str,
    device_map: str | None,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = None
    if torch.cuda.is_available():
        # Volta/V100 cannot execute BF16 kernels.  Ampere and newer use BF16;
        # older CUDA devices load the same frozen model in FP16.
        major, _ = torch.cuda.get_device_capability(0)
        torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map or ("auto" if torch.cuda.is_available() else None),
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


class _VllmBackend:
    """Small compatibility wrapper around vLLM's offline batched engine.

    Each logical candidate is submitted as its own request with its frozen
    candidate/turn seed.  vLLM may schedule those requests together, but a
    candidate therefore does not inherit an RNG stream from its batch
    neighbours.  The wrapper deliberately exposes the same `(text, prefix)`
    records consumed by the existing Qwen tool-call parser.
    """

    def __init__(self, engine: Any, tokenizer: Any, sampling_params: Any, lora_request: Any):
        self.engine = engine
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.lora_request = lora_request

    def generate(
        self,
        conversations: Sequence[list[dict[str, Any]]],
        tools: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: Sequence[int],
    ) -> list[tuple[str, Any]]:
        if not conversations or len(conversations) != len(seeds):
            raise ValueError("vLLM conversations and seeds must be non-empty and aligned")
        prompts: list[str] = []
        for messages in conversations:
            kwargs: dict[str, Any] = {
                "conversation": messages,
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = tools
            prompts.append(str(self.tokenizer.apply_chat_template(**kwargs)))

        sampling = [
            self.sampling_params(
                max_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                top_p=top_p,
                seed=int(seed),
                n=1,
            )
            for seed in seeds
        ]
        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self.lora_request is not None:
            generate_kwargs["lora_request"] = self.lora_request
        outputs = self.engine.generate(prompts, sampling, **generate_kwargs)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} requests for {len(prompts)} prompts"
            )

        records: list[tuple[str, Any]] = []
        for output in outputs:
            candidates = list(getattr(output, "outputs", []) or [])
            if len(candidates) != 1:
                raise RuntimeError(
                    f"vLLM request returned {len(candidates)} candidates; expected one"
                )
            prefix = list(getattr(output, "prompt_token_ids", []) or [])
            records.append((str(candidates[0].text), prefix))
        return records


def _adapter_lora_rank(adapter: str) -> int:
    if not adapter:
        return 16
    config = _load_json_object(Path(adapter) / "adapter_config.json")
    return max(1, int(config.get("r") or 16))


def _load_vllm_backend(
    model_name: str,
    adapter: str,
    *,
    revision: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    enable_prefix_caching: bool,
) -> tuple[_VllmBackend, Any]:
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "--backend vllm requires a compatible vllm installation"
        ) from exc

    dtype = "float16"
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            dtype = "bfloat16"

    engine_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": enable_prefix_caching,
    }
    if revision and not Path(model_name).exists():
        engine_kwargs["revision"] = revision
    if adapter:
        engine_kwargs.update(
            enable_lora=True,
            max_lora_rank=_adapter_lora_rank(adapter),
            max_loras=1,
        )
    engine = LLM(**engine_kwargs)
    tokenizer = engine.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_request = None
    if adapter:
        try:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("mechet_adapter", 1, str(Path(adapter).resolve()))
        except Exception as exc:
            raise RuntimeError(
                f"vLLM could not construct a LoRA request for {adapter}: {exc}"
            ) from exc
    backend = _VllmBackend(engine, tokenizer, SamplingParams, lora_request)
    return backend, tokenizer


def _generate_response(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, Any]:
    texts, prefix = _generate_responses(
        model,
        tokenizer,
        messages,
        tools,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seeds=[seed],
    )
    return texts[0], prefix


@contextmanager
def _independent_multinomial_streams(seeds: Sequence[int], device: Any) -> Iterator[None]:
    """Give every row in a sampled batch its own deterministic RNG stream.

    Transformers normally draws one batched multinomial from the global CUDA
    generator.  That makes a candidate's samples depend on its neighbours and
    prevents safe dynamic batching.  During one ``generate`` call, route each
    row through a generator seeded by that candidate/iteration instead.
    """

    import torch

    original = torch.multinomial
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]

    def independent(input: Any, num_samples: int, replacement: bool = False, *, out: Any = None, generator: Any = None):
        if (
            generator is None
            and out is None
            and getattr(input, "ndim", 0) == 2
            and int(input.shape[0]) == len(generators)
        ):
            return torch.cat(
                [
                    original(
                        input[index : index + 1],
                        num_samples,
                        replacement,
                        generator=generators[index],
                    )
                    for index in range(len(generators))
                ],
                dim=0,
            )
        return original(
            input,
            num_samples,
            replacement,
            out=out,
            generator=generator,
        )

    torch.multinomial = independent
    try:
        yield
    finally:
        torch.multinomial = original


def _generate_trace_responses(
    model: Any,
    tokenizer: Any,
    conversations: Sequence[list[dict[str, Any]]],
    tools: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seeds: Sequence[int],
) -> list[tuple[str, Any]]:
    """Generate one tool response for several independent active rollouts."""

    if isinstance(model, _VllmBackend):
        return model.generate(
            conversations,
            tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=seeds,
        )

    import torch

    if not conversations or len(conversations) != len(seeds):
        raise ValueError("trace conversations and seeds must be non-empty and aligned")
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        kwargs: dict[str, Any] = {
            "conversation": list(conversations),
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "padding": True,
        }
        if tools:
            kwargs["tools"] = tools
        encoded = tokenizer.apply_chat_template(**kwargs)
    finally:
        tokenizer.padding_side = previous_padding_side
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("batched chat template did not return model inputs")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    input_length = int(inputs["input_ids"].shape[-1])
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    with torch.inference_mode(), _independent_multinomial_streams(seeds, device):
        output = model.generate(**inputs, **generate_kwargs)
    generated = output[:, input_length:]

    eos_token_ids: set[int] = set()
    configured_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(configured_eos, int):
        eos_token_ids.add(configured_eos)
    elif configured_eos is not None:
        eos_token_ids.update(int(token_id) for token_id in configured_eos)
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(int(tokenizer.eos_token_id))

    results: list[tuple[str, Any]] = []
    for index, tokens in enumerate(generated):
        token_values = [int(token_id) for token_id in tokens.tolist()]
        stop = next(
            (
                position + 1
                for position, token_id in enumerate(token_values)
                if token_id in eos_token_ids
            ),
            len(token_values),
        )
        attention = inputs.get("attention_mask")
        prefix = (
            inputs["input_ids"][index][attention[index].bool()]
            if attention is not None
            else inputs["input_ids"][index]
        )
        results.append(
            (tokenizer.decode(tokens[:stop], skip_special_tokens=False), prefix)
        )
    return results


def _generate_trace_responses_resilient(
    model: Any,
    tokenizer: Any,
    requests: Sequence[tuple[list[dict[str, Any]], int]],
    tools: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[str, Any, str]]:
    """Run a trace microbatch and split it deterministically after CUDA OOM."""

    if not requests:
        return []
    try:
        generated = _generate_trace_responses(
            model,
            tokenizer,
            [item[0] for item in requests],
            tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=[item[1] for item in requests],
        )
        return [(raw, prefix, "") for raw, prefix in generated]
    except Exception as exc:
        if len(requests) > 1 and "out of memory" in str(exc).lower():
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            middle = len(requests) // 2
            return _generate_trace_responses_resilient(
                model,
                tokenizer,
                requests[:middle],
                tools,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ) + _generate_trace_responses_resilient(
                model,
                tokenizer,
                requests[middle:],
                tools,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        return [("", None, f"{type(exc).__name__}: {exc}") for _ in requests]


def _generate_responses(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seeds: list[int],
) -> tuple[list[str], Any]:
    """Sample several direct candidates in one model.generate call.

    Transformer generation expands the single encoded prompt to
    ``num_return_sequences`` internally, so the prompt/model load is shared
    across K candidates.  The batch is deterministically seeded by its first
    candidate; every candidate still records its stable logical seed and the
    common batch seed in the output artifact.
    """
    if isinstance(model, _VllmBackend):
        generated = model.generate(
            [messages for _ in seeds],
            tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=seeds,
        )
        return [item[0] for item in generated], generated[0][1]

    import torch

    if not seeds:
        raise ValueError("generation seeds must not be empty")
    if len(seeds) > 1 and temperature <= 0:
        raise ValueError("batched direct candidates require stochastic decoding")
    _set_seed(seeds[0])
    kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if tools:
        kwargs["tools"] = tools
    encoded = tokenizer.apply_chat_template(**kwargs)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("chat template did not return model inputs")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    input_length = int(inputs["input_ids"].shape[-1])
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "num_return_sequences": len(seeds),
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)
    generated = output[:, input_length:]

    # ``generate`` right-pads an already-finished sequence to the longest
    # sequence in the sampled batch.  Qwen uses ``<|im_end|>`` as both EOS and
    # (in this launcher) the pad token, so decoding the untrimmed tensor can
    # turn a short valid answer into thousands of repeated ``<|im_end|>``
    # strings whenever a sibling candidate hits the token budget.  Besides
    # bloating the artifact, that poisoned downstream NLL ranking.  Preserve
    # the first EOS token and discard only the batch-alignment padding after it.
    eos_token_ids: set[int] = set()
    configured_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(configured_eos, int):
        eos_token_ids.add(configured_eos)
    elif configured_eos is not None:
        eos_token_ids.update(int(token_id) for token_id in configured_eos)
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(int(tokenizer.eos_token_id))

    trimmed = []
    for tokens in generated:
        token_values = [int(token_id) for token_id in tokens.tolist()]
        stop = next(
            (
                index + 1
                for index, token_id in enumerate(token_values)
                if token_id in eos_token_ids
            ),
            len(token_values),
        )
        trimmed.append(tokens[:stop])
    return [
        tokenizer.decode(tokens, skip_special_tokens=False) for tokens in trimmed
    ], inputs["input_ids"][0]


def _rank_candidate(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    state = dict(candidate.get("rollout_state") or {})
    final = dict(state.get("final_result") or {})
    return (
        float(bool(final.get("formal_execute") or final.get("ok"))),
        float(bool(final.get("trace_bound"))),
        float(final.get("reward") or state.get("reward") or -1e9),
        -float(state.get("failed_steps") or 0),
        -float(state.get("tool_calls") or 0),
        -float(candidate.get("sample_index", 0)),
    )


class _TraceRollout:
    def __init__(
        self,
        *,
        env: Any,
        messages: list[dict[str, Any]],
        sample_index: int,
        seed: int,
        max_iterations: int,
        exchanges: list[dict[str, Any]],
    ) -> None:
        self.env = env
        self.messages = messages
        self.sample_index = sample_index
        self.seed = seed
        self.max_iterations = max_iterations
        self.exchanges = exchanges
        self.iteration = 0
        self.termination_reason = "max_iterations"
        self.generation_error = ""
        self.finished = False

    def request(self) -> tuple[list[dict[str, Any]], int]:
        return self.messages, self.seed + self.iteration

    def advance(self, raw: str, prefix: Any, error: str, tokenizer: Any) -> None:
        if self.finished:
            raise RuntimeError("cannot advance a finished trace rollout")
        if error:
            self.generation_error = error
            self.termination_reason = "generation_or_parse_error"
            self.finished = True
            return
        try:
            calls = parse_tool_calls(raw, tokenizer=tokenizer, prefix=prefix)
        except Exception as exc:
            self.generation_error = str(exc)
            self.termination_reason = "generation_or_parse_error"
            self.finished = True
            return
        self.iteration += 1
        if not calls:
            self.messages.append({"role": "assistant", "content": raw})
            self.termination_reason = "no_tool_call"
            self.finished = True
            return
        self.exchanges.extend(append_tool_exchange(self.messages, raw, calls, self.env))
        state = self.env._snapshot()
        if state.get("finalized"):
            self.termination_reason = (
                "abstained" if state.get("abstained") else "terminal_tool"
            )
            self.finished = True
        elif self.iteration >= self.max_iterations:
            self.termination_reason = "max_iterations"
            self.finished = True

    def record(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "messages": self.messages,
            "exchanges": self.exchanges,
            "rollout_state": self.env._snapshot(),
            "termination_reason": self.termination_reason,
            "candidate_status": (
                "generation_error" if self.generation_error else self.termination_reason
            ),
            "generation_error": self.generation_error,
        }


def _create_trace_rollout(
    row: Mapping[str, Any],
    cfg: dict[str, Any],
    mode: str,
    *,
    intervention: str,
    shuffled: dict[str, Any],
    model: Any,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    max_iterations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    scripted_actions: list[dict[str, Any]] | None,
    prompt_source: str,
    sample_index: int,
    seed: int,
) -> _TraceRollout | dict[str, Any]:
    rollout_cfg = cfg
    rollout_iterations = max_iterations
    if prompt_source == "reference":
        reference_budget = int(
            _reference_initial_observation(row).get("max_tool_calls") or 0
        )
        if reference_budget < 1 or reference_budget > max_iterations:
            raise ValueError(
                f"invalid per-row reference budget {reference_budget}; "
                f"runtime max_iterations={max_iterations}"
            )
        rollout_cfg = dict(cfg)
        rollout_environment = dict(cfg.get("environment") or {})
        rollout_environment["max_tool_calls"] = reference_budget
        rollout_cfg["environment"] = rollout_environment
        rollout_iterations = reference_budget
    env = _environment(
        rollout_cfg, mode, intervention=intervention, shuffled=shuffled
    )
    competitors, conditions = _condition_metadata(row)
    reset_kwargs: dict[str, Any] = {
        "target_smiles": str(row.get("target_smiles") or ""),
        "expected_precursor": "",
        "competitor_products": competitors,
        "conditions": conditions,
    }
    if mode in {"textbook", "irrelevant", "combined"}:
        reset_kwargs["frozen_textbook_result"] = _tool_result(
            row, "retrieve_textbook_guidance"
        )
    if mode in {"anchors", "combined"}:
        reset_kwargs["frozen_anchor_result"] = _tool_result(row, "retrieve_primitives")
    observation = env.reset(**reset_kwargs)
    messages = _trace_messages(
        row,
        mode,
        observation,
        prompt_source=prompt_source,
    )
    if scripted_actions is not None:
        result = scripted_rollout(env, scripted_actions, messages=messages)
        result.update(
            {
                "sample_index": sample_index,
                "seed": seed,
                "termination_reason": "scripted",
                "candidate_status": "scripted",
            }
        )
        return result
    return _TraceRollout(
        env=env,
        messages=messages,
        sample_index=sample_index,
        seed=seed,
        max_iterations=rollout_iterations,
        exchanges=[],
    )


def _run_trace_candidate(
    row: Mapping[str, Any],
    cfg: dict[str, Any],
    mode: str,
    *,
    intervention: str,
    shuffled: dict[str, Any],
    model: Any,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    max_iterations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    scripted_actions: list[dict[str, Any]] | None,
    prompt_source: str,
    sample_index: int,
    seed: int,
) -> dict[str, Any]:
    rollout = _create_trace_rollout(
        row,
        cfg,
        mode,
        intervention=intervention,
        shuffled=shuffled,
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        max_iterations=max_iterations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        scripted_actions=scripted_actions,
        prompt_source=prompt_source,
        sample_index=sample_index,
        seed=seed,
    )
    if isinstance(rollout, dict):
        return rollout
    while not rollout.finished:
        try:
            raw, prefix = _generate_response(
                model,
                tokenizer,
                rollout.messages,
                tools,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=rollout.seed + rollout.iteration,
            )
            error = ""
        except Exception as exc:
            raw, prefix, error = "", None, str(exc)
        rollout.advance(raw, prefix, error, tokenizer)
    return rollout.record()


def _run_trace_candidates_batched(
    row: Mapping[str, Any],
    cfg: dict[str, Any],
    mode: str,
    *,
    intervention: str,
    shuffled: dict[str, Any],
    model: Any,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    max_iterations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_source: str,
    sample_indices: Sequence[int],
    seeds: Sequence[int],
    microbatch_size: int,
) -> list[dict[str, Any]]:
    """Dynamically batch active candidates while retaining independent envs."""

    if len(sample_indices) != len(seeds) or not seeds:
        raise ValueError("trace candidate indices/seeds must be non-empty and aligned")
    rollouts: list[_TraceRollout] = []
    for sample_index, seed in zip(sample_indices, seeds, strict=True):
        created = _create_trace_rollout(
            row,
            cfg,
            mode,
            intervention=intervention,
            shuffled=shuffled,
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            scripted_actions=None,
            prompt_source=prompt_source,
            sample_index=sample_index,
            seed=seed,
        )
        if isinstance(created, dict):
            raise RuntimeError("unexpected scripted record in batched trace inference")
        rollouts.append(created)

    while True:
        active = [rollout for rollout in rollouts if not rollout.finished]
        if not active:
            break
        # Similar-length candidates stay adjacent because every candidate for
        # this target advances at most one turn per scheduler round.
        for start in range(0, len(active), microbatch_size):
            group = active[start : start + microbatch_size]
            generated = _generate_trace_responses_resilient(
                model,
                tokenizer,
                [rollout.request() for rollout in group],
                tools,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for rollout, (raw, prefix, error) in zip(group, generated, strict=True):
                rollout.advance(raw, prefix, error, tokenizer)
    return [rollout.record() for rollout in rollouts]


def _run_direct_candidate(
    row: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sample_index: int,
    scripted_response: str | None,
    seed: int,
) -> dict[str, Any]:
    if scripted_response is None:
        return _run_direct_candidates(
            row,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            sample_indices=[sample_index],
            seeds=[seed],
        )[0]
    return _direct_candidate_record(
        row,
        raw=scripted_response,
        sample_index=sample_index,
        seed=seed,
        sampling_batch_seed=seed,
        sampling_batch_size=1,
        generation_error="",
    )


def _direct_candidate_record(
    row: Mapping[str, Any],
    *,
    raw: str,
    sample_index: int,
    seed: int,
    sampling_batch_seed: int,
    sampling_batch_size: int,
    generation_error: str,
) -> dict[str, Any]:
    messages = _direct_messages(row)
    messages.append({"role": "assistant", "content": raw})
    return {
        "sample_index": sample_index,
        "seed": seed,
        "sampling_batch_seed": sampling_batch_seed,
        "sampling_batch_size": sampling_batch_size,
        "messages": messages,
        "prediction": raw,
        "generation_error": generation_error,
        "termination_reason": (
            "direct_generation" if not generation_error else "generation_error"
        ),
        "rollout_state": {},
    }


def _run_direct_candidates(
    row: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sample_indices: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    if len(sample_indices) != len(seeds) or not seeds:
        raise ValueError("direct candidate indices/seeds must be non-empty and aligned")
    generation_error = ""
    try:
        raw_texts, _ = _generate_responses(
            model,
            tokenizer,
            _direct_messages(row),
            [],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=seeds,
        )
    except Exception as exc:
        # A conservative recursive fallback keeps every requested candidate if
        # a chosen microbatch is too large for a particular GPU.  A singleton
        # failure is recorded in the artifact instead of aborting the shard.
        if len(seeds) > 1 and "out of memory" in str(exc).lower():
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            middle = len(seeds) // 2
            return _run_direct_candidates(
                row,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                sample_indices=sample_indices[:middle],
                seeds=seeds[:middle],
            ) + _run_direct_candidates(
                row,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                sample_indices=sample_indices[middle:],
                seeds=seeds[middle:],
            )
        raw_texts = [""] * len(seeds)
        generation_error = str(exc)
    batch_seed = seeds[0]
    return [
        _direct_candidate_record(
            row,
            raw=raw,
            sample_index=sample_index,
            seed=seed,
            sampling_batch_seed=batch_seed,
            sampling_batch_size=len(seeds),
            generation_error=generation_error,
        )
        for sample_index, seed, raw in zip(sample_indices, seeds, raw_texts, strict=True)
    ]


def _script_map(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return {"*": value}
    if not isinstance(value, dict):
        raise ValueError("scripted-actions must be a list or ID mapping")
    return dict(value)


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row_id(row) for row in read_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "trace",
            "textbook",
            "irrelevant",
            "anchors",
            "combined",
            "legacy",
            "direct",
        ],
        required=True,
    )
    parser.add_argument("--condition-name", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--device-map", default="")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help=(
            "Generation backend. vllm preserves the same prompt, tool parser, "
            "environment, and output contract while continuously batching requests."
        ),
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=0,
        help="vLLM context capacity; 0 inherits training.max_length or 16384.",
    )
    parser.add_argument("--vllm-max-num-seqs", type=int, default=128)
    parser.add_argument(
        "--vllm-disable-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching (enabled by default).",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--intervention",
        choices=[
            "none",
            "remove_tool_observations",
            "stale_tool_observations",
            "shuffle_tool_observations",
            "disable_inspect_state",
            "disable_intermediate_execution",
        ],
        default="none",
    )
    parser.add_argument(
        "--observation-mode",
        choices=[
            "action_delta",
            "compact_full_state",
            "reaction_center_delta",
            "full_state",
        ],
        default=None,
        help=(
            "Model-visible environment feedback. If omitted, trace conditions "
            "default to action_delta and legacy complete-proof defaults to full_state."
        ),
    )
    parser.add_argument("--intervention-source", type=Path)
    parser.add_argument("--samples-per-target", type=int, default=1)
    parser.add_argument(
        "--prompt-source",
        choices=["runtime", "reference"],
        default="runtime",
        help=(
            "runtime rebuilds the inference preamble; reference reuses the exact "
            "system/user training messages and hard-validates budget/schema parity"
        ),
    )
    parser.add_argument(
        "--direct-sample-batch-size",
        type=int,
        default=1,
        help=(
            "For --mode direct, generate this many sampled candidates in one "
            "model.generate call. OOM batches are bisected automatically."
        ),
    )
    parser.add_argument(
        "--trace-sample-batch-size",
        type=int,
        default=1,
        help=(
            "For interactive trace mode, dynamically batch this many active "
            "candidate trajectories on one model replica. Each candidate keeps "
            "an independent environment and RNG stream; CUDA OOM batches are "
            "bisected automatically."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--id",
        action="append",
        dest="selected_ids",
        default=[],
        help="evaluate only this stable row ID; repeat to select multiple rows",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="selected_sources",
        default=[],
        help="evaluate only rows with this metadata.mixture_source",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--prevalidated-data-sha256",
        default="",
        help=(
            "Reuse a dataset digest already checked by the parent launcher. "
            "This avoids every inference worker rereading a large shared file."
        ),
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Atomically updated small JSON progress record for this shard.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--scripted-actions", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.samples_per_target < 1:
        raise ValueError("samples-per-target must be >= 1")
    if args.progress_every < 1:
        raise ValueError("progress-every must be >= 1")
    if args.direct_sample_batch_size < 1:
        raise ValueError("direct-sample-batch-size must be >= 1")
    if args.trace_sample_batch_size < 1:
        raise ValueError("trace-sample-batch-size must be >= 1")
    if args.vllm_tensor_parallel_size < 1:
        raise ValueError("vllm-tensor-parallel-size must be >= 1")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("vllm-gpu-memory-utilization must be in (0, 1]")
    if args.vllm_max_model_len < 0:
        raise ValueError("vllm-max-model-len must be >= 0")
    if args.vllm_max_num_seqs < 1:
        raise ValueError("vllm-max-num-seqs must be >= 1")
    if (
        args.mode == "direct"
        and args.direct_sample_batch_size > 1
        and args.temperature <= 0
    ):
        raise ValueError("batched direct sampling requires temperature > 0")
    cfg = load_yaml(args.config)
    if args.mode != "direct":
        environment = dict(cfg.get("environment") or {})
        if args.observation_mode:
            environment["observation_mode"] = args.observation_mode
        else:
            environment.setdefault(
                "observation_mode",
                "full_state" if args.mode == "legacy" else "action_delta",
            )
        cfg["environment"] = environment
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("require shard_count >= 1 and 0 <= shard_index < shard_count")
    requested_sources = set(args.selected_sources)
    requested_ids = set(args.selected_ids)
    found_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    filtered_index = 0
    with args.data.open("r", encoding="utf-8") as data_handle:
        for line in data_handle:
            if not line.strip():
                continue
            # The headline launch has no source/ID filter. In that common case,
            # decide the shard before JSON decoding so each worker materializes
            # only 1/shard_count of the 727-MiB FlowER test artifact.
            if not requested_sources and not requested_ids:
                if args.limit and filtered_index >= args.limit:
                    break
                keep = filtered_index % args.shard_count == args.shard_index
                filtered_index += 1
                if keep:
                    rows.append(json.loads(line))
                continue

            row = json.loads(line)
            identifier = row_id(row)
            if requested_sources and str(
                (row.get("metadata") or {}).get("mixture_source") or ""
            ) not in requested_sources:
                continue
            if requested_ids and identifier not in requested_ids:
                continue
            found_ids.add(identifier)
            if args.limit and filtered_index >= args.limit:
                break
            if filtered_index % args.shard_count == args.shard_index:
                rows.append(row)
            filtered_index += 1
    if requested_ids:
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f"selected inference IDs were absent: {missing_ids}")
    if not rows:
        raise ValueError("inference data is empty")
    condition_name = args.condition_name or args.mode
    model_name = args.model_name or str(cfg.get("model_name_or_path") or "")
    adapter = args.adapter or str(
        cfg.get("output_dir") or cfg.get("initial_adapter_path") or ""
    )
    tools = _tools(args.mode)
    scripts = _script_map(args.scripted_actions)
    adapter_manifest = _adapter_manifest(adapter)
    model_revision = _resolve_revision(
        cfg, args.model_revision, adapter_manifest, scripted=bool(scripts)
    )
    tokenizer_revision = str(
        adapter_manifest.get("tokenizer_revision") or model_revision
    )
    adapter_hash = path_sha256(adapter) if adapter else ""
    data_hash = args.prevalidated_data_sha256 or path_sha256(args.data)
    versions = _software_versions()
    env_config = _environment_config(cfg, args.mode) if args.mode != "direct" else None
    if env_config is not None:
        budget = int(env_config.max_tool_calls)
        oversized = [
            (row_id(row), _required_tool_calls(row))
            for row in rows
            if _required_tool_calls(row) > budget
        ]
        if oversized:
            raise ValueError(
                f"{len(oversized)} rows exceed environment.max_tool_calls={budget}; "
                f"examples={oversized[:10]}"
            )
    if args.prompt_source == "reference":
        if env_config is None:
            raise ValueError("reference prompt source requires a tool environment")
        tools = _reference_tools(rows, tools)
        prompt_contract = _reference_prompt_contract(
            rows,
            tools=tools,
            env_config=env_config,
            max_iterations=args.max_iterations,
        )
    else:
        schema_payload = json.dumps(tools, sort_keys=True, separators=(",", ":"))
        prompt_contract = {
            "source": "runtime_generated_v1",
            "max_tool_calls": (
                int(env_config.max_tool_calls) if env_config is not None else None
            ),
            "max_iterations": args.max_iterations,
            "observation_mode": (
                str(env_config.observation_mode) if env_config is not None else None
            ),
            "tool_schema_sha256": hashlib.sha256(schema_payload.encode()).hexdigest(),
        }
        prompt_contract["contract_sha256"] = hashlib.sha256(
            json.dumps(prompt_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    config_hash = path_sha256(args.config) if args.config is not None else ""

    dry_payload = {
        "artifact_type": "inference_dry_run",
        "mode": args.mode,
        "condition_name": condition_name,
        "n_rows": len(rows),
        "tool_names": [
            str((item.get("function") or {}).get("name") or "") for item in tools
        ],
        "intervention": args.intervention,
        "observation_mode": (
            env_config.observation_mode if env_config is not None else None
        ),
        "scripted": bool(scripts),
        "model_name": model_name or None,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "adapter": adapter or None,
        "adapter_sha256": adapter_hash or None,
        "config_sha256": config_hash or None,
        "data_sha256": data_hash,
        "seed": args.seed,
        "candidate_selector": CANDIDATE_SELECTOR,
        "backend": args.backend,
        "prompt_contract": prompt_contract,
        "direct_sample_batch_size": args.direct_sample_batch_size,
        "trace_sample_batch_size": args.trace_sample_batch_size,
        "selected_sources": list(args.selected_sources),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return 0

    shuffled: dict[str, Any] = {}
    if args.intervention == "shuffle_tool_observations":
        if args.intervention_source is None:
            raise ValueError("shuffle intervention requires --intervention-source")
        shuffled = tool_result_pool(read_jsonl(args.intervention_source))

    model = tokenizer = None
    if not scripts:
        if not model_name:
            raise ValueError("model name is required for non-scripted inference")
        if args.backend == "vllm":
            training = dict(cfg.get("training") or {})
            vllm_max_model_len = int(
                args.vllm_max_model_len
                or training.get("max_length")
                or 16384
            )
            model, tokenizer = _load_vllm_backend(
                model_name,
                adapter,
                revision=model_revision,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=vllm_max_model_len,
                max_num_seqs=args.vllm_max_num_seqs,
                enable_prefix_caching=not args.vllm_disable_prefix_caching,
            )
        else:
            model, tokenizer = _load_model(
                model_name,
                adapter,
                revision=model_revision,
                device_map=args.device_map or None,
            )
        # The tokenizer is loaded from the same frozen model revision. Preserve the
        # immutable commit in prediction metadata instead of replacing it with a
        # mutable repository/model name such as Qwen/Qwen3-0.6B.
        tokenizer_revision = model_revision

    completed = _completed_ids(args.output) if args.resume else set()
    if args.output.exists() and not args.resume:
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume else "w"
    n_written = 0
    n_skipped = 0

    def write_progress(status: str) -> None:
        if args.progress_file is None:
            return
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "completed_targets": n_written + n_skipped,
            "written_targets": n_written,
            "resumed_targets": n_skipped,
            "assigned_targets": len(rows),
        }
        temporary = args.progress_file.with_suffix(args.progress_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.progress_file)

    write_progress("running")
    with args.output.open(output_mode, encoding="utf-8") as handle:
        for row in rows:
            identifier = row_id(row)
            if identifier in completed:
                n_skipped += 1
                if (n_written + n_skipped) % args.progress_every == 0:
                    write_progress("running")
                continue
            script = scripts.get(identifier, scripts.get("*"))
            candidates: list[dict[str, Any]] = []
            if args.mode == "direct" and script is None:
                for start in range(0, args.samples_per_target, args.direct_sample_batch_size):
                    sample_indices = list(
                        range(
                            start,
                            min(
                                start + args.direct_sample_batch_size,
                                args.samples_per_target,
                            ),
                        )
                    )
                    seeds = [
                        _candidate_seed(args.seed, identifier, sample_index)
                        for sample_index in sample_indices
                    ]
                    candidates.extend(
                        _run_direct_candidates(
                            row,
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            sample_indices=sample_indices,
                            seeds=seeds,
                        )
                    )
            elif script is None and args.trace_sample_batch_size > 1:
                sample_indices = list(range(args.samples_per_target))
                seeds = [
                    _candidate_seed(args.seed, identifier, sample_index)
                    for sample_index in sample_indices
                ]
                candidates = _run_trace_candidates_batched(
                    row,
                    cfg,
                    args.mode,
                    intervention=args.intervention,
                    shuffled=shuffled,
                    model=model,
                    tokenizer=tokenizer,
                    tools=tools,
                    max_iterations=args.max_iterations,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    prompt_source=args.prompt_source,
                    sample_indices=sample_indices,
                    seeds=seeds,
                    microbatch_size=args.trace_sample_batch_size,
                )
            else:
                for sample_index in range(args.samples_per_target):
                    seed = _candidate_seed(args.seed, identifier, sample_index)
                    if args.mode == "direct":
                        scripted_response = (
                            str(script[min(sample_index, len(script) - 1)])
                            if isinstance(script, list)
                            else str(script)
                        )
                        candidate = _run_direct_candidate(
                            row,
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            sample_index=sample_index,
                            scripted_response=scripted_response,
                            seed=seed,
                        )
                    else:
                        actions = None
                        if script is not None:
                            if not isinstance(script, list):
                                raise ValueError(
                                    f"scripted actions for {identifier} must be a list"
                                )
                            actions = [dict(item) for item in script]
                        candidate = _run_trace_candidate(
                            row,
                            cfg,
                            args.mode,
                            intervention=args.intervention,
                            shuffled=shuffled,
                            model=model,
                            tokenizer=tokenizer,
                            tools=tools,
                            max_iterations=args.max_iterations,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            scripted_actions=actions,
                            prompt_source=args.prompt_source,
                            sample_index=sample_index,
                            seed=seed,
                        )
                    candidates.append(candidate)
            selected = (
                candidates[0]
                if args.mode == "direct"
                else max(candidates, key=_rank_candidate)
            )
            rollout_state = dict(selected.get("rollout_state") or {})
            final_result = dict(rollout_state.get("final_result") or {})
            prediction = {
                "id": identifier,
                "artifact_type": "prediction",
                "prediction_status": "completed",
                "prediction_mode": args.mode,
                "condition_name": condition_name,
                "target_smiles": str(row.get("target_smiles") or ""),
                "messages": selected.get("messages") or [],
                "tools": tools,
                "rollout_state": rollout_state,
                "terminal_result": final_result,
                "prediction": selected.get("prediction") or "",
                "selected_candidate_index": int(selected.get("sample_index", 0)),
                "selected_candidate_seed": int(selected.get("seed", 0)),
                "candidates": candidates,
                "model": {
                    "base_model": model_name or "scripted",
                    "adapter": adapter or None,
                    "adapter_sha256": adapter_hash or None,
                    "model_revision": model_revision,
                    "tokenizer_revision": tokenizer_revision,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "max_iterations": args.max_iterations,
                    "samples_per_target": args.samples_per_target,
                    "direct_sample_batch_size": args.direct_sample_batch_size,
                    "trace_sample_batch_size": args.trace_sample_batch_size,
                    "seed": args.seed,
                    "candidate_selector": CANDIDATE_SELECTOR,
                    "backend": args.backend,
                    "max_tool_calls": (
                        int(env_config.max_tool_calls)
                        if env_config is not None
                        else None
                    ),
                    "prompt_source": args.prompt_source,
                    "prompt_contract_sha256": prompt_contract["contract_sha256"],
                    "tool_schema_sha256": prompt_contract["tool_schema_sha256"],
                    "config_sha256": config_hash or None,
                    "data_sha256": data_hash,
                    "software_versions": versions,
                },
                "metadata": {
                    "condition_name": condition_name,
                    "intervention": args.intervention,
                    "observation_mode": (
                        env_config.observation_mode
                        if env_config is not None
                        else None
                    ),
                    "reference_source_id": row.get("source_id"),
                    "frozen_evidence_replay": args.mode
                    in {"textbook", "irrelevant", "anchors", "combined"},
                },
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            n_written += 1
            if (n_written + n_skipped) % args.progress_every == 0:
                write_progress("running")

    manifest = {
        "artifact_type": "prediction_manifest",
        "data": str(args.data),
        "data_sha256": data_hash,
        "output": str(args.output),
        "n_predictions_written": n_written,
        "n_predictions_skipped_by_resume": n_skipped,
        "mode": args.mode,
        "condition_name": condition_name,
        "intervention": args.intervention,
        "observation_mode": (
            env_config.observation_mode if env_config is not None else None
        ),
        "base_model": model_name or "scripted",
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "adapter": adapter or None,
        "adapter_sha256": adapter_hash or None,
        "config_sha256": config_hash or None,
        "data_sha256": data_hash,
        "seed": args.seed,
        "candidate_selector": CANDIDATE_SELECTOR,
        "backend": args.backend,
        "prompt_contract": prompt_contract,
        "trace_sample_batch_size": args.trace_sample_batch_size,
        "selected_sources": list(args.selected_sources),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "software_versions": versions,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_progress("complete")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
