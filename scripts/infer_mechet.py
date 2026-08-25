#!/usr/bin/env python3
"""Run reproducible MechET rollouts, matched conditions, and H1 controls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

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
    for name in ("torch", "transformers", "peft", "trl", "datasets", "rdkit"):
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


def _trace_messages(
    row: Mapping[str, Any], mode: str, observation: str
) -> list[dict[str, Any]]:
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
    sample_index: int,
    seed: int,
) -> dict[str, Any]:
    env = _environment(cfg, mode, intervention=intervention, shuffled=shuffled)
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
    messages = _trace_messages(row, mode, observation)
    if scripted_actions is not None:
        result = scripted_rollout(env, scripted_actions, messages=messages)
        result.update(
            {
                "sample_index": sample_index,
                "seed": seed,
                "termination_reason": "scripted",
            }
        )
        return result

    exchanges: list[dict[str, Any]] = []
    termination_reason = "max_iterations"
    generation_error = ""
    for iteration in range(max_iterations):
        try:
            raw, prefix = _generate_response(
                model,
                tokenizer,
                messages,
                tools,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed + iteration,
            )
            calls = parse_tool_calls(raw, tokenizer=tokenizer, prefix=prefix)
        except Exception as exc:
            generation_error = str(exc)
            termination_reason = "generation_or_parse_error"
            break
        if not calls:
            messages.append({"role": "assistant", "content": raw})
            termination_reason = "no_tool_call"
            break
        exchanges.extend(append_tool_exchange(messages, raw, calls, env))
        state = env._snapshot()
        if state.get("finalized"):
            termination_reason = (
                "abstained" if state.get("abstained") else "terminal_tool"
            )
            break
    return {
        "sample_index": sample_index,
        "seed": seed,
        "messages": messages,
        "exchanges": exchanges,
        "rollout_state": env._snapshot(),
        "termination_reason": termination_reason,
        "generation_error": generation_error,
    }


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
        choices=["action_delta", "reaction_center_delta", "full_state"],
        default=None,
        help=(
            "Model-visible environment feedback. If omitted, trace conditions "
            "default to action_delta and legacy complete-proof defaults to full_state."
        ),
    )
    parser.add_argument("--intervention-source", type=Path)
    parser.add_argument("--samples-per-target", type=int, default=1)
    parser.add_argument(
        "--direct-sample-batch-size",
        type=int,
        default=1,
        help=(
            "For --mode direct, generate this many sampled candidates in one "
            "model.generate call. OOM batches are bisected automatically."
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
    parser.add_argument("--scripted-actions", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.samples_per_target < 1:
        raise ValueError("samples-per-target must be >= 1")
    if args.direct_sample_batch_size < 1:
        raise ValueError("direct-sample-batch-size must be >= 1")
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
    rows = read_jsonl(args.data)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("require shard_count >= 1 and 0 <= shard_index < shard_count")
    if args.selected_sources:
        requested_sources = set(args.selected_sources)
        rows = [
            row
            for row in rows
            if str((row.get("metadata") or {}).get("mixture_source") or "")
            in requested_sources
        ]
    if args.selected_ids:
        requested_ids = set(args.selected_ids)
        rows = [row for row in rows if row_id(row) in requested_ids]
        found_ids = {row_id(row) for row in rows}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f"selected inference IDs were absent: {missing_ids}")
    if args.limit:
        rows = rows[: args.limit]
    rows = [
        row
        for index, row in enumerate(rows)
        if index % args.shard_count == args.shard_index
    ]
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
    data_hash = path_sha256(args.data)
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
        "seed": args.seed,
        "candidate_selector": CANDIDATE_SELECTOR,
        "direct_sample_batch_size": args.direct_sample_batch_size,
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
    with args.output.open(output_mode, encoding="utf-8") as handle:
        for row in rows:
            identifier = row_id(row)
            if identifier in completed:
                n_skipped += 1
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
                    "seed": args.seed,
                    "candidate_selector": CANDIDATE_SELECTOR,
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
        "seed": args.seed,
        "candidate_selector": CANDIDATE_SELECTOR,
        "selected_sources": list(args.selected_sources),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "software_versions": versions,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
