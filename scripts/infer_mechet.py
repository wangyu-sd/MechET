#!/usr/bin/env python3
"""Run MechET prediction rollouts, matched evidence conditions, and H1 controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from mechet.endpoints import reference_structural_precursor
from mechet.frozen_evidence_environments import (
    FrozenAnchorTraceOwnedTRLEnvironment,
    FrozenTextbookAnchorTraceOwnedTRLEnvironment,
    FrozenTextbookTraceOwnedTRLEnvironment,
)
from mechet.knowledge_ablation import read_jsonl, row_id, write_jsonl
from mechet.knowledge_agent_env import KnowledgeAgentConfig
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


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


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


def _environment_config(
    cfg: dict[str, Any], mode: str
) -> AgentEnvConfig | KnowledgeAgentConfig:
    payload = dict(cfg.get("environment") or {})
    if mode in {"textbook", "irrelevant", "anchors", "combined"}:
        payload.setdefault(
            "textbook_corpus_path",
            str(
                cfg.get("textbook_corpus_path")
                or REPO / "knowledge/corpus/passages.jsonl"
            ),
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
        payload["require_textbook_corpus"] = mode in {"textbook", "irrelevant", "combined"}
        return KnowledgeAgentConfig(**payload)
    return AgentEnvConfig(**payload)


def _environment(
    cfg: dict[str, Any],
    mode: str,
    *,
    intervention: str,
    shuffled: dict[str, list[str]],
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
        return FrozenTextbookTraceOwnedTRLEnvironment(
            **common, **intervention_kwargs
        )
    if mode == "anchors":
        return FrozenAnchorTraceOwnedTRLEnvironment(
            **common, **intervention_kwargs
        )
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
    suffix = EVIDENCE_SUFFIX if mode in {"textbook", "irrelevant", "anchors", "combined"} else ""
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
    revision: str | None,
    device_map: str | None,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
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
) -> tuple[str, Any]:
    import torch

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
    if not isinstance(encoded, dict) or "input_ids" not in encoded:
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
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
    generated = output[0, input_length:]
    return (
        tokenizer.decode(generated, skip_special_tokens=False),
        inputs["input_ids"][0],
    )


def _rank_candidate(candidate: Mapping[str, Any]) -> tuple[int, int, int]:
    state = dict(candidate.get("rollout_state") or {})
    final = dict(state.get("final_result") or {})
    return (
        int(bool(final.get("formal_execute") or final.get("ok"))),
        -int(bool(state.get("abstained"))),
        -int(candidate.get("sample_index", 0)),
    )


def _run_trace_candidate(
    row: Mapping[str, Any],
    cfg: dict[str, Any],
    mode: str,
    *,
    intervention: str,
    shuffled: dict[str, list[str]],
    model: Any,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    max_iterations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    scripted_actions: list[dict[str, Any]] | None,
    sample_index: int,
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
        reset_kwargs["frozen_anchor_result"] = _tool_result(
            row, "retrieve_primitives"
        )
    observation = env.reset(**reset_kwargs)
    messages = _trace_messages(row, mode, observation)
    if scripted_actions is not None:
        result = scripted_rollout(env, scripted_actions, messages=messages)
        result.update({"sample_index": sample_index, "termination_reason": "scripted"})
        return result

    exchanges: list[dict[str, Any]] = []
    termination_reason = "max_iterations"
    for _ in range(max_iterations):
        raw, prefix = _generate_response(
            model,
            tokenizer,
            messages,
            tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        calls = parse_tool_calls(raw, tokenizer=tokenizer, prefix=prefix)
        if not calls:
            messages.append({"role": "assistant", "content": raw})
            termination_reason = "no_tool_call"
            break
        exchanges.extend(append_tool_exchange(messages, raw, calls, env))
        state = env._snapshot()
        if state.get("finalized"):
            termination_reason = "abstained" if state.get("abstained") else "terminal_tool"
            break
    return {
        "sample_index": sample_index,
        "messages": messages,
        "exchanges": exchanges,
        "rollout_state": env._snapshot(),
        "termination_reason": termination_reason,
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
) -> dict[str, Any]:
    messages = _direct_messages(row)
    raw = scripted_response
    if raw is None:
        raw, _ = _generate_response(
            model,
            tokenizer,
            messages,
            [],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    messages.append({"role": "assistant", "content": raw})
    return {
        "sample_index": sample_index,
        "messages": messages,
        "prediction": raw,
        "termination_reason": "direct_generation",
        "rollout_state": {},
    }


def _script_map(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return {"*": value}
    if not isinstance(value, dict):
        raise ValueError("scripted-actions must be a list or ID mapping")
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["trace", "textbook", "irrelevant", "anchors", "combined", "legacy", "direct"],
        required=True,
    )
    parser.add_argument("--condition-name", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--device-map", default="")
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
    parser.add_argument("--intervention-source", type=Path)
    parser.add_argument("--samples-per-target", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scripted-actions", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.samples_per_target < 1:
        raise ValueError("samples-per-target must be >= 1")
    cfg = load_yaml(args.config)
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("inference data is empty")
    condition_name = args.condition_name or args.mode
    model_name = args.model_name or str(cfg.get("model_name_or_path") or "")
    adapter = args.adapter or str(cfg.get("output_dir") or cfg.get("initial_adapter_path") or "")
    tools = _tools(args.mode)
    scripts = _script_map(args.scripted_actions)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "artifact_type": "inference_dry_run",
                    "mode": args.mode,
                    "condition_name": condition_name,
                    "n_rows": len(rows),
                    "tool_names": [
                        str((item.get("function") or {}).get("name") or "")
                        for item in tools
                    ],
                    "intervention": args.intervention,
                    "scripted": bool(scripts),
                    "model_name": model_name or None,
                    "adapter": adapter or None,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    shuffled: dict[str, list[str]] = {}
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
            revision=args.model_revision or None,
            device_map=args.device_map or None,
        )

    predictions: list[dict[str, Any]] = []
    for row in rows:
        identifier = row_id(row)
        script = scripts.get(identifier, scripts.get("*"))
        candidates: list[dict[str, Any]] = []
        for sample_index in range(args.samples_per_target):
            if args.mode == "direct":
                scripted_response = None
                if script is not None:
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
                )
            else:
                actions = None
                if script is not None:
                    if not isinstance(script, list):
                        raise ValueError(f"scripted actions for {identifier} must be a list")
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
                )
            candidates.append(candidate)
        selected = max(candidates, key=_rank_candidate)
        rollout_state = dict(selected.get("rollout_state") or {})
        final_result = dict(rollout_state.get("final_result") or {})
        predictions.append(
            {
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
                "candidates": candidates,
                "model": {
                    "base_model": model_name or None,
                    "adapter": adapter or None,
                    "adapter_sha256": path_sha256(adapter) if adapter else None,
                    "model_revision": args.model_revision or None,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "max_iterations": args.max_iterations,
                    "samples_per_target": args.samples_per_target,
                },
                "metadata": {
                    "condition_name": condition_name,
                    "intervention": args.intervention,
                    "reference_source_id": row.get("source_id"),
                    "frozen_evidence_replay": args.mode in {"textbook", "irrelevant", "anchors", "combined"},
                },
            }
        )

    write_jsonl(args.output, predictions)
    manifest = {
        "artifact_type": "prediction_manifest",
        "data": str(args.data),
        "data_sha256": path_sha256(args.data),
        "output": str(args.output),
        "n_predictions": len(predictions),
        "mode": args.mode,
        "condition_name": condition_name,
        "intervention": args.intervention,
        "base_model": model_name or None,
        "adapter": adapter or None,
        "adapter_sha256": path_sha256(adapter) if adapter else None,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
