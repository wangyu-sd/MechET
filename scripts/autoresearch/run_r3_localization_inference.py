#!/usr/bin/env python3
"""Infer one first-divergence index from each frozen, unmarked R3 trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from mechet.assistant_masking import render_chat
from scripts.agent_model_init import path_sha256
from scripts.autoresearch.prepare_r3_localization_prompts import PROMPT_VERSION, SYSTEM
from scripts.autoresearch.run_r3_repair_inference import verify_local_base
from scripts.autoresearch.stratified_manifest import digest
from scripts.eval_natural_language_event_local import _trim_completion


def load_inputs(prompts: Path, queries: Path, adapter: Path,
                *, expected_cases: int = 288) -> tuple[list[dict[str, Any]], dict[str, str]]:
    prompt_manifest_path = prompts.parent / "manifest.json"
    prompt_status_path = prompts.parent / "ARTIFACT_STATUS.json"
    query_manifest_path = queries.parent / "manifest.json"
    query_status_path = queries.parent / "ARTIFACT_STATUS.json"
    adapter_manifest_path = adapter / "adapter_manifest.json"
    for path in (prompts, queries, prompt_manifest_path, prompt_status_path,
                 query_manifest_path, query_status_path, adapter_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    prompt_manifest = json.loads(prompt_manifest_path.read_text())
    prompt_status = json.loads(prompt_status_path.read_text())
    query_manifest = json.loads(query_manifest_path.read_text())
    query_status = json.loads(query_status_path.read_text())
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    prompt_sha, query_sha = digest(prompts), digest(queries)
    if (prompt_manifest.get("prompts_sha256") != prompt_sha
            or prompt_status.get("prompts_sha256") != prompt_sha
            or prompt_status.get("inference_allowed") is not True
            or prompt_status.get("evaluation_allowed") is not False
            or prompt_manifest.get("queries_sha256") != query_sha
            or prompt_manifest.get("queries_manifest_sha256") != digest(query_manifest_path)
            or query_manifest.get("query_sha256") != query_sha
            or query_status.get("query_sha256") != query_sha
            or query_status.get("localization_inference_allowed") is not True
            or prompt_manifest.get("cases") != expected_cases
            or query_manifest.get("cases") != expected_cases
            or prompt_manifest.get("prompt_version") != PROMPT_VERSION
            or prompt_manifest.get("system_sha256")
            != hashlib.sha256(SYSTEM.encode()).hexdigest()):
        raise ValueError("R3 frozen localization prompt/query contract drifted")
    adapter_sha = path_sha256(adapter)
    if (adapter_manifest.get("adapter_sha256") != adapter_sha
            or adapter_manifest.get("base_model") != "Qwen/Qwen3-8B"
            or adapter_manifest.get("condition_name")
            != "flower_natural_language_event_compact_history_v2_qwen3_8b"):
        raise ValueError("R3 Stage-II adapter provenance drifted")
    prompt_rows = [json.loads(line) for line in prompts.read_text().splitlines()]
    query_rows = [json.loads(line) for line in queries.read_text().splitlines()]
    if len(prompt_rows) != expected_cases or len(query_rows) != expected_cases:
        raise ValueError("R3 localization denominator changed")
    seen: set[str] = set()
    for prompt, query in zip(prompt_rows, query_rows, strict=True):
        case_id = prompt.get("case_id")
        if (not isinstance(case_id, str) or case_id in seen
                or case_id != query.get("case_id")
                or set(prompt) != {"case_id", "prompt_version", "user_prompt",
                                   "candidate_action_count"}
                or prompt.get("prompt_version") != PROMPT_VERSION
                or not isinstance(prompt.get("user_prompt"), str)
                or prompt.get("candidate_action_count")
                != len(query["model_input"]["candidate_actions"])):
            raise ValueError("R3 localization prompt/query row alignment drifted")
        seen.add(case_id)
    return prompt_rows, {
        "prompt_sha256": prompt_sha, "query_sha256": query_sha,
        "checkpoint_identifier": str(adapter.resolve()),
        "checkpoint_sha256": adapter_sha,
        "base_model": str(adapter_manifest["base_model"]),
        "base_model_revision": str(adapter_manifest["base_model_revision"]),
    }


def prediction_row(case_id: str, generated_text: str,
                   action_count: int) -> dict[str, Any]:
    try:
        value = json.loads(generated_text.strip())
    except json.JSONDecodeError:
        value = None
    index = value.get("first_error_index") if isinstance(value, dict) and set(value) == {
        "first_error_index"} else None
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < action_count:
        return {"case_id": case_id, "generation_status": "failed",
                "predicted_failure_index": None}
    return {"case_id": case_id, "generation_status": "completed",
            "predicted_failure_index": index}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"R3 localization inference output already exists: {args.output}")
    rows, provenance = load_inputs(args.prompts, args.queries, args.adapter)
    if args.base_model_path is not None:
        verify_local_base(args.base_model_path, provenance["base_model_revision"])
        model_source = str(args.base_model_path.resolve())
    else:
        model_source = provenance["base_model"]
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("R3 localization inference requires exactly one CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, revision=provenance["base_model_revision"],
        trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model_source, revision=provenance["base_model_revision"],
        trust_remote_code=True, torch_dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    device = next(model.parameters()).device
    args.output.mkdir(parents=True)
    predictions = args.output / "predictions.jsonl"
    raw = args.output / "raw_generations.jsonl"
    counts = {"completed": 0, "failed": 0}
    with predictions.open("w", encoding="utf-8") as pred_sink, \
            raw.open("w", encoding="utf-8") as raw_sink:
        for number, row in enumerate(rows, 1):
            started = time.monotonic()
            rendered = render_chat(tokenizer, [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["user_prompt"]},
            ], add_generation_prompt=True)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_tokens = int(encoded["input_ids"].shape[1])
            if input_tokens + args.max_new_tokens > args.max_context:
                raise ValueError(f"R3 localization context budget exceeded: {row['case_id']}")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            ids = _trim_completion(
                [int(item) for item in generated[0, input_tokens:].tolist()], tokenizer)
            # _trim_completion retains its stop token. Strip that ChatML token
            # before strict JSON parsing, while preserving the raw text itself.
            completion = tokenizer.decode(ids, skip_special_tokens=True)
            prediction = prediction_row(
                row["case_id"], completion, row["candidate_action_count"])
            counts[prediction["generation_status"]] += 1
            pred_sink.write(json.dumps(prediction, sort_keys=True) + "\n")
            pred_sink.flush()
            raw_sink.write(json.dumps({
                "case_id": row["case_id"], "generated_text": completion,
                "generated_tokens": len(ids), "input_tokens": input_tokens,
                "seconds": round(time.monotonic() - started, 3),
            }, sort_keys=True, ensure_ascii=False) + "\n")
            raw_sink.flush()
            print(f"[meteor-r3-localization] {number}/{len(rows)} "
                  f"status={prediction['generation_status']} tokens={len(ids)} "
                  f"seconds={time.monotonic()-started:.1f}", flush=True)
    sidecar = {
        "artifact_type": "r3_unmarked_stageii_localization_predictions_manifest_v1",
        "predictions_sha256": digest(predictions),
        "query_sha256": provenance["query_sha256"],
        "prompt_sha256": provenance["prompt_sha256"],
        "checkpoint_identifier": provenance["checkpoint_identifier"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "base_model_local_path": model_source if args.base_model_path else None,
        "base_model_revision": provenance["base_model_revision"],
        "input_fields": ["model_input"],
        "prediction_semantics": "zero_based_first_reference_divergence_index_v1",
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "counts": counts, "raw_generations_sha256": digest(raw),
    }
    (args.output / "predictions.jsonl.manifest.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
