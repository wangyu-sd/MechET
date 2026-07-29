#!/usr/bin/env python3
"""Generate MechET MECH_ET v3 CoT predictions with Qwen (+ optional LoRA adapter)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.chat_template import build_generation_prompt
from mechet.metrics import extract_answer_from_prediction, extract_gold_answer, extract_product_from_user
from mechet.model import resolve_qwen_model_path
from mechet.sft import MECH_ET_SYSTEM_PROMPT


def _load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _build_model(base_model: str, adapter: str | None, *, use_4bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tok_path = Path(adapter).parent / "tokenizer" if adapter and (Path(adapter).parent / "tokenizer").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = None
    if use_4bit and torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


def _generate_one(
    model,
    tokenizer,
    user_content: str,
    *,
    max_new_tokens: int,
    max_input_tokens: int,
    num_beams: int,
    num_return_sequences: int,
    temperature: float,
    do_sample: bool,
):
    import torch

    prompt_messages = [
        {"role": "system", "content": MECH_ET_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = build_generation_prompt(tokenizer, prompt_messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if num_beams > 1:
        gen_kwargs.update(
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
    else:
        gen_kwargs["do_sample"] = do_sample
        if do_sample:
            gen_kwargs["temperature"] = temperature
        gen_kwargs["num_return_sequences"] = 1

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    texts: list[str] = []
    for seq in out:
        texts.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=False))
    return texts


def _to_record(row: dict, user_content: str, raw_texts: list[str]) -> dict:
    prediction = raw_texts[0] if raw_texts else ""
    return {
        "id": row.get("id"),
        "prediction": prediction,
        "candidates": [extract_answer_from_prediction(t) for t in raw_texts if extract_answer_from_prediction(t)],
        "raw_generations": raw_texts,
        "product": extract_product_from_user(user_content),
        "gold_answer": extract_gold_answer(row),
        "topology": (row.get("metadata") or {}).get("topology"),
    }


def _build_prompt(row: dict, tokenizer) -> tuple[str, str]:
    messages = row.get("messages") or []
    user = next((m for m in messages if m.get("role") == "user"), {"content": ""})
    user_content = str(user.get("content") or "")
    prompt_messages = [
        {"role": "system", "content": MECH_ET_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return user_content, build_generation_prompt(tokenizer, prompt_messages)


def _build_vllm_engine(
    base_model: str,
    *,
    max_model_len: int,
    max_num_seqs: int,
    enable_lora: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    distributed_executor_backend: str | None = None,
):
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    import inspect

    engine_args = AsyncEngineArgs(
        model=base_model,
        trust_remote_code=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    if enable_lora:
        sig = inspect.signature(AsyncEngineArgs.__init__)
        if "enable_lora" in sig.parameters:
            setattr(engine_args, "enable_lora", True)
            if "max_lora_rank" in sig.parameters:
                setattr(engine_args, "max_lora_rank", 64)
    sig = inspect.signature(AsyncEngineArgs.__init__)
    if "tensor_parallel_size" in sig.parameters and tensor_parallel_size > 1:
        setattr(engine_args, "tensor_parallel_size", tensor_parallel_size)
    if "pipeline_parallel_size" in sig.parameters and pipeline_parallel_size > 1:
        setattr(engine_args, "pipeline_parallel_size", pipeline_parallel_size)
    if distributed_executor_backend and "distributed_executor_backend" in sig.parameters:
        setattr(engine_args, "distributed_executor_backend", distributed_executor_backend)
    return AsyncLLMEngine.from_engine_args(engine_args)


def _build_lora_request(adapter_path: str | None):
    if not adapter_path:
        return None

    try:
        from vllm.lora.request import LoRARequest
    except Exception:
        return None

    adapter_path = str(Path(adapter_path))
    adapter_name = Path(adapter_path).name or "mechet_adapter"
    candidates = [
        {"lora_name": adapter_name, "lora_int_id": 1, "lora_path": adapter_path},
        {"lora_name": adapter_name, "lora_id": 1, "lora_path": adapter_path},
        {"name": adapter_name, "lora_id": 1, "path": adapter_path},
        {"lora_name": adapter_name, "lora_int_id": 1, "lora_local_path": adapter_path},
    ]
    for kwargs in candidates:
        try:
            return LoRARequest(**kwargs)
        except Exception:
            continue

    try:
        return LoRARequest(adapter_name, 1, adapter_path)
    except Exception:
        return None


def _supports_lora_request(engine) -> bool:
    import inspect

    return "lora_request" in inspect.signature(engine.generate).parameters


def _build_sampling_params(max_new_tokens: int, *, num_beams: int, num_return_sequences: int, temperature: float, do_sample: bool):
    from vllm import SamplingParams

    n = max(1, min(num_return_sequences, max(1, num_beams)))
    kwargs = {
        "max_tokens": max_new_tokens,
        "n": n,
    }
    if num_beams > 1 and not do_sample:
        kwargs["use_beam_search"] = True
        kwargs["best_of"] = num_beams
    else:
        kwargs["temperature"] = temperature if do_sample else 0.0

    return SamplingParams(**kwargs)


async def _generate_one_vllm(
    engine,
    prompt: str,
    sampling_params,
    lora_request,
    *,
    request_id: str,
) -> list[str]:
    outputs = None
    import inspect

    sig = inspect.signature(engine.generate)
    gen_kwargs = {"request_id": request_id}
    if "lora_request" in sig.parameters and lora_request is not None:
        gen_kwargs["lora_request"] = lora_request
    elif lora_request is not None:
        return []

    generator = engine.generate(prompt, sampling_params, **gen_kwargs)
    async for output in generator:
        outputs = output

    if outputs is None or not outputs.outputs:
        return []
    return [candidate.text for candidate in outputs.outputs]


async def _run_single_vllm_task(
    idx: int,
    engine,
    row: dict,
    prompt: str,
    user_content: str,
    sampling_params,
    lora_request,
) -> dict:
    raw_texts = await _generate_one_vllm(
        engine,
        prompt,
        sampling_params,
        lora_request,
        request_id=f"mechet-{row.get('id', idx)}-{idx}",
    )
    return _to_record(row, user_content, raw_texts)


async def _run_vllm_infer(
    rows: list[dict],
    tokenizer,
    engine,
    args,
    lora_request,
    handle,
) -> int:
    from tqdm.asyncio import tqdm

    sampling_params = _build_sampling_params(
        args.max_new_tokens,
        num_beams=max(1, args.num_beams),
        num_return_sequences=max(1, args.num_return_sequences),
        temperature=args.temperature,
        do_sample=args.sample,
    )

    tasks = []
    for idx, row in enumerate(rows):
        user_content, prompt = _build_prompt(row, tokenizer)
        tasks.append(_run_single_vllm_task(
            idx,
            engine,
            row,
            prompt,
            user_content,
            sampling_params,
            lora_request,
        ))

    n_written = 0
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="infer"):
        try:
            record = await coro
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
        except Exception as exc:
            print(json.dumps({"warn": "single inference failed", "error": str(exc)}))
            continue

    return n_written


def _run_transformers_infer(rows: list[dict], tokenizer, model, args, handle) -> int:
    from tqdm import tqdm

    n_written = 0
    for row in tqdm(rows, desc="infer"):
        user_content, _ = _build_prompt(row, tokenizer)
        raw_texts = _generate_one(
            model,
            tokenizer,
            user_content,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            num_beams=max(1, args.num_beams),
            num_return_sequences=max(1, args.num_return_sequences),
            temperature=args.temperature,
            do_sample=args.sample,
        )
        handle.write(json.dumps(_to_record(row, user_content, raw_texts), ensure_ascii=False) + "\n")
        n_written += 1
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/mechet_sft/valid.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/mechet_eval/generations.jsonl")
    parser.add_argument("--manifest", type=Path, default=None, help="defaults to <out>.manifest.json")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-vllm", action="store_true", help="fallback to transformers generation")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--distributed-executor-backend", type=str, default=None)
    args = parser.parse_args()

    base = args.model_path or os.environ.get("QWEN_MODEL_PATH") or resolve_qwen_model_path() or ""
    if not base or not Path(base).exists():
        print(json.dumps({"error": "missing base model; set QWEN_MODEL_PATH"}))
        return 1

    rows = _load_rows(args.data, args.limit)
    use_vllm = not args.no_vllm

    if args.tensor_parallel_size < 1:
        print(json.dumps({"error": "--tensor-parallel-size must be >= 1"}))
        return 1
    if args.pipeline_parallel_size < 1:
        print(json.dumps({"error": "--pipeline-parallel-size must be >= 1"}))
        return 1
    if not (0 < args.gpu_memory_utilization <= 1.0):
        print(json.dumps({"error": "--gpu-memory-utilization must be in (0, 1.0]"}))
        return 1

    model = None
    tokenizer = None
    llm = None
    lora_request = None

    if use_vllm:
        from transformers import AutoTokenizer

        try:
            adapter_path = str(args.adapter) if args.adapter else None
            lora_request = _build_lora_request(adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(
                base,
                trust_remote_code=True,
                local_files_only=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            llm = _build_vllm_engine(
                base,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                enable_lora=lora_request is not None,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                distributed_executor_backend=args.distributed_executor_backend,
            )
            if args.adapter and lora_request is None:
                print(json.dumps({"warn": "--adapter provided but vLLM LoRA request construction failed, using base model in vLLM"}))
            if args.adapter and lora_request is not None and not _supports_lora_request(llm):
                print(json.dumps({"warn": "vLLM build does not expose lora_request; fallback to transformers for LoRA"}))
                use_vllm = False
        except Exception as exc:
            print(json.dumps({"warn": "vLLM init failed, fallback to transformers", "error": str(exc)}))
            use_vllm = False

    if not use_vllm:
        model, tokenizer = _build_model(base, str(args.adapter) if args.adapter else None, use_4bit=not args.no_4bit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".manifest.json")
    started = datetime.now(timezone.utc).isoformat()

    with args.out.open("w", encoding="utf-8") as handle:
        if use_vllm and llm is not None and tokenizer is not None:
            n = asyncio.run(_run_vllm_infer(rows, tokenizer, llm, args, lora_request, handle))
        else:
            n = _run_transformers_infer(rows, tokenizer, model, args, handle)

    if llm is not None:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    manifest = {
        "generated_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "out": str(args.out),
        "n": n,
        "model_path": base,
        "adapter": str(args.adapter) if args.adapter else None,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "sample": args.sample,
        "use_vllm": use_vllm,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "distributed_executor_backend": args.distributed_executor_backend,
        "use_4bit": not args.no_4bit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "manifest": str(manifest_path), "n": n}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
