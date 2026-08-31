# -*- coding: utf-8 -*-
"""Run one-trajectory-per-case RxnNano inference without evaluation metrics."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one RxnNano reasoning trajectory per input and stream it to JSONL"
    )
    parser.add_argument(
        "--cuda_device",
        default="1",
        help="One physical GPU index. Multiple GPUs are intentionally unsupported.",
    )
    parser.add_argument("--model_name", required=True, help="Model name or local path")
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, help="Output trajectory JSONL")
    parser.add_argument(
        "--task",
        choices=("retrosynthesis", "retrosynthesis_class", "forward_prediction"),
        default="retrosynthesis",
    )
    parser.add_argument("--prompt_type", default="1-plan")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Inputs submitted per vLLM call; each completed batch is saved immediately",
    )
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--max_model_len", type=int, default=2500)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=3407)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Continue after the complete, ordered rows already present in output",
    )
    output_mode.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    args = parser.parse_args()

    if "," in args.cuda_device or not args.cuda_device.strip().isdigit():
        parser.error("--cuda_device must contain exactly one numeric GPU index")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.max_tokens <= 0 or args.max_model_len <= 0:
        parser.error("--max_tokens and --max_model_len must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be in (0, 1]")
    return args


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def completed_row_count(output_path):
    completed = 0
    with output_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot resume: invalid JSON at {output_path}:{line_number}"
                ) from exc
            source_index = row.get("source_index")
            if source_index != completed:
                raise ValueError(
                    "Cannot resume: output source_index values must be contiguous "
                    f"from 0; expected {completed}, found {source_index}"
                )
            completed += 1
    return completed


def prepare_output(output_path, resume, overwrite, total_rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        return 0, "w"
    if overwrite:
        return 0, "w"
    if not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --resume or --overwrite."
        )

    completed = completed_row_count(output_path)
    if completed > total_rows:
        raise ValueError(
            f"Output has {completed} rows but input only has {total_rows} rows"
        )
    return completed, "a"


def prompt_messages(row, task, prompt_type, dataset_file, get_prompt_formatter):
    # The existing formatter includes an unused expected value. Supply a placeholder
    # so inference also works on unlabeled JSONL files.
    prompt_row = dict(row)
    if task in ("retrosynthesis", "retrosynthesis_class"):
        prompt_row.setdefault("reactants", "")
    else:
        prompt_row.setdefault("products", "")

    formatter = get_prompt_formatter(task)
    if task in ("retrosynthesis", "retrosynthesis_class"):
        formatted = formatter(
            prompt_row,
            task=task,
            prompt_type=prompt_type,
            dataset_file=str(dataset_file),
        )
    else:
        formatted = formatter(
            prompt_row,
            prompt_type=prompt_type,
            dataset_file=str(dataset_file),
        )
    return formatted["prompt"]


def source_value(row, task):
    if task in ("retrosynthesis", "retrosynthesis_class"):
        return row.get("product", row.get("product_unmapped", ""))
    return row.get("reactants", row.get("reactants_unmapped", ""))


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()
    sys.path.insert(0, str(PROJECT_DIR))

    # Import CUDA libraries only after selecting the requested physical GPU.
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from src.evaluation.prompt_formatters import get_prompt_formatter

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")

    rows = load_jsonl(input_path)
    start_index, output_mode = prepare_output(
        output_path, args.resume, args.overwrite, len(rows)
    )
    if start_index == len(rows):
        print(f"Output already contains all {len(rows)} inputs: {output_path}")
        return

    print(
        f"Loading {args.model_name} on physical GPU {args.cuda_device}; "
        "generating one trajectory for each of "
        f"{len(rows) - start_index}/{len(rows)} inputs"
    )
    model = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        n=1,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    started = time.perf_counter()
    with output_path.open(output_mode, encoding="utf-8") as output_handle:
        for batch_start in range(start_index, len(rows), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(rows))
            batch_rows = rows[batch_start:batch_end]
            prompts = [
                tokenizer.apply_chat_template(
                    prompt_messages(
                        row,
                        args.task,
                        args.prompt_type,
                        input_path,
                        get_prompt_formatter,
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch_rows
            ]

            print(
                f"Generating inputs {batch_start + 1}-{batch_end}/{len(rows)}...",
                flush=True,
            )
            batch_started = time.perf_counter()
            batch_outputs = model.generate(prompts, sampling_params=sampling_params)
            batch_runtime_ms = (time.perf_counter() - batch_started) * 1000.0

            for offset, (source_row, request_output) in enumerate(
                zip(batch_rows, batch_outputs)
            ):
                source_index = batch_start + offset
                if len(request_output.outputs) != 1:
                    raise RuntimeError(
                        f"Expected one generation for input {source_index}, got "
                        f"{len(request_output.outputs)}"
                    )
                generation = request_output.outputs[0]
                result = {
                    "source_index": source_index,
                    "stable_id": source_row.get("stable_id", str(source_index)),
                    "task": args.task,
                    "source": source_value(source_row, args.task),
                    "source_method": "RxnNano",
                    "checkpoint": args.model_name,
                    "trajectory": generation.text,
                    "finish_reason": generation.finish_reason,
                    "generated_token_count": len(generation.token_ids),
                    "batch_runtime_ms_per_input": batch_runtime_ms
                    / max(1, len(batch_outputs)),
                }
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")

            output_handle.flush()
            os.fsync(output_handle.fileno())
            elapsed = time.perf_counter() - started
            processed = batch_end - start_index
            rate = processed / elapsed if elapsed else 0.0
            remaining = (len(rows) - batch_end) / rate if rate else float("inf")
            print(
                f"Saved through input {batch_end}/{len(rows)} to {output_path}; "
                f"estimated remaining {remaining / 3600:.2f} h",
                flush=True,
            )

    print(f"Inference trajectories saved to {output_path}")


if __name__ == "__main__":
    main()
