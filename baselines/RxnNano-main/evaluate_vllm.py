#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-GPU vLLM evaluation for RxnNano.

Data parallelism is the default: one subprocess and one model replica per GPU,
with deterministic strided dataset sharding and ordered result merging. Tensor
parallelism is also available for models that do not fit on one GPU.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5 models with multi-GPU vLLM inference"
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="1",
        help="Comma-separated physical GPU IDs, for example '2,3,6,7'.",
    )
    parser.add_argument(
        "--parallel_mode",
        choices=("data", "tensor"),
        default="data",
        help=(
            "data: replicate the model and shard examples across GPUs; "
            "tensor: shard one model across all selected GPUs"
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU memory made available to vLLM.",
    )
    parser.add_argument(
        "--keep_worker_outputs",
        action="store_true",
        help="Keep temporary per-GPU metrics and predictions after a successful merge.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="retrosynthesis",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--retrosynthesis_dataset",
        type=str,
        default="test_50K.jsonl",
        help="Dataset file path for retrosynthesis",
    )
    parser.add_argument(
        "--retrosynthesis_class_dataset",
        type=str,
        default="test_typed.jsonl",
        help="Dataset file path for retrosynthesis_class",
    )
    parser.add_argument(
        "--forward_prediction_dataset",
        type=str,
        default="test_480K.jsonl",
        help="Dataset file path for forward_prediction",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="1-plan",
        help="System prompt type from available prompt types",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of sequences retained after deduplication and reranking",
    )
    parser.add_argument(
        "--generate_n",
        type=int,
        default=50,
        help="Number of initial sequences generated for each input",
    )
    parser.add_argument(
        "--top_k_metrics",
        type=str,
        default="1,3,5,10",
        help="Comma-separated top-k accuracies to report",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Outer batch size passed to vLLM inference",
    )
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Maximum generated tokens per candidate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Frozen vLLM sampling seed",
    )
    parser.add_argument(
        "--predictions_output",
        type=str,
        default=None,
        help="Optional merged shared-contract predictions JSONL path",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Optional merged evaluation metrics JSON path",
    )
    parser.add_argument(
        "--disable_logging_csv",
        action="store_true",
        help="Disable detailed per-sample logs and CSV output.",
    )

    # Internal flags used by data-parallel child processes.
    parser.add_argument("--_worker_rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_worker_count", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def parse_cuda_devices(value):
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--cuda_device must contain at least one GPU ID")
    if len(set(devices)) != len(devices):
        raise ValueError(f"--cuda_device contains duplicate GPU IDs: {value}")
    return devices


def selected_tasks(value):
    return [task.strip() for task in value.split(",") if task.strip()]


def task_output_path(base_path, task, task_count, default_suffix):
    path = Path(base_path)
    if task_count == 1:
        return path
    suffix = path.suffix or default_suffix
    return path.with_name(f"{path.stem}.{task}{suffix}")


def worker_metrics_base(parts_dir, rank):
    return parts_dir / f"worker-{rank}.metrics.json"


def worker_predictions_base(parts_dir, rank):
    return parts_dir / f"worker-{rank}.predictions.jsonl"


def build_worker_command(args, device, rank, worker_count, parts_dir):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--cuda_device",
        device,
        "--parallel_mode",
        "data",
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--model_name",
        args.model_name,
        "--tasks",
        args.tasks,
        "--retrosynthesis_dataset",
        args.retrosynthesis_dataset,
        "--retrosynthesis_class_dataset",
        args.retrosynthesis_class_dataset,
        "--forward_prediction_dataset",
        args.forward_prediction_dataset,
        "--prompt_type",
        args.prompt_type,
        "--n",
        str(args.n),
        "--generate_n",
        str(args.generate_n),
        "--top_k_metrics",
        args.top_k_metrics,
        "--batch_size",
        str(args.batch_size),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--max_tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--metrics_output",
        str(worker_metrics_base(parts_dir, rank)),
        "--disable_logging_csv",
        "--_worker_rank",
        str(rank),
        "--_worker_count",
        str(worker_count),
    ]
    if args.predictions_output:
        command.extend(
            ["--predictions_output", str(worker_predictions_base(parts_dir, rank))]
        )
    return command


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_metrics(part_paths, top_k_values, devices):
    parts = [read_json(path) for path in part_paths]
    if not parts:
        raise ValueError("No worker metric files were produced")

    identity_keys = (
        "task",
        "prompt_type",
        "dataset",
        "model",
        "generate_n",
        "n",
        "top_k_metrics",
    )
    reference = parts[0]
    for part in parts[1:]:
        for key in identity_keys:
            if part.get(key) != reference.get(key):
                raise ValueError(
                    f"Worker metric mismatch for {key}: "
                    f"{reference.get(key)!r} != {part.get(key)!r}"
                )

    result = dict(reference)
    total_rows = sum(int(part.get("prediction_rows", 0)) for part in parts)
    result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    result["prediction_rows"] = total_rows

    for k in top_k_values:
        correct_key = f"top{k}_correct"
        accuracy_key = f"top{k}_accuracy"
        correct = sum(
            int(
                part.get(
                    correct_key,
                    round(
                        float(part.get(accuracy_key, 0.0))
                        * int(part.get("prediction_rows", 0))
                    ),
                )
            )
            for part in parts
        )
        result[correct_key] = correct
        result[accuracy_key] = correct / total_rows if total_rows else 0.0

    result["parallel"] = {
        "mode": "data",
        "world_size": len(devices),
        "physical_cuda_devices": devices,
        "sharding": "strided_by_original_row_index",
        "worker_prediction_rows": [
            int(part.get("prediction_rows", 0)) for part in parts
        ],
    }
    return result


def merge_predictions(part_paths, output_path, worker_count):
    indexed_rows = []
    for rank, path in enumerate(part_paths):
        with path.open(encoding="utf-8") as handle:
            worker_rows = [json.loads(line) for line in handle if line.strip()]
        indexed_rows.extend(
            (rank + local_index * worker_count, row)
            for local_index, row in enumerate(worker_rows)
        )

    indexed_rows.sort(key=lambda item: item[0])
    observed_indices = [index for index, _ in indexed_rows]
    expected_indices = list(range(len(indexed_rows)))
    if observed_indices != expected_indices:
        raise ValueError(
            "Worker prediction shards do not cover a contiguous set of source rows"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in indexed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(indexed_rows)


def merge_worker_outputs(args, devices, parts_dir):
    tasks = selected_tasks(args.tasks)
    top_k_values = [int(value) for value in args.top_k_metrics.split(",")]

    for task in tasks:
        metric_parts = [
            task_output_path(
                worker_metrics_base(parts_dir, rank), task, len(tasks), ".json"
            )
            for rank in range(len(devices))
        ]
        missing_metrics = [str(path) for path in metric_parts if not path.is_file()]
        if missing_metrics:
            raise FileNotFoundError(
                "Missing worker metric files: " + ", ".join(missing_metrics)
            )

        metrics = aggregate_metrics(metric_parts, top_k_values, devices)
        if args.metrics_output:
            metrics_path = task_output_path(
                args.metrics_output, task, len(tasks), ".json"
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Merged metrics saved to {metrics_path}")

        if args.predictions_output:
            prediction_parts = [
                task_output_path(
                    worker_predictions_base(parts_dir, rank),
                    task,
                    len(tasks),
                    ".jsonl",
                )
                for rank in range(len(devices))
            ]
            missing_predictions = [
                str(path) for path in prediction_parts if not path.is_file()
            ]
            if missing_predictions:
                raise FileNotFoundError(
                    "Missing worker prediction files: "
                    + ", ".join(missing_predictions)
                )
            predictions_path = task_output_path(
                args.predictions_output, task, len(tasks), ".jsonl"
            )
            row_count = merge_predictions(
                prediction_parts, predictions_path, len(devices)
            )
            if row_count != metrics["prediction_rows"]:
                raise ValueError(
                    f"Merged prediction count {row_count} does not match metrics "
                    f"count {metrics['prediction_rows']}"
                )
            print(f"Merged predictions saved to {predictions_path}")

        print(
            f"{task.upper()} merged data-parallel results: "
            + ", ".join(
                f"top-{k}={metrics[f'top{k}_accuracy']:.4f}"
                for k in top_k_values
            )
        )


def choose_parts_parent(args):
    if args.metrics_output:
        return Path(args.metrics_output).expanduser().resolve().parent
    if args.predictions_output:
        return Path(args.predictions_output).expanduser().resolve().parent
    return PROJECT_DIR


def run_data_parallel(args, devices):
    parts_parent = choose_parts_parent(args)
    parts_parent.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(
        tempfile.mkdtemp(prefix=".rxnnano-eval-parts-", dir=str(parts_parent))
    )
    print(
        f"Starting RxnNano data-parallel evaluation on {len(devices)} GPUs: "
        f"{','.join(devices)}"
    )
    print(f"Temporary worker outputs: {parts_dir}")

    processes = []
    try:
        for rank, device in enumerate(devices):
            command = build_worker_command(
                args, device, rank, len(devices), parts_dir
            )
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            print(f"Launching worker {rank}/{len(devices)} on physical GPU {device}")
            processes.append(
                subprocess.Popen(command, cwd=PROJECT_DIR, env=environment)
            )

        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        print("Interrupted; terminating RxnNano workers", file=sys.stderr)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        print(f"Worker outputs retained at {parts_dir}", file=sys.stderr)
        return 130
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        print(f"Worker outputs retained at {parts_dir}", file=sys.stderr)
        raise

    failed_workers = [
        rank for rank, return_code in enumerate(return_codes) if return_code != 0
    ]
    if failed_workers:
        print(
            f"Workers failed: {failed_workers}; outputs retained at {parts_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        merge_worker_outputs(args, devices, parts_dir)
    except Exception:
        print(f"Merge failed; worker outputs retained at {parts_dir}", file=sys.stderr)
        raise

    if args.keep_worker_outputs:
        print(f"Worker outputs retained at {parts_dir}")
    else:
        shutil.rmtree(parts_dir)
    return 0


def create_config(args):
    from src.config.settings import EvaluationConfig

    return EvaluationConfig(
        cuda_device=args.cuda_device,
        model_name=args.model_name,
        tasks=args.tasks,
        retrosynthesis_dataset=args.retrosynthesis_dataset,
        retrosynthesis_class_dataset=args.retrosynthesis_class_dataset,
        forward_prediction_dataset=args.forward_prediction_dataset,
        prompt_type=args.prompt_type,
        n=args.n,
        generate_n=args.generate_n,
        top_k_metrics=args.top_k_metrics,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        predictions_output=args.predictions_output,
        metrics_output=args.metrics_output,
        disable_logging_csv=args.disable_logging_csv,
    )


def load_model_and_tokenizer(config, tensor_parallel_size, gpu_memory_utilization):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    print(
        f"Loading model: {config.model_name} "
        f"(tensor_parallel_size={tensor_parallel_size})"
    )
    model = LLM(
        model=config.model_name,
        max_model_len=config.max_seq_length,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    return model, tokenizer


def run_evaluation_process(args, devices):
    # This must happen before importing torch or vLLM.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    sys.path.insert(0, str(PROJECT_DIR / "src"))

    from src.data.dataset_loader import get_dataset_files, load_dataset_for_task
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.prompt_formatters import get_prompt_formatter
    from src.utils.logging import setup_logging

    try:
        config = create_config(args)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    tensor_parallel_size = len(devices) if args.parallel_mode == "tensor" else 1
    model, tokenizer = load_model_and_tokenizer(
        config, tensor_parallel_size, args.gpu_memory_utilization
    )
    dataset_files = get_dataset_files(config)
    evaluator = Evaluator(model, tokenizer, config)
    had_error = False

    for task in config.selected_tasks:
        dataset_file = dataset_files[task]
        dataset_name = Path(dataset_file).stem
        logger = None
        log_file = None

        try:
            logger, log_file = setup_logging(
                task,
                config.model_name,
                dataset_name,
                config,
                config.disable_logging_csv,
            )
            dataset = load_dataset_for_task(task, dataset_file)
            original_size = len(dataset)
            if args._worker_count > 1:
                shard_indices = list(
                    range(args._worker_rank, original_size, args._worker_count)
                )
                dataset = dataset.select(shard_indices)
                print(
                    f"Worker {args._worker_rank}/{args._worker_count}: "
                    f"selected {len(dataset)}/{original_size} {task} rows"
                )

            format_func = get_prompt_formatter(task)
            if task in ("retrosynthesis", "retrosynthesis_class"):
                eval_dataset = dataset.map(
                    lambda example: format_func(
                        example,
                        task=task,
                        prompt_type=config.prompt_type,
                        dataset_file=dataset_file,
                    ),
                    batched=False,
                )
            else:
                eval_dataset = dataset.map(
                    lambda example: format_func(
                        example,
                        prompt_type=config.prompt_type,
                        dataset_file=dataset_file,
                    ),
                    batched=False,
                )

            results = evaluator.evaluate_dataset(
                task=task,
                dataset=eval_dataset,
                dataset_name=dataset_name,
                prompt_type=config.prompt_type,
                batch_size=config.batch_size,
                logger=logger,
            )
            for k in config.top_k_metrics_list:
                results[f"top{k}_correct"] = int(
                    round(results[f"top{k}_accuracy"] * results["prediction_rows"])
                )
            results["parallel"] = {
                "mode": args.parallel_mode,
                "tensor_parallel_size": tensor_parallel_size,
                "worker_rank": args._worker_rank,
                "worker_count": args._worker_count,
                "visible_cuda_devices": devices,
            }

            if config.metrics_output:
                metrics_path = task_output_path(
                    config.metrics_output,
                    task,
                    len(config.selected_tasks),
                    ".json",
                )
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(
                    json.dumps(results, indent=2) + "\n", encoding="utf-8"
                )
                print(f"Metrics saved to {metrics_path}")

            if not config.disable_logging_csv:
                print(f"Full log for {task} saved to {log_file}")
                if logger:
                    logger.info(f"Full log for {task} saved to {log_file}")
            else:
                print("Detailed logging and CSV saving are disabled.")
        except Exception as error:
            had_error = True
            print(f"Error evaluating task {task}: {error}", file=sys.stderr)
            if logger:
                logger.exception(f"Error evaluating task {task}: {error}")

    return 1 if had_error else 0


def main():
    args = build_parser().parse_args()
    try:
        devices = parse_cuda_devices(args.cuda_device)
        if not 0.0 < args.gpu_memory_utilization <= 1.0:
            raise ValueError("--gpu_memory_utilization must be in (0, 1]")
        if args._worker_count < 1:
            raise ValueError("internal worker count must be positive")
        if not 0 <= args._worker_rank < args._worker_count:
            raise ValueError("internal worker rank is outside the worker count")
        if args._worker_count > 1 and len(devices) != 1:
            raise ValueError("a data-parallel worker must receive exactly one GPU")
    except ValueError as error:
        print(f"Argument error: {error}", file=sys.stderr)
        return 2

    is_data_parallel_parent = (
        args.parallel_mode == "data"
        and len(devices) > 1
        and args._worker_count == 1
    )
    if is_data_parallel_parent:
        return run_data_parallel(args, devices)
    return run_evaluation_process(args, devices)


if __name__ == "__main__":
    raise SystemExit(main())
