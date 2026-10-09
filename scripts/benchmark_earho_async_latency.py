#!/usr/bin/env python3
"""Measure Stage-III rollout latency with one loaded vLLM async engine.

This is a bounded performance diagnostic, not a benchmark accuracy run.
Both passes reuse the same model, LoRA, reaction IDs, engine, and request seeds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import scripts.natural_language_anchor_branch_stage as stage
from scripts.earho_v2_protocol import replay_reference


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int((len(ordered) - 1) * fraction), len(ordered) - 1)]


def _summary(name: str, wall_s: float, groups: list[dict]) -> dict:
    times = [float(row["elapsed_wall_s"]) for row in groups]
    model_wait = [float(row.get("model_wait_wall_s", 0.0)) for row in groups]
    cpu = [float(row["collector_thread_cpu_s"]) for row in groups]
    fields = ("probe_wall_s", "first_generate_wall_s", "continuation_wall_s")
    return {
        "name": name,
        "reaction_count": len(groups),
        "wall_s": round(wall_s, 3),
        "reactions_per_s": round(len(groups) / wall_s, 4),
        "reaction_wall_p50_s": round(statistics.median(times), 3),
        "reaction_wall_p90_s": round(_percentile(times, 0.9), 3),
        "model_wait_sum_s": round(sum(model_wait), 3),
        "model_wait_p50_s": round(statistics.median(model_wait), 3),
        "model_wait_p90_s": round(_percentile(model_wait, 0.9), 3),
        "collector_thread_cpu_sum_s": round(sum(cpu), 3),
        "non_model_reaction_wall_sum_s": round(sum(times) - sum(model_wait), 3),
        **{field + "_sum": round(sum(float(row[field]) for row in groups), 3)
           for field in fields},
        "model_calls": sum(int(row.get("model_calls", 0)) for row in groups),
        "model_requests": sum(int(row.get("model_requests", 0)) for row in groups),
        "model_prompt_tokens": sum(int(row.get("model_prompt_tokens", 0)) for row in groups),
        "model_output_tokens": sum(int(row.get("model_output_tokens", 0)) for row in groups),
        "model_output_tokens_per_wall_s": round(
            sum(int(row.get("model_output_tokens", 0)) for row in groups) / wall_s, 2
        ),
        "model_queue_sum_s": round(sum(float(row.get("model_queue_s", 0.0)) for row in groups), 3),
        "model_ttft_sum_s": round(sum(float(row.get("model_ttft_s", 0.0)) for row in groups), 3),
        "model_decode_sum_s": round(sum(float(row.get("model_decode_s", 0.0)) for row in groups), 3),
        "collector_errors": sum("collector_error" in row for row in groups),
    }


def _collector_args(args, data: Path, output: Path, concurrency: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=str(data), output=str(output), model=str(args.model), adapter=str(args.adapter),
        rank=0, world_size=1, k=args.k, seed=17, round_index=0, frontier=2,
        full_episode_fraction=1.0, invalid_penalty=0.1, wrong_terminal_penalty=0.5,
        endpoint_similarity_weight=0.45, first_successor_progress_weight=0.25,
        nonexact_reward_ceiling=0.01, target_retained_penalty=0.5,
        reference_first_successor_weight=0.0, endpoint_metric="structural",
        value_adapter=None, value_kind="successor_pn",
        continuation_candidates_per_mode=args.continuation_candidates,
        continuation_temperature=0.7, value_score_weight=1.0,
        policy_score_weight=0.1, continuation_beam_width=args.beam_width,
        success_gated_advantages=True, temperature=1.0,
        max_new_tokens=args.max_new_tokens, max_context=args.max_context,
        max_decisions=args.max_decisions, max_imports=32,
        evaluation=True, full_only=True, reject_target_retained_finish=True,
        legacy_dual_prompt=False, protocol_v2=True,
        state_only_observation=False, vnext_credit=None,
        vnext_private_reference_credit=False, async_reactions=concurrency,
        engine_mode="eager", dtype=args.dtype,
    )


def _select_rows(source: Path, count: int, output: Path) -> dict:
    selected = []
    skipped = []
    started = time.monotonic()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            try:
                replay_reference(row, row["earho_v2_reference_decisions"], max_imports=32)
            except (KeyError, ValueError) as exc:
                skipped.append({"id": row.get("source_id"), "error": str(exc)})
                continue
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} locally replayable reactions, wanted {count}")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    return {
        "selection_s": round(time.monotonic() - started, 3),
        "selected_ids": [row["source_id"] for row in selected],
        "skipped_due_to_local_replay": skipped,
    }


async def _run(args, data: Path, output_dir: Path) -> dict:
    started = time.monotonic()
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    import vllm

    import_s = time.monotonic() - started
    if vllm.__version__ != "0.8.5":
        raise ValueError(f"expected vLLM 0.8.5, got {vllm.__version__}")
    engine_args = stage._engine_kwargs(_collector_args(args, data, output_dir / "unused.jsonl", 1))
    engine_args["enable_prefix_caching"] = False  # avoid second-pass cache advantage
    engine_started = time.monotonic()
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
    try:
        tokenizer = await engine.get_tokenizer()
        engine_init_s = time.monotonic() - engine_started
        bridge = stage.AsyncVLLMBridge(engine, asyncio.get_running_loop(), seed=17, rank=0)
        original_log = stage.log
        passes = []
        try:
            for name, concurrency in (("sequential", 1), ("concurrent4", 4)):
                groups: list[dict] = []

                def capture(**fields):
                    if fields.get("stage") == "nl-anchor-group":
                        groups.append(fields)
                        if len(groups) % 4 == 0:
                            print(json.dumps({"stage": "latency-progress", "mode": name,
                                              "completed": len(groups)}), flush=True)

                stage.log = capture
                collector_args = _collector_args(
                    args, data, output_dir / f"{name}.jsonl", concurrency,
                )
                print(json.dumps({"stage": "latency-start", "mode": name}), flush=True)
                run_started = time.monotonic()
                await asyncio.to_thread(
                    stage._collect_initialized, collector_args, bridge, tokenizer,
                    SamplingParams, LoRARequest,
                )
                elapsed = time.monotonic() - run_started
                if len(groups) != args.reactions:
                    raise ValueError(f"{name}: only {len(groups)}/{args.reactions} groups logged")
                result = _summary(name, elapsed, groups)
                result["output"] = collector_args.output
                passes.append(result)
                print(json.dumps({"stage": "latency-done", **result}), flush=True)
        finally:
            stage.log = original_log
        return {
            "vllm_import_s": round(import_s, 3),
            "engine_init_s": round(engine_init_s, 3),
            "model": str(args.model), "adapter": str(args.adapter),
            "device_dtype": args.dtype, "sample_size": args.reactions,
            "max_new_tokens": args.max_new_tokens,
            "max_decisions": args.max_decisions,
            "prefix_cache_enabled": False,
            "passes": passes,
        }
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        else:
            engine.shutdown_background_loop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reactions", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-decisions", type=int, default=4)
    parser.add_argument("--continuation-candidates", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    if not 1 <= args.reactions <= 64 or args.k < 2:
        raise ValueError("benchmark requires 1-64 reactions and K>=2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("VLLM_CACHE_ROOT", str(args.output_dir / "vllm-cache"))
    os.environ.setdefault("TMPDIR", str(args.output_dir))
    selection = _select_rows(args.source, args.reactions, args.output_dir / "source.jsonl")
    print(json.dumps({"stage": "latency-selection", **selection}), flush=True)
    report = asyncio.run(_run(args, args.output_dir / "source.jsonl", args.output_dir))
    report["selection"] = selection
    (args.output_dir / "latency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"stage": "latency-report", "path": str(args.output_dir / "latency_report.json")}), flush=True)


if __name__ == "__main__":
    main()
