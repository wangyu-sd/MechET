#!/usr/bin/env python3
"""Opt-in MechET product-start evaluation with CRN-JEPA FLOW + frozen Qwen.

Qwen is never asked to generate apply_electron_flow arguments. Qwen's existing
adapter routes IMPORT/FLOW/FINISH; only IMPORT triggers token generation.
The chemistry executor, structural endpoint scoring, and frozen denominator
remain those of MechET. This does not modify historical MechET scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.crn_graph_bridge import (
    FrozenCRNGraphPolicy, GRAPH_PRETRAIN_SHA256, file_sha256,
    generate_qwen_import, route_tool_name,
)
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from scripts.pr83_resume import evaluation_manifest, prepare_resume
from scripts.run_natural_language_value_search import (
    Action, Node, Runtime, policy_prompt, read_selected, rollout,
    validate_v2_adapter_manifest,
)


class HybridRuntime(Runtime):
    def __init__(self, args: argparse.Namespace, local_rank: int):
        super().__init__(args, local_rank)
        self.model.set_adapter("policy")
        self.graph = FrozenCRNGraphPolicy(
            root=args.crn_root, checkpoint=args.graph_checkpoint,
            expected_sha256=args.graph_sha256, device=self.device,
            policy_head_checkpoint=args.graph_rl_head,
        )
        self.timing = {"qwen_route_seconds": 0.0, "qwen_import_seconds": 0.0,
                       "graph_flow_seconds": 0.0}
        self.counts = {"qwen_routes": 0, "qwen_import_calls": 0,
                       "graph_flow_calls": 0, "qwen_flow_generations": 0}

    def proposals(self, node: Node, *, candidates: int, max_new_tokens: int,
                  compact_history: bool = False, planning_sample: bool = False):
        if candidates != 1 or planning_sample:
            raise ValueError("hybrid v1 uses greedy controller and one routing decision")
        self.last_proposal_error = ""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": policy_prompt(
                node.target, node.state, include_inventory=True,
                actions=node.actions, compact_history=compact_history,
            )},
        ]
        prompt = render_qwen_sft_tool_prefix(self.tokenizer, messages, tools=TOOLS)
        if len(self.tokenizer.encode(prompt, add_special_tokens=False)) >= self.args.max_context:
            self.last_proposal_error = "CONTEXT_BUDGET_EXCEEDED"
            return []
        t = time.perf_counter()
        kind, logits = route_tool_name(self.model, self.tokenizer, self.device, prompt)
        self.timing["qwen_route_seconds"] += time.perf_counter() - t
        self.counts["qwen_routes"] += 1
        tool_score = float(logits[kind])

        if kind == "finish_trace":
            return [Action(
                name=kind, arguments={}, raw="<qwen_routed_finish>",
                logprob=tool_score, tokens=1,
            )]
        if kind == "import_fragments":
            if len(self.tokenizer.encode(prompt, add_special_tokens=False)) + max_new_tokens > self.args.max_context:
                self.last_proposal_error = "IMPORT_CONTEXT_BUDGET_EXCEEDED"
                return []
            t = time.perf_counter()
            try:
                args, raw, tokens = generate_qwen_import(
                    self.model, self.tokenizer, self.device, prompt,
                    max_new_tokens=max_new_tokens,
                )
            except (ValueError, RuntimeError) as exc:
                self.last_proposal_error = str(exc)
                return []
            finally:
                self.timing["qwen_import_seconds"] += time.perf_counter() - t
                self.counts["qwen_import_calls"] += 1
            return [Action(name=kind, arguments=args, raw=raw,
                           logprob=tool_score, tokens=max(1, tokens))]

        t = time.perf_counter()
        try:
            events = self.graph.candidates(
                node.state, limit=self.args.graph_flow_topk,
                beam_size=self.args.graph_beam,
                source_width=self.args.graph_source_width,
                sink_width=self.args.graph_sink_width,
                move_width=self.args.graph_move_width,
            )
        except (ValueError, RuntimeError) as exc:
            self.last_proposal_error = f"GRAPH_FLOW_ERROR:{exc}"
            return []
        finally:
            self.timing["graph_flow_seconds"] += time.perf_counter() - t
            self.counts["graph_flow_calls"] += 1
        if not events:
            self.last_proposal_error = "GRAPH_NO_FLOW_CANDIDATE"
            return []
        return [
            Action(
                name="apply_electron_flow", arguments=event.arguments,
                raw="<crn_jepa_executed_flow>", logprob=event.logprob,
                tokens=max(1, len(event.moves)),  # score denominator is graph moves, not LM tokens
            )
            for event in events
        ]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True,
                   help="MechET strict train/valid JSONL; locked test requires separate approval")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="resume only the exact same selected IDs, checkpoint, and code")
    p.add_argument("--sample-reactions", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    p.add_argument("--policy-adapter", type=Path, required=True)
    p.add_argument("--compact-history", action="store_true")
    p.add_argument("--crn-root", type=Path, required=True)
    p.add_argument("--graph-checkpoint", type=Path, required=True)
    p.add_argument("--graph-sha256", default=GRAPH_PRETRAIN_SHA256)
    p.add_argument("--graph-rl-head", type=Path, default=None)
    p.add_argument("--graph-flow-topk", type=int, default=1)
    p.add_argument("--graph-beam", type=int, default=16)
    p.add_argument("--graph-source-width", type=int, default=8)
    p.add_argument("--graph-sink-width", type=int, default=8)
    p.add_argument("--graph-move-width", type=int, default=16)
    p.add_argument("--max-import-tokens", type=int, default=256)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--max-decisions", type=int, default=40)
    p.add_argument("--max-imports", type=int, default=32)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--no-4bit", action="store_true", default=True)
    return p


def prepare_args(args):
    if not 1 <= args.sample_reactions or not 1 <= args.graph_flow_topk <= 32:
        raise ValueError("invalid reaction count or graph-flow candidate budget")
    if (args.max_decisions != 40 or args.max_imports != 32 or
            args.max_context != 4096):
        raise ValueError("v1 preserves the frozen 40/32/4096 MechET budgets")
    if not 8 <= args.max_import_tokens <= 512:
        raise ValueError("invalid import token budget")
    validate_v2_adapter_manifest(
        args.policy_adapter, compact_history=args.compact_history,
        expected_model=args.model, expected_revision=args.model_revision,
    )
    # Existing MechET Runtime expects these fields; no value/pointer critic is used.
    args.value_adapter = ""
    args.value_kind = "successor_pn"
    args.pointer_head = ""
    args.matched_v2 = False   # hybrid is a new protocol, not legacy pure-Qwen v2
    args.vnext_v2_prefix = False
    args.legacy_dual_prompt = False
    args.product_only_remap = True
    args.branching = 1
    args.early_beam = 1
    args.late_beam = 1
    args.early_depth = 2
    args.value_weight = 0.0
    args.pointer_weight = 0.0
    args.reject_target_retained_finish = True
    args.planning_sample = False
    args.max_new_tokens = args.max_import_tokens
    return args


def main() -> int:
    args = prepare_args(parser().parse_args())
    import torch
    if torch.cuda.device_count() < 1:
        raise RuntimeError("hybrid evaluation requires a CUDA device")
    # The same fixed ID selection is shared by the original-Qwen control.
    rows = read_selected(args.data, args.sample_reactions, args.seed)
    if len(rows) != args.sample_reactions:
        raise ValueError("requested reactions exceed frozen source")
    qwen_adapter_sha = file_sha256(args.policy_adapter / "adapter_model.safetensors")
    graph_sha = file_sha256(args.graph_checkpoint)
    if graph_sha != args.graph_sha256:
        raise ValueError("CRN pretrained checkpoint SHA-256 differs from requested checkpoint")
    manifest = evaluation_manifest(
        selected_ids=[str(row["id"]) for row in rows],
        source_sha256=file_sha256(args.data),
        source_path=str(args.data),
        qwen_adapter_sha256=qwen_adapter_sha,
        graph_sha256=graph_sha,
        graph_rl_head_sha256=(
            file_sha256(args.graph_rl_head) if args.graph_rl_head else None
        ),
        model=args.model, model_revision=args.model_revision,
        config={
            "sample_reactions": args.sample_reactions, "seed": args.seed,
            "compact_history": args.compact_history,
            "graph_flow_topk": args.graph_flow_topk,
            "graph_beam": args.graph_beam,
            "graph_source_width": args.graph_source_width,
            "graph_sink_width": args.graph_sink_width,
            "graph_move_width": args.graph_move_width,
            "max_import_tokens": args.max_import_tokens,
            "max_context": args.max_context,
            "max_decisions": args.max_decisions,
            "max_imports": args.max_imports,
            "dtype": args.dtype, "no_4bit": args.no_4bit,
            "product_only_remap": args.product_only_remap,
            "reject_target_retained_finish": args.reject_target_retained_finish,
        },
        code_sha256={
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/run_crn_qwen_hybrid.py",
                "scripts/pr83_resume.py",
                "scripts/run_natural_language_value_search.py",
                "src/mechet/crn_graph_bridge.py",
                "src/mechet/endpoints.py",
            )
        },
    )
    saved = prepare_resume(
        args.output, manifest=manifest,
        selected_ids=manifest["selected_ids"], resume=args.resume,
    )
    output = args.output / "results.shard-00-of-01.jsonl"
    # Loading weights is not evidence of inference success.
    runtime = HybridRuntime(args, 0)
    resumed_count = len(saved)
    all_rows = list(saved)
    started = time.perf_counter()
    with output.open("a", encoding="utf-8") as handle:
        for row in rows[resumed_count:]:
            counts_before = dict(runtime.counts)
            timing_before = dict(runtime.timing)
            t0 = time.perf_counter()
            result = rollout(runtime, row, args)
            elapsed = time.perf_counter() - t0
            result["policy"] = "crn_jepa_qwen_routed_flow_v1"
            result["graph_checkpoint_sha256"] = runtime.graph.sha256
            result["qwen_adapter_sha256"] = qwen_adapter_sha
            result["wall_seconds"] = elapsed
            result["hybrid_counts"] = {
                k: runtime.counts[k] - counts_before.get(k, 0)
                for k in runtime.counts
            }
            result["hybrid_timing"] = {
                k: runtime.timing[k] - timing_before.get(k, 0.0)
                for k in runtime.timing
            }
            handle.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            handle.flush()
            all_rows.append(result)
    if len(all_rows) != len(rows):
        raise RuntimeError("hybrid evaluation ended with missing reaction IDs")
    run_elapsed = time.perf_counter() - started
    timing = {
        k: sum(float(item["hybrid_timing"][k]) for item in all_rows)
        for k in runtime.timing
    }
    counts = {
        k: sum(int(item["hybrid_counts"][k]) for item in all_rows)
        for k in runtime.counts
    }
    seconds = sum(float(item["wall_seconds"]) for item in all_rows)
    exact = sum(bool(item["top1_structural_exact"]) for item in all_rows)
    report = {
        "protocol": "opt_in_crn_jepa_qwen_routed_flow_v1",
        "scope": "selected_source_rows_not_full_benchmark",
        "evaluated": len(all_rows), "top1_exact": exact,
        "top1_rate": exact / len(all_rows),
        "total_recorded_reaction_seconds": seconds,
        "seconds_per_reaction": seconds / len(all_rows),
        "this_invocation_elapsed_seconds": run_elapsed,
        "resumed_rows": resumed_count,
        "new_rows": len(all_rows) - resumed_count,
        "timing": timing, "calls": counts,
        "qwen_model": args.model, "qwen_revision": args.model_revision,
        "qwen_adapter_sha256": qwen_adapter_sha,
        "graph_sha256": runtime.graph.sha256,
        "graph_rl_head_sha256": manifest["graph_rl_head_sha256"],
        "graph_flow_topk": args.graph_flow_topk,
        "run_manifest": "run_manifest.json",
        "note": (
            "Qwen short tool-name scoring is a new decision protocol; compare "
            "against the same IDs and candidate/time budget. Resumed records "
            "are validated against a frozen SHA-bound run manifest."
        ),
    }
    summary = args.output / "summary.json"
    summary_tmp = summary.with_suffix(".json.tmp")
    summary_tmp.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    summary_tmp.replace(summary)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
