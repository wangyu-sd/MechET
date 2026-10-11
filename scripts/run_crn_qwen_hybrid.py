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
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.crn_graph_bridge import (
    FrozenCRNGraphPolicy, GRAPH_PRETRAIN_SHA256, file_sha256,
    generate_qwen_import,
    graph_event_coordinates, route_tool_name, route_tool_name_greedy,
)
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
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
                       "graph_flow_calls": 0, "qwen_flow_generations": 0,
                       "qwen_route_tokens": 0}

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
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens + max_new_tokens > self.args.max_context:
            self.last_proposal_error = "CONTEXT_BUDGET_EXCEEDED"
            return []
        t = time.perf_counter()
        route = None
        try:
            if self.args.tool_route == "greedy_name":
                route = route_tool_name_greedy(
                    self.model, self.tokenizer, self.device, prompt,
                    max_route_tokens=max_new_tokens,
                )
                kind, tool_score = route.name, route.logprob
                self.counts["qwen_route_tokens"] += route.name_tokens
            else:
                kind, logits = route_tool_name(
                    self.model, self.tokenizer, self.device, prompt,
                )
                tool_score = float(logits[kind])
        except (ValueError, RuntimeError) as exc:
            self.last_proposal_error = (
                str(exc) if str(exc).startswith("QWEN_IMPORT_INVALID:")
                else f"QWEN_ROUTE_ERROR:{exc}"
            )
            return []
        route_elapsed = time.perf_counter() - t
        if route is not None and kind == "import_fragments":
            self.timing["qwen_import_seconds"] += route_elapsed
        else:
            self.timing["qwen_route_seconds"] += route_elapsed
        self.counts["qwen_routes"] += 1

        if kind == "finish_trace":
            return [Action(
                name=kind, arguments={}, raw="<qwen_routed_finish>",
                logprob=tool_score, tokens=1,
            )]
        if kind == "import_fragments":
            if route is not None:
                if route.import_arguments is None:
                    self.last_proposal_error = "QWEN_IMPORT_MISSING_ARGUMENTS"
                    return []
                args, raw, tokens = route.import_arguments, route.raw, len(route.token_ids)
            else:
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
            event_depth, remaining_budget = graph_event_coordinates(
                node.actions, self.args.graph_initial_budget,
            )
            events = self.graph.candidates(
                node.state, limit=self.args.graph_flow_topk,
                event_depth=event_depth, remaining_budget=remaining_budget,
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
    p.add_argument("--sample-reactions", type=int, default=128)
    p.add_argument("--source-sha256", default="",
                   help="optional pinned SHA-256 for the complete source JSONL")
    p.add_argument("--resume", action="store_true",
                   help="resume only an identical, previously initialized shard")
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
    p.add_argument("--graph-initial-budget", type=int, default=4,
                   help="non-oracle electron-event budget supplied to CRN-JEPA")
    p.add_argument("--tool-route", choices=("greedy_name", "prefix_nll"),
                   default="greedy_name",
                   help="greedy_name matches historical full-call tool selection")
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


def selected_shard(rows, rank: int, world: int):
    """Partition one frozen, seed-selected reaction list without dropping IDs."""
    if world < 1 or not 0 <= rank < world:
        raise ValueError("invalid distributed rank/world")
    return [row for index, row in enumerate(rows) if index % world == rank]


def completed_ids(path: Path, expected_ids: set[str]) -> set[str]:
    """Read only complete JSONL records; fail on duplicate or alien results."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            identifier = str(row["id"])
            if identifier not in expected_ids or identifier in done:
                raise ValueError(f"invalid existing result at line {line_number}: {identifier}")
            done.add(identifier)
    return done


def prepare_args(args):
    if not 1 <= args.sample_reactions or not 1 <= args.graph_flow_topk <= 32:
        raise ValueError("invalid reaction count or graph-flow candidate budget")
    if (args.max_decisions != 40 or args.max_imports != 32 or
            args.max_context != 4096):
        raise ValueError("v1 preserves the frozen 40/32/4096 MechET budgets")
    if not 8 <= args.max_import_tokens <= 512:
        raise ValueError("invalid import token budget")
    if not 1 <= args.graph_initial_budget <= 32:
        raise ValueError("invalid graph initial budget")
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
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world < 1 or not 0 <= rank < world:
        raise ValueError("invalid distributed rank/world")
    import torch
    if torch.cuda.device_count() <= local_rank:
        raise RuntimeError("hybrid evaluation requires one CUDA device per local rank")
    # One graph+Qwen policy per GPU; no hidden use of GT to select actions.
    rows = read_selected(args.data, args.sample_reactions, args.seed)
    if len(rows) != args.sample_reactions:
        raise ValueError("requested reactions exceed frozen source")
    if args.source_sha256 and file_sha256(args.data) != args.source_sha256:
        raise ValueError("evaluation source SHA-256 mismatch")
    selected = selected_shard(rows, rank, world)
    expected_ids = {str(row["id"]) for row in selected}
    if len(expected_ids) != len(selected):
        raise ValueError("duplicate IDs in selected evaluation shard")
    qwen_adapter_sha = file_sha256(args.policy_adapter / "adapter_model.safetensors")
    run_contract = {
        "source": str(args.data.resolve()), "source_sha256": args.source_sha256,
        "sample_reactions": args.sample_reactions, "seed": args.seed,
        "rank": rank, "world": world,
        "model": args.model, "model_revision": args.model_revision,
        "policy_adapter_sha256": qwen_adapter_sha,
        "compact_history": args.compact_history,
        "graph_checkpoint_sha256": args.graph_sha256,
        "graph_rl_head_sha256": file_sha256(args.graph_rl_head) if args.graph_rl_head else None,
        "graph_flow_topk": args.graph_flow_topk,
        "graph_initial_budget": args.graph_initial_budget,
        "tool_route": args.tool_route,
        "greedy_import_decode": "single_generate_to_eos_v2",
        "graph_beam": args.graph_beam, "graph_source_width": args.graph_source_width,
        "graph_sink_width": args.graph_sink_width, "graph_move_width": args.graph_move_width,
        "max_import_tokens": args.max_import_tokens,
        "max_context": args.max_context, "max_decisions": args.max_decisions,
        "max_imports": args.max_imports,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output / f"contract.shard-{rank:02d}-of-{world:02d}.json"
    output = args.output / f"results.shard-{rank:02d}-of-{world:02d}.jsonl"
    if args.resume:
        if contract_path.exists():
            if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
                raise ValueError("resume contract differs from the saved shard")
        elif output.exists():
            raise FileExistsError("cannot resume result rows without a saved contract")
    elif contract_path.exists() or output.exists():
        raise FileExistsError("hybrid shard exists; pass --resume for an identical run")
    contract_path.write_text(json.dumps(run_contract, indent=2) + "\n", encoding="utf-8")
    done = completed_ids(output, expected_ids)
    initially_done = len(done)
    runtime = HybridRuntime(args, local_rank)
    started = time.perf_counter()
    successful = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in selected:
            if str(row["id"]) in done:
                continue
            result = rollout(runtime, row, args)
            result["policy"] = "crn_jepa_qwen_routed_flow_singlepass_import_v2"
            result["graph_checkpoint_sha256"] = runtime.graph.sha256
            result["qwen_adapter_sha256"] = qwen_adapter_sha
            successful += bool(result["top1_structural_exact"])
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            print(json.dumps({"stage": "hybrid-evaluation", "rank": rank,
                              "done": len(done) + 1, "total": len(selected),
                              "id": result["id"], "top1_exact": result["top1_structural_exact"],
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}),
                  flush=True)
            done.add(str(row["id"]))
    elapsed = time.perf_counter() - started
    if len(done) != len(selected):
        raise RuntimeError("evaluation shard ended without every selected reaction")
    if args.resume:
        with output.open(encoding="utf-8") as handle:
            successful = sum(bool(json.loads(line)["top1_structural_exact"]) for line in handle)
    report = {
        "protocol": "opt_in_crn_jepa_qwen_routed_flow_singlepass_import_v2",
        "scope": "selected_source_rows; aggregate all shards before benchmark reporting",
        "rank": rank, "world": world, "global_selected": len(rows),
        "evaluated": len(selected), "top1_exact": successful,
        "top1_rate": successful / len(selected) if selected else 0.0,
        "elapsed_seconds": elapsed,
        "newly_evaluated": len(selected) - initially_done,
        "seconds_per_new_reaction": elapsed / max(1, len(selected) - initially_done),
        "timing": runtime.timing, "calls": runtime.counts,
        "qwen_model": args.model, "qwen_revision": args.model_revision,
        "qwen_adapter_sha256": qwen_adapter_sha,
        "graph_sha256": runtime.graph.sha256,
        "graph_rl_head": str(args.graph_rl_head) if args.graph_rl_head else None,
        "graph_flow_topk": args.graph_flow_topk,
        "graph_initial_budget": args.graph_initial_budget,
        "tool_route": args.tool_route,
        "note": ("greedy_name generates IMPORT to EOS in one Qwen call and truncates FLOW "
                 "immediately after the tool name; prefix_nll is a diagnostic ablation."),
    }
    (args.output / f"summary.shard-{rank:02d}-of-{world:02d}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
