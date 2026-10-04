#!/usr/bin/env python3
"""Evaluate frozen System-One decisions by executing predicted electron flows.

This is a local gold-state evaluation, not product-start retrosynthesis. Fixed-1,
fixed-2 and executor-validity backoff policies are gold-independent; oracle-count
is reported only as a diagnostic upper bound and cannot qualify a model for
promotion. The backoff policy was chosen after inspecting validation results,
so it is exploratory on validation and must be frozen before held-out testing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.electron_pointer_model import pairs_for_observation
from mechet.natural_language_electron_flow import execute_event_arguments
from mechet.structural_overlap import canonical_unmapped_smiles
from mechet.system_one_decision import ElectronFlowDecisionHead
from mechet.system_one_replay import execute_pair_indices, reconstruct_mapped_state
from scripts.train_system_one_electron_flow import (
    file_sha256, load, prepare, recall_at_k, verify_source,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--valid", "--data", dest="valid", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def select_flow_counts(flow_count: int, executed: dict[int, dict]) -> dict[str, int]:
    """Select execution lengths without reading the reference, except oracle."""
    return {
        "fixed1": 1,
        "fixed2": 2,
        "validity_backoff_2_to_1": 2 if executed[2].get("ok") else 1,
        "oracle_count": flow_count,
    }


def main() -> int:
    args = arguments()
    if args.limit < 0 or args.log_every < 1:
        raise ValueError("limit cannot be negative and log-every must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(
        (args.checkpoint / f"run_manifest_epoch{args.epoch}.json").read_text()
    )
    if manifest.get("artifact_type") != "system_one_phase0_decision_policy":
        raise ValueError("not a System-One decision checkpoint")
    source = verify_source(args.valid)
    if args.valid.stem != args.split:
        raise ValueError("evaluation path and declared split disagree")
    if args.split == "valid":
        if source["sha256"] != manifest["valid_source"]["sha256"]:
            raise ValueError("checkpoint validation source differs from evaluation source")
    elif (
        source["declared_sha256"] != source["sha256"]
        or source["artifact_id"] != manifest["valid_source"]["artifact_id"]
        or source["event_decisions"] is None
        or args.valid.parent.resolve() != Path(manifest["valid_source"]["path"]).parent.resolve()
    ):
        raise ValueError("test source is not the frozen checkpoint-lineage trace view")
    adapter = args.checkpoint / f"adapter_epoch{args.epoch}"
    head_path = args.checkpoint / f"decision_head_epoch{args.epoch}.pt"
    if file_sha256(adapter / "adapter_model.safetensors") != manifest["adapter_model_sha256"]:
        raise ValueError("adapter weights differ from checkpoint manifest")
    if file_sha256(head_path) != manifest["decision_head_sha256"]:
        raise ValueError("decision head differs from checkpoint manifest")

    examples, counts = load(args.valid)
    if source["decision_rows"] is not None and counts["input_rows"] != source["decision_rows"]:
        raise ValueError(f"{args.split} decision-row denominator mismatch")
    if source["event_decisions"] is not None and counts["flow_events"] != source["event_decisions"]:
        raise ValueError(f"{args.split} event denominator mismatch")
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("empty evaluation selection")

    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    model_name = manifest["model"]
    revision = manifest["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    if not tokenizer.is_fast:
        raise ValueError("fast tokenizer required for option alignment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda" else torch.float32
    )
    base = AutoModel.from_pretrained(
        model_name, revision=revision, trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    policy = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    weights = torch.load(head_path, map_location="cpu", weights_only=True)
    pointer_dim = weights["source.query.weight"].shape[0]
    head = ElectronFlowDecisionHead(int(policy.config.hidden_size), pointer_dim)
    head.load_state_dict(weights, strict=True)
    head = head.to(device).eval()

    modes = ("fixed1", "fixed2", "validity_backoff_2_to_1", "oracle_count")
    metrics: dict[str, Counter] = {mode: Counter() for mode in modes}
    strata: dict[str, dict[str, Counter]] = {mode: {} for mode in modes}
    paired = Counter()
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, example in enumerate(examples, 1):
            item = prepare(example, tokenizer)
            mapped = reconstruct_mapped_state(example)
            gold_args = example.assistant_message["tool_calls"][0]["function"]["arguments"]
            gold = execute_event_arguments(mapped, gold_args)
            if not gold.get("ok"):
                raise ValueError(f"{example.row_id}: gold event does not execute: {gold.get('code')}")
            paired_gold = execute_pair_indices(mapped, example, item.pair_targets)
            if paired_gold != gold:
                raise ValueError(f"{example.row_id}: pair labels do not replay the gold successor")
            gold_successor = canonical_unmapped_smiles(gold["state_smiles"])

            input_ids = torch.tensor([item.input_ids], dtype=torch.long, device=device)
            states = policy(input_ids=input_ids, use_cache=False).last_hidden_state[0]
            source_pairs, sink_pairs = pairs_for_observation(example, device)
            decision = head(
                states[-1], states[item.marker_indices], source_pairs, sink_pairs
            )
            ranked = decision.pair_logits.flatten().topk(
                min(8, item.pair_count)
            ).indices.tolist()
            paired.update(recall_at_k(ranked, item.pair_targets))
            flow_count = len(item.pair_targets)
            row = {
                "id": example.row_id,
                "gold_flow_count": flow_count,
                "pair_targets": item.pair_targets,
                "ranked_top8": ranked,
                "gold_successor": gold_successor,
                "policies": {},
            }
            executed = {
                count: execute_pair_indices(mapped, example, ranked[:count])
                for count in {1, 2, flow_count}
            }
            selected_counts = select_flow_counts(flow_count, executed)
            for mode in modes:
                chosen = selected_counts[mode]
                result = executed[chosen]
                executable = bool(result.get("ok"))
                successor = (
                    canonical_unmapped_smiles(result["state_smiles"])
                    if executable else None
                )
                match = executable and successor == gold_successor
                for bucket in (metrics[mode], strata[mode].setdefault(str(flow_count), Counter())):
                    bucket["n"] += 1
                    bucket["execute_ok"] += int(executable)
                    bucket["successor_exact"] += int(match)
                    bucket[f"code_{result.get('code', 'UNKNOWN')}"] += 1
                row["policies"][mode] = {
                    "selected_flow_count": chosen,
                    "pair_indices": ranked[:chosen],
                    "execute_ok": executable,
                    "code": result.get("code"),
                    "successor": successor,
                    "successor_exact": bool(match),
                }
            rows.append(row)
            if index % args.log_every == 0 or index == len(examples):
                print(json.dumps({
                    "phase": "successor_eval", "events": index,
                    "total": len(examples), "elapsed_s": round(time.perf_counter() - started, 1),
                }), flush=True)

    def summarize(bucket: Counter) -> dict:
        n = bucket["n"]
        return {
            "n": n, "execute_ok": bucket["execute_ok"],
            "execute_rate": bucket["execute_ok"] / n,
            "successor_exact": bucket["successor_exact"],
            "successor_exact_rate": bucket["successor_exact"] / n,
            "codes": {key[5:]: count for key, count in bucket.items() if key.startswith("code_")},
        }

    report = {
        "artifact_type": "system_one_phase0_local_successor_evaluation",
        "scope": "reference_current_state_not_product_start",
        "split": args.split,
        "valid_source": source,
        "checkpoint_manifest": str((args.checkpoint / f"run_manifest_epoch{args.epoch}.json").resolve()),
        "checkpoint_adapter_sha256": manifest["adapter_model_sha256"],
        "checkpoint_head_sha256": manifest["decision_head_sha256"],
        "evaluated_events": len(examples),
        "gold_replay_ok": len(examples),
        "pair_recall": {key: value / len(examples) for key, value in paired.items()},
        "policies": {
            mode: {"overall": summarize(metrics[mode]),
                   "by_gold_flow_count": {key: summarize(value) for key, value in strata[mode].items()},
                   "gold_independent": mode != "oracle_count",
                   "validation_post_hoc": mode == "validity_backoff_2_to_1"}
            for mode in modes
        },
        "elapsed_s": time.perf_counter() - started,
        "promotion_ready": False,
        "promotion_note": "Local successor agreement is not product-start endpoint performance; oracle-count is diagnostic only.",
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "cases.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"phase": "complete", "output": str(args.output), "summary": report["policies"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
