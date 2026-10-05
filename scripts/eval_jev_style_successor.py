#!/usr/bin/env python3
"""Executor-grounded evaluation for the typed Jev-style electron-flow policy.

This is a local reference-current-state evaluation, not product-start
retrosynthesis. Gold flow count is used only by the explicitly named oracle
diagnostic.
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

from mechet.jev_style_decision import FactorizedTypedElectronFlowHead, block_causal_option_mask
from mechet.natural_language_electron_flow import execute_event_arguments
from mechet.structural_overlap import canonical_unmapped_smiles
from mechet.system_one_replay import execute_pair_indices, reconstruct_mapped_state
from scripts.train_jev_style_electron_flow import prepare_typed
from scripts.train_system_one_electron_flow import (
    file_sha256,
    load,
    recall_at_k,
    verify_source,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def select_counts(gold_count: int, executed: dict[int, dict]) -> dict[str, int]:
    return {
        "fixed1": 1,
        "fixed2": 2,
        "validity_backoff_2_to_1": 2 if executed[2].get("ok") else 1,
        "oracle_count": gold_count,
    }


def main() -> int:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.checkpoint / f"run_manifest_epoch{args.epoch}.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_type") != "system_one_jev_typed_v2_factorized_decision_policy":
        raise ValueError("checkpoint is not factorized typed Jev-style v2")

    source = verify_source(args.data)
    if args.data.stem != args.split:
        raise ValueError("evaluation path and declared split disagree")
    if args.split == "valid":
        if source["sha256"] != manifest["valid_source"]["sha256"]:
            raise ValueError("validation source differs from checkpoint lineage")
    else:
        if (
            source["declared_sha256"] != source["sha256"]
            or source["artifact_id"] != manifest["valid_source"]["artifact_id"]
            or args.data.parent.resolve() != Path(manifest["valid_source"]["path"]).parent.resolve()
        ):
            raise ValueError("test source is not the frozen checkpoint-lineage trace view")

    adapter = args.checkpoint / f"adapter_epoch{args.epoch}"
    head_path = args.checkpoint / f"typed_head_epoch{args.epoch}.pt"
    if file_sha256(adapter / "adapter_model.safetensors") != manifest["adapter_model_sha256"]:
        raise ValueError("adapter weights differ from manifest")
    if file_sha256(head_path) != manifest["decision_head_sha256"]:
        raise ValueError("typed head differs from manifest")

    examples, counts = load(args.data)
    if source["event_decisions"] is not None and counts["flow_events"] != source["event_decisions"]:
        raise ValueError("event denominator mismatch")
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("empty evaluation selection")

    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"], revision=manifest["model_revision"], trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda" else torch.float32
    )
    base = AutoModel.from_pretrained(
        manifest["model"],
        revision=manifest["model_revision"],
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    policy = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    head = FactorizedTypedElectronFlowHead(
        int(policy.config.hidden_size), int(manifest["pointer_dim"])
    ).to(device)
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
    head.eval()

    modes = ("fixed1", "fixed2", "validity_backoff_2_to_1", "oracle_count")
    metrics = {mode: Counter() for mode in modes}
    paired = Counter()
    rows = []
    started = time.perf_counter()

    with torch.inference_mode():
        for index, example in enumerate(examples, 1):
            item = prepare_typed(example, tokenizer)
            enc = item.encoding
            ids = torch.tensor([enc.input_ids], dtype=torch.long, device=device)
            positions = torch.tensor([enc.position_ids], dtype=torch.long, device=device)
            mask = block_causal_option_mask(enc, device=device, dtype=dtype)
            states = policy(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
            ).last_hidden_state[0]
            decision = head(
                states[enc.decide_indices[0]],
                states[list(enc.option_indices[0])],
                states[enc.decide_indices[1]],
                states[list(enc.option_indices[1])],
            )
            ranked = decision.pair_logits.flatten().topk(
                min(8, item.pair_count)
            ).indices.tolist()
            paired.update(recall_at_k(ranked, list(item.pair_targets)))

            mapped = reconstruct_mapped_state(example)
            gold_args = example.assistant_message["tool_calls"][0]["function"]["arguments"]
            gold = execute_event_arguments(mapped, gold_args)
            if not gold.get("ok"):
                raise ValueError(f"{example.row_id}: gold action does not execute")
            gold_pair = execute_pair_indices(mapped, example, list(item.pair_targets))
            if gold_pair != gold:
                raise ValueError(f"{example.row_id}: typed pair labels do not replay gold")
            gold_successor = canonical_unmapped_smiles(gold["state_smiles"])

            gold_count = len(item.pair_targets)
            executed = {
                count: execute_pair_indices(mapped, example, ranked[:count])
                for count in {1, 2, gold_count}
            }
            selected = select_counts(gold_count, executed)
            row = {
                "id": example.row_id,
                "gold_flow_count": gold_count,
                "pair_targets": list(item.pair_targets),
                "ranked_top8": ranked,
                "gold_successor": gold_successor,
                "policies": {},
            }
            for mode in modes:
                count = selected[mode]
                result = executed[count]
                ok = bool(result.get("ok"))
                successor = canonical_unmapped_smiles(result["state_smiles"]) if ok else None
                exact = bool(ok and successor == gold_successor)
                bucket = metrics[mode]
                bucket["n"] += 1
                bucket["execute_ok"] += int(ok)
                bucket["successor_exact"] += int(exact)
                bucket[f"code_{result.get('code', 'UNKNOWN')}"] += 1
                row["policies"][mode] = {
                    "selected_flow_count": count,
                    "pair_indices": ranked[:count],
                    "execute_ok": ok,
                    "successor_exact": exact,
                    "successor": successor,
                    "code": result.get("code"),
                }
            rows.append(row)
            if index % args.log_every == 0 or index == len(examples):
                print(json.dumps({
                    "phase": "jev_typed_successor_eval",
                    "events": index,
                    "total": len(examples),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }), flush=True)

    def summarize(bucket: Counter):
        n = bucket["n"]
        return {
            "n": n,
            "execute_ok": bucket["execute_ok"],
            "execute_rate": bucket["execute_ok"] / n,
            "successor_exact": bucket["successor_exact"],
            "successor_exact_rate": bucket["successor_exact"] / n,
            "codes": {
                key[5:]: value for key, value in bucket.items()
                if key.startswith("code_")
            },
        }

    report = {
        "artifact_type": "system_one_jev_typed_v2_local_successor_evaluation",
        "architecture": manifest["architecture"],
        "scope": "reference_current_state_not_product_start",
        "split": args.split,
        "evaluated_events": len(examples),
        "pair_recall": {key: value / len(examples) for key, value in paired.items()},
        "policies": {
            mode: {
                "overall": summarize(metrics[mode]),
                "gold_independent": mode != "oracle_count",
                "validation_post_hoc": mode == "validity_backoff_2_to_1",
            }
            for mode in modes
        },
        "elapsed_s": time.perf_counter() - started,
        "checkpoint_manifest": str(manifest_path.resolve()),
        "adapter_sha256": manifest["adapter_model_sha256"],
        "head_sha256": manifest["decision_head_sha256"],
        "promotion_note": "Local successor agreement only; product-start endpoint performance remains unmeasured.",
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "cases.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"phase": "complete", "report": report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
