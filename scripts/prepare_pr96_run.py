#!/usr/bin/env python3
"""Create independent, initially disabled training views from a reference approval."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.pr96_eval_runtime import sha
from mechet.qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION, LORA_TARGETS


def prepare(approval_path, base_path, run_dir):
    import yaml
    approval_path, base_path, run_dir = map(Path, (approval_path, base_path, run_dir))
    approval, base = json.loads(approval_path.read_text()), json.loads(base_path.read_text())
    if not approval.get("reference_training_authorized"):
        raise ValueError("PR96_REFERENCE_NOT_AUTHORIZED")
    if base.get("model_id") != OFFICIAL_MODEL or base.get("revision") != OFFICIAL_REVISION:
        raise ValueError("PR96_WRONG_OFFICIAL_BASE")
    ref_path, hist_path = Path(approval["reference_manifest"]), Path(approval["history_manifest"])
    if sha(ref_path) != approval["reference_manifest_sha256"] or sha(hist_path) != approval["history_manifest_sha256"]:
        raise ValueError("PR96_REFERENCE_APPROVAL_PARENT_CHANGED")
    data, history = json.loads(ref_path.read_text()), json.loads(hist_path.read_text())
    cohorts = ref_path.parent / "frozen_valid_cohorts.json"
    if sha(cohorts) != approval["frozen_cohorts_sha256"]:
        raise ValueError("PR96_FROZEN_COHORTS_CHANGED")
    track = approval["track"]
    run_dir.mkdir(parents=True, exist_ok=False)
    configs = {}
    for stage in ("state", "history"):
        view = run_dir / "views" / stage
        view.mkdir(parents=True)
        source_train = (Path(data["reports"]["train"]["outputs"]["state_sft"]["path"])
                        if stage == "state" else Path(history["reports"]["train"]["path"]))
        expected_hash = (data["reports"]["train"]["outputs"]["state_sft"]["sha256"]
                         if stage == "state" else history["reports"]["train"]["sha256"])
        validation = approval["validation_views"][stage]
        if sha(source_train) != expected_hash or sha(validation["path"]) != validation["sha256"]:
            raise ValueError("PR96_TRAINING_SOURCE_CHANGED")
        (view / "train.jsonl").symlink_to(source_train)
        (view / "valid.jsonl").symlink_to(Path(validation["path"]))
        (view / "ARTIFACT_STATUS.json").write_text(json.dumps({
            "training_allowed": False, "status": "pending_deployed_preflight_and_complete_token_audit",
            "reference_approval_sha256": sha(approval_path), "stage": stage}) + "\n")
        counts = (data["reports"]["train"]["counts"] if stage == "state" else history["reports"]["train"]["counts"])
        ntrain = (sum(counts[k + "_decisions"] for k in ("import", "event", "finish")) if stage == "state"
                  else sum(counts[k] for k in ("import", "event", "finish")))
        epochs = {("mech31k", "state"): 10, ("mech31k", "history"): 5,
                  ("flower", "state"): 5, ("flower", "history"): 3}[(track, stage)]
        cfg = {
            "condition_name": f"pr96_{track}_mainonly_qwen35_08b_{stage}",
            "model_name_or_path": OFFICIAL_MODEL, "train_file": str(view / "train.jsonl"),
            "validation_file": str(view / "valid.jsonl"),
            "pretokenized_cache_dir": str(run_dir / "tokens" / stage), "pretokenization_world_size": 8,
            "output_dir": str(run_dir / "models" / stage),
            "pr96": {"track": track, "stage": stage, "reference_dir": str(ref_path.parent),
                     "reference_approval": str(approval_path), "reference_approval_sha256": sha(approval_path),
                     "base_manifest": str(base_path), "base_manifest_sha256": sha(base_path),
                     "cohorts_file": str(cohorts), "cohorts_sha256": sha(cohorts),
                     "max_new_tokens": 512, "evaluation_batch_size": 8,
                     "original_denominators": {s:r["original_count"] for s,r in approval["reports"].items()},
                     "accepted_reactions": {s:r["reference_pass"] for s,r in approval["reports"].items()}},
            "contract": {"require_trace_owned": False, "require_tool_decision": True,
                         "expected_train_rows": ntrain, "expected_validation_rows": validation["decision_rows"],
                         "environment_revision": "natural_language_electron_event_v2" if stage == "state" else "natural_language_electron_event_history_v2",
                         "executor_revision": "pr95_main_product_only_actual_executor"},
            "lora": {"r": 16, "alpha": 32, "dropout": .05, "target_modules": LORA_TARGETS},
            "training": {"model_revision": OFFICIAL_REVISION, "trust_remote_code": False,
                         "num_train_epochs": epochs, "max_steps": -1, "max_length": 32768,
                         "window_context_tokens": 0, "encoding_protocol": "qwen35_tool_span_v1",
                         "per_device_train_batch_size": 4, "gradient_accumulation_steps": 2,
                         "per_device_eval_batch_size": 4, "learning_rate": 1e-4 if stage == "state" else 5e-5,
                         "bf16": True, "fp16": False, "qlora": False, "tf32": True,
                         "attention_implementation": "sdpa", "gradient_checkpointing": True,
                         "assistant_only_loss": True, "packing": False, "group_by_length": True,
                         "logging_steps": 10, "save_total_limit": 3, "eval_strategy": "epoch",
                         "dataloader_num_workers": 4, "optim": "adamw_torch_fused", "seed": 17,
                         "data_seed": 17, "lr_scheduler_type": "cosine", "warmup_ratio": .03,
                         "report_to": []},
        }
        target = run_dir / f"{stage}.yaml"
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[stage] = str(target)
    plan = {"track": track, "run_dir": str(run_dir), "configs": configs,
            "reference_approval": str(approval_path), "reference_approval_sha256": sha(approval_path),
            "base_manifest": str(base_path), "base_manifest_sha256": sha(base_path),
            "formal_training_submitted": False, "test_inference_executed": False}
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.approval, args.base_manifest, args.run_dir)), flush=True)


if __name__ == "__main__":
    main()
