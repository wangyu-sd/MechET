#!/usr/bin/env python3
"""Freeze equal-update Qwen3-8B SFT configs for the three H2 conditions."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any

import yaml


BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CONDITIONS = ("direct", "open_flow", "closed_loop")


def create_configs(
    matched_dir: Path, output_dir: Path, base_config: Path,
    *, run_output_root: Path, epochs_reference: int = 3, gpu_count: int = 8,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing frozen configs: {output_dir}")
    if epochs_reference < 1 or gpu_count < 1:
        raise ValueError("epochs_reference and gpu_count must be positive")
    base = yaml.safe_load(base_config.read_text())
    if base.get("model_name_or_path") != "Qwen/Qwen3-8B":
        raise ValueError("base config is not Qwen3-8B")
    training = dict(base["training"])
    if str(training.get("model_revision")) != BASE_REVISION:
        raise ValueError("base model revision drifted")
    lora = dict(base["lora"])
    manifests = {
        condition: json.loads((matched_dir / condition / "manifest.json").read_text())
        for condition in CONDITIONS
    }
    split_hashes = {value["parent_split_manifest_sha256"] for value in manifests.values()}
    id_hashes = {json.dumps(value["stable_reaction_ids"], sort_keys=True) for value in manifests.values()}
    row_counts = {json.dumps(value["rows"], sort_keys=True) for value in manifests.values()}
    if len(split_hashes) != 1 or len(id_hashes) != 1 or len(row_counts) != 1:
        raise ValueError("matched conditions do not share frozen IDs and row counts")
    rows = manifests["direct"]["rows"]
    if any(manifests[condition].get("condition") != condition or
           not manifests[condition].get("training_allowed")
           for condition in CONDITIONS):
        raise ValueError("condition manifest mismatch or training not allowed")
    if any(not (matched_dir / condition / f"{split}.jsonl").is_file()
           for condition in CONDITIONS for split in ("train", "valid", "test")):
        raise ValueError("matched representation file missing")
    verification_path = matched_dir / "verification.json"
    verification = json.loads(verification_path.read_text())
    if (verification.get("passed") is not True or
        verification.get("parent_split_manifest_sha256") != next(iter(split_hashes)) or
        verification.get("stable_reaction_ids") != manifests["direct"]["stable_reaction_ids"]):
        raise ValueError("matched product/endpoint verification is missing or mismatched")

    global_batch = gpu_count * int(training["per_device_train_batch_size"]) * int(training["gradient_accumulation_steps"])
    updates = ceil(int(rows["train"]) / global_batch) * epochs_reference
    training.update({
        "attention_implementation": "sdpa",
        "require_flash_sdp": True,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "max_length": 16384,
        "window_context_tokens": 8192,
        "num_train_epochs": float(epochs_reference),
        "max_steps": updates,
        "seed": 17,
        "data_seed": 17,
        "report_to": [],
    })
    output_dir.mkdir(parents=True)
    config_hashes = {}
    for condition in CONDITIONS:
        path = matched_dir / condition
        contract = {
            "paper_method_name": condition,
            "paper_run_role": "issue79_h2_matched_retrain",
            "stable_id_manifest": str((path / "manifest.json").resolve()),
            "expected_train_rows": int(rows["train"]),
            "expected_validation_rows": int(rows["valid"]),
            "expected_test_rows": int(rows["test"]),
            "expected_upstream_endpoint_fallback_rows": 0,
            "require_trace_owned": condition == "closed_loop",
            "corpus_used": False,
            "source_dataset": "strict_executable_flower_training_pool_H2_composition_disjoint",
            "h2_parent_split_manifest_sha256": next(iter(split_hashes)),
            "test_use_policy": "no checkpoint selection on H2 test",
        }
        if condition == "closed_loop":
            contract.update({
                "observation_contract": "compact_full_state_v1",
                "endpoint_source": "environment_owned_trace",
                "terminal_tool": "finish_trace",
                "free_form_proof_submission": False,
                "environment_revision": "trace_owned_compact_full_state_v1",
                "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
            })
        cfg = {
            "condition_name": f"nmi_h2_{condition}_qwen3_8b_seed17",
            "scientific_hypothesis": "issue79_local_primitive_recombination_closed_vs_open_execution",
            "model_name_or_path": "Qwen/Qwen3-8B",
            "train_file": str((path / "train.jsonl").resolve()),
            "validation_file": str((path / "valid.jsonl").resolve()),
            "test_file": str((path / "test.jsonl").resolve()),
            "pretokenized_cache_dir": str((path / "qwen3_8b_tokens_16384").resolve()),
            "pretokenization_world_size": gpu_count,
            "output_dir": str((run_output_root / f"h2_{condition}_seed17").resolve()),
            "limit_examples": 0,
            "training": dict(training),
            "lora": dict(lora),
            "contract": contract,
        }
        if condition == "closed_loop":
            cfg["environment"] = {"max_tool_calls": 40, "observation_mode": "compact_full_state"}
        target = output_dir / f"{condition}.yaml"
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        config_hashes[condition] = hashlib.sha256(target.read_bytes()).hexdigest()
    report = {
        "artifact_type": "nmi_h2_matched_qwen3_8b_sft_configs_v1",
        "conditions": list(CONDITIONS),
        "model_revision": BASE_REVISION,
        "split_manifest_sha256": next(iter(split_hashes)),
        "representation_verification_sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest(),
        "stable_reaction_ids": manifests["direct"]["stable_reaction_ids"],
        "rows": rows,
        "gpu_count": gpu_count,
        "global_batch": global_batch,
        "epochs_reference": epochs_reference,
        "equal_optimizer_updates": updates,
        "equal_token_budget": False,
        "token_budget_caveat": "Representation lengths differ; report actual input/supervised tokens from tokenizer manifests.",
        "config_sha256": config_hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/iclr/a4_open_flow_sft.yaml"))
    parser.add_argument("--run-output-root", type=Path, required=True)
    parser.add_argument("--epochs-reference", type=int, default=3)
    parser.add_argument("--gpu-count", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(create_configs(
        args.matched_dir, args.output_dir, args.base_config,
        run_output_root=args.run_output_root,
        epochs_reference=args.epochs_reference, gpu_count=args.gpu_count,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
