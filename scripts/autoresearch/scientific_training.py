#!/usr/bin/env python3
"""Prepare matched PR #69 scientific State-SFT jobs only after full freeze."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, resolve, verify_evaluation_source
from scripts.autoresearch.taiji_backend import render_job


def _same_or_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"frozen scientific preparation differs: {path}")
    else:
        path.write_text(content)


def build_configs(config: dict[str, Any], root: Path, repo: Path,
                  output: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    frozen_path = output / "scientific_freeze/manifests/freeze.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("PR69 scientific freeze must precede training preparation")
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("engineering_only") is not False:
        raise ValueError("engineering rows cannot be used for scientific smoke")
    if frozen.get("campaign_id") != config["campaign_id"]:
        raise ValueError("scientific freeze campaign identity differs")
    expected_sources = set(config["evaluation_sources"])
    if set(frozen.get("evaluation_hashes") or {}) != expected_sources:
        raise ValueError("scientific freeze does not cover every R1–R5 evaluation source")
    for name, raw in config["evaluation_sources"].items():
        path = resolve(root, raw)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"scientific evaluation source missing: {name}")
        if verify_evaluation_source(path, name=name) != frozen["evaluation_hashes"][name]:
            raise ValueError(f"scientific evaluation source drifted: {name}")
    if not frozen.get("mech_comparison_identifiable") or int(frozen.get("curated_accepted_rows") or 0) == 0:
        raise ValueError("Mech-vs-Base contrast needs replay-compatible curated rows")
    for source, entry in frozen.get("sources", {}).items():
        source_path = Path(entry["path"])
        if not source_path.is_file() or digest(source_path) != entry["sha256"]:
            raise ValueError(f"frozen training source drifted: {source}")

    expected_rows = int(config["scientific_smoke"]["rows_per_condition"])
    if expected_rows <= 0:
        raise ValueError("scientific smoke row budget must be positive")
    template = yaml.safe_load((repo / config["model"]["training_template"]).read_text())
    model_revision = str(config["model"]["revision"])
    if len(model_revision) != 40 or any(ch not in "0123456789abcdef" for ch in model_revision):
        raise ValueError("scientific model revision must be a pinned commit SHA")
    flower_manifest = resolve(root, config["sources"]["flower"]["manifest"])
    if flower_manifest is None or not flower_manifest.is_file():
        raise FileNotFoundError("FlowER State-SFT source manifest is missing")
    flower = json.loads(flower_manifest.read_text())
    validation = resolve(root, flower["splits"]["valid"]["target"])
    if validation is None or not validation.is_file():
        raise FileNotFoundError("shared scientific validation source is missing")
    validation_sha = digest(validation)
    if validation_sha != flower["splits"]["valid"]["output_sha256"]:
        raise ValueError("shared scientific validation source hash drifted")
    freeze_sha = digest(frozen_path)

    training_configs = {}
    files = {}
    for condition in ("base", "mech"):
        quota = frozen.get("resolved_quotas", {}).get(condition, {})
        if sum(int(value) for value in quota.values()) != expected_rows:
            raise ValueError(f"{condition} resolved source quotas do not total {expected_rows}")
        selected = frozen["files"].get(condition)
        if not selected or int(selected["rows"]) != expected_rows:
            raise ValueError(f"{condition} scientific selection is not {expected_rows} rows")
        train = Path(selected["train"])
        strata = Path(selected["strata"])
        if not train.is_file() or digest(train) != selected["train_sha256"]:
            raise ValueError(f"{condition} selected training file hash drifted")
        if not strata.is_file() or digest(strata) != selected["strata_sha256"]:
            raise ValueError(f"{condition} selected strata file hash drifted")
        if sum(1 for _ in train.open(encoding="utf-8")) != expected_rows:
            raise ValueError(f"{condition} selected training file row count drifted")
        cfg = deepcopy(template)
        cfg.pop("pretokenized_cache_dir", None)
        cfg.pop("pretokenization_world_size", None)
        cfg["condition_name"] = config["campaign_id"] + "_" + condition
        cfg["scientific_hypothesis"] = "curated_replay_compatible_mechanisms_improve_verified_retrosynthesis"
        cfg["model_name_or_path"] = config["model"]["name"]
        cfg["train_file"] = str(train)
        cfg["validation_file"] = str(validation)
        cfg["test_file"] = None
        cfg["output_dir"] = str(output / f"jobs/scientific_{condition}_model")
        cfg["limit_examples"] = 0
        cfg["training"].update({
            "qlora": False, "bf16": True, "fp16": False,
            "require_flash_sdp": False, "use_liger_kernel": False,
            "max_steps": -1, "num_train_epochs": 1.0,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "eval_strategy": "no", "validation_limit": 256,
            "save_steps": 500, "dataloader_num_workers": 2,
            "seed": int(config["seed"]), "data_seed": int(config["seed"]),
            "model_revision": model_revision,
        })
        cfg["contract"].update({
            "paper_baseline_id": f"pr69_{condition}_smoke_state_sft",
            "paper_method_name": f"pr69_{condition}_smoke_state_sft",
            "paper_run_role": "matched_scientific_smoke_state_sft",
            "stable_id_manifest": str(frozen_path),
            "validation_report": str(flower_manifest),
            "expected_train_rows": expected_rows,
            "expected_validation_rows": 256,
            "expected_test_rows": -1,
            "require_strict_trace_universe_complete": False,
            "source_dataset": "pr69_frozen_stratified_decision_sample",
            "source_artifact": str(frozen_path),
            "training_decision_rows": expected_rows,
            "scientific_condition": condition,
            "evaluation_source_hashes": frozen["evaluation_hashes"],
        })
        cfg["contract"].pop("reaction_denominator", None)
        training_configs[condition] = cfg
        files[condition] = {"train": str(train), "train_sha256": selected["train_sha256"],
                            "strata_sha256": selected["strata_sha256"],
                            "output_dir": cfg["output_dir"]}
    if training_configs["base"]["training"] != training_configs["mech"]["training"]:
        raise ValueError("scientific conditions have mismatched optimization budgets")
    return training_configs, {
        "artifact_type": "pr69_scientific_training_preparation_v1",
        "campaign_id": config["campaign_id"],
        "scientific_freeze": str(frozen_path), "scientific_freeze_sha256": freeze_sha,
        "evaluation_hashes": frozen["evaluation_hashes"],
        "curated_accepted_rows": frozen["curated_accepted_rows"],
        "rows_per_condition": expected_rows,
        "model_name_or_path": config["model"]["name"],
        "model_revision": model_revision,
        "shared_validation": str(validation),
        "shared_validation_sha256": validation_sha,
        "training": training_configs["base"]["training"],
        "files": files,
        "expected_optimizer_updates_per_condition": (expected_rows + 7) // 8,
        "test_used": False,
    }


def prepare(config: dict[str, Any], root: Path, repo: Path,
            output: Path, *, model_cache: Path | None = None) -> dict[str, Any]:
    configs, report = build_configs(config, root, repo, output)
    prepared_path = output / "jobs/scientific_prepared.json"
    if prepared_path.exists():
        existing = json.loads(prepared_path.read_text())
        stable = deepcopy(existing)
        audit_fields = ("dry_run_n_rows", "training_config", "training_config_sha256",
                        "token_audit", "token_audit_sha256", "total_input_tokens",
                        "total_supervised_tokens")
        for condition in ("base", "mech"):
            for field in audit_fields:
                stable["files"][condition].pop(field, None)
        if stable != report:
            raise ValueError("existing scientific preparation differs from frozen inputs")
        for condition in ("base", "mech"):
            row = existing["files"][condition]
            training_config = Path(row["training_config"])
            audit_path = Path(row["token_audit"])
            if training_config != output / f"jobs/scientific_{condition}_training.yaml":
                raise ValueError(f"{condition} prepared training config path drifted")
            if not training_config.is_file() or digest(training_config) != row["training_config_sha256"]:
                raise ValueError(f"{condition} prepared training config drifted")
            if yaml.safe_load(training_config.read_text()) != configs[condition]:
                raise ValueError(f"{condition} prepared training config differs from current contract")
            if not audit_path.is_file() or digest(audit_path) != row["token_audit_sha256"]:
                raise ValueError(f"{condition} prepared token audit drifted")
            audit = json.loads(audit_path.read_text())
            if not audit.get("passed") or audit.get("rows") != report["rows_per_condition"] or \
               audit.get("input_sha256") != row["train_sha256"] or \
               audit.get("resolved_model_revision") != report["model_revision"]:
                raise ValueError(f"{condition} prepared token audit is invalid")
        return existing
    audit_environment = None
    if model_cache is not None:
        snapshot = (model_cache / "models--Qwen--Qwen3-0.6B" / "snapshots" /
                    report["model_revision"])
        if not (snapshot / "model.safetensors").is_file():
            raise FileNotFoundError(f"pinned scientific model snapshot is incomplete: {snapshot}")
        audit_environment = {**os.environ, "HF_HUB_CACHE": str(model_cache),
                             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for condition, cfg in configs.items():
        config_path = output / f"jobs/scientific_{condition}_training.yaml"
        _same_or_write(config_path, yaml.safe_dump(cfg, sort_keys=False))
        dry = subprocess.run(
            [sys.executable, str(repo / "scripts/train_tool_sft.py"),
             "--config", str(config_path), "--dry-run"],
            cwd=repo, text=True, capture_output=True, check=False)
        if dry.returncode:
            raise RuntimeError(f"{condition} trainer dry-run failed: {dry.stderr[-1500:]}")
        contract = json.loads(dry.stdout)
        if contract["n_rows"] != report["rows_per_condition"] or \
           contract["train_file_sha256"] != report["files"][condition]["train_sha256"] or \
           contract["validation_file_sha256"] != report["shared_validation_sha256"] or \
           contract["validation"]["n_rows"] != 256 or \
           contract["max_steps"] != -1 or contract["num_train_epochs"] != 1.0 or \
           contract["model_name_or_path"] != report["model_name_or_path"]:
            raise ValueError(f"{condition} trainer validated a different frozen sample")
        report["files"][condition]["dry_run_n_rows"] = contract["n_rows"]
        audit_path = output / f"jobs/scientific_{condition}_token_audit.json"
        audit = subprocess.run(
            [sys.executable, str(repo / "scripts/audit_tool_sft_token_lengths.py"),
             "--config", str(config_path), "--output", str(audit_path)],
            cwd=repo, text=True, capture_output=True, check=False,
            env=audit_environment)
        if audit.returncode:
            raise RuntimeError(f"{condition} token audit failed: {(audit.stderr or audit.stdout)[-1500:]}")
        audit_report = json.loads(audit_path.read_text())
        if not audit_report["passed"] or audit_report["rows"] != report["rows_per_condition"] or \
           audit_report["input_sha256"] != report["files"][condition]["train_sha256"] or \
           audit_report["resolved_model_revision"] != report["model_revision"]:
            raise ValueError(f"{condition} token audit differs from frozen sample")
        report["files"][condition]["training_config"] = str(config_path)
        report["files"][condition]["training_config_sha256"] = digest(config_path)
        report["files"][condition]["token_audit"] = str(audit_path)
        report["files"][condition]["token_audit_sha256"] = digest(audit_path)
        report["files"][condition]["total_input_tokens"] = audit_report["total_input_tokens"]
        report["files"][condition]["total_supervised_tokens"] = audit_report["total_supervised_tokens"]
    _same_or_write(prepared_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def render_scientific_job(
    config: dict[str, Any], root: Path, repo: Path, output: Path,
    *, condition: str, template: Path, job_config: Path, task_flag: str,
    gpu_name: str, model_cache: Path,
) -> dict[str, Any]:
    if condition not in {"base", "mech"}:
        raise ValueError("scientific training condition must be base or mech")
    prepared = prepare(config, root, repo, output, model_cache=model_cache)
    selected = prepared["files"][condition]
    training_config = Path(selected["training_config"])
    if digest(training_config) != selected["training_config_sha256"]:
        raise ValueError("scientific training config drifted after token audit")
    frozen_path = Path(prepared["scientific_freeze"])
    if digest(frozen_path) != prepared["scientific_freeze_sha256"]:
        raise ValueError("scientific freeze drifted after training preparation")
    result = render_job(
        template, job_config, task_flag=task_flag,
        readable_name=f"meteor MechET PR69 {condition} scientific State-SFT (1x{gpu_name})",
        repo=repo, training_config=training_config, gpu_name=gpu_name,
        model_cache=model_cache, model_revision=prepared["model_revision"],
        description=(
            f"PR69 scientific {condition} condition: {prepared['rows_per_condition']} "
            "frozen State-SFT decision rows; Qwen3-0.6B pinned revision; "
            "one epoch, shared frozen validation, R1-R5 remain held out."
        ),
    )
    return {**result, "scientific_condition": condition,
            "job_config_sha256": digest(job_config),
            "scientific_freeze_sha256": prepared["scientific_freeze_sha256"],
            "training_config_sha256": selected["training_config_sha256"],
            "train_sha256": selected["train_sha256"],
            "token_audit_sha256": selected["token_audit_sha256"],
            "model_revision": prepared["model_revision"],
            "rows": prepared["rows_per_condition"],
            "expected_optimizer_updates": prepared["expected_optimizer_updates_per_condition"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(prepare(config, args.data_root, repo, args.output,
                             model_cache=args.model_cache),
                     indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
