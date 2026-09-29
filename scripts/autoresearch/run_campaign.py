#!/usr/bin/env python3
"""Auditable finite-state entry point for the PR #69 smoke campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import (
    digest, freeze, resolve, verify_evaluation_source,
)
from scripts.autoresearch.scientific_training import (
    prepare as prepare_scientific, render_scientific_job,
)
from scripts.autoresearch.taiji_backend import poll, render_heldout_job, render_job, submit
from scripts.autoresearch.ledger import append as append_ledger


STAGES = (
    "FREEZE_MANIFESTS", "ENGINEERING_SMOKE", "MECHANISM_COMPATIBILITY_AUDIT",
    "TRAIN_BASE_SMOKE", "TRAIN_MECH_SMOKE", "RUN_R1", "RUN_R2", "RUN_R3",
    "RUN_R4", "RUN_R5", "COLLECT_SCORECARD", "RECOMMEND_SCALE",
)
EVAL_STAGE_INPUTS = {
    "RUN_R1": ("r1_multi_reference",),
    "RUN_R2": ("r2_plausibility",),
    "RUN_R3": ("r3_corruptions",),
    "RUN_R4": ("r4_pmechdb_challenging", "r4_pmechrp_pathways", "r4_literature_cycles"),
    "RUN_R5": ("r5_external_predictions",),
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def save_new(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def plan(config: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    source_status = {}
    for name, spec in config["sources"].items():
        path = resolve(root, spec.get("train"))
        source_status[name] = {"path": str(path) if path else None,
                               "available": bool(path and path.is_file())}
    eval_status = {}
    for name, raw in config["evaluation_sources"].items():
        path = resolve(root, raw)
        status = {"path": str(path) if path else None,
                  "available": bool(path and path.is_file())}
        if status["available"]:
            try:
                status["sha256"] = verify_evaluation_source(path)
            except ValueError as exc:
                status["available"] = False
                status["invalid_reason"] = str(exc)
        eval_status[name] = status
    engineering = output / "engineering_freeze/manifests/freeze.json"
    scientific = output / "scientific_freeze/manifests/freeze.json"
    scientific_prepared = output / "jobs/scientific_prepared.json"
    scientific_adapters = {
        condition: output / f"jobs/scientific_{condition}_model/adapter_model.safetensors"
        for condition in ("base", "mech")
    }
    package_results = {f"r{index}": output / f"r{index}/result.json"
                       for index in range(1, 6)}
    stages = []
    for stage in STAGES:
        prerequisites: list[str] = []
        if stage == "FREEZE_MANIFESTS":
            prerequisites = [name for name, entry in eval_status.items() if not entry["available"]]
        elif stage == "ENGINEERING_SMOKE":
            if not engineering.is_file():
                prerequisites.append("engineering_freeze")
        elif stage == "MECHANISM_COMPATIBILITY_AUDIT":
            if not source_status["curated"]["available"]:
                prerequisites.append("curated_replay_compatible_rows")
        elif stage in {"TRAIN_BASE_SMOKE", "TRAIN_MECH_SMOKE"}:
            if not scientific.is_file():
                prerequisites.append("scientific_freeze")
            if not scientific_prepared.is_file():
                prerequisites.append("scientific_prepared_token_audited_configs")
            if stage == "TRAIN_MECH_SMOKE" and not source_status["curated"]["available"]:
                prerequisites.append("curated_augmentation_unavailable")
        elif stage in EVAL_STAGE_INPUTS:
            prerequisites = [name for name in EVAL_STAGE_INPUTS[stage]
                             if not eval_status[name]["available"]]
            if not scientific.is_file():
                prerequisites.append("scientific_freeze")
            for condition, adapter in scientific_adapters.items():
                if not adapter.is_file():
                    prerequisites.append(f"scientific_{condition}_trained_adapter")
        elif stage == "COLLECT_SCORECARD":
            prerequisites = [f"{name}_result" for name, path in package_results.items()
                             if not path.is_file()]
            if not (output / "jobs/scientific_training_metrics.json").is_file():
                prerequisites.append("scientific_training_metrics")
            if not (output / "manifests/train_eval_overlap_audit.json").is_file():
                prerequisites.append("train_eval_overlap_audit")
        elif stage == "RECOMMEND_SCALE":
            if not (output / "scorecard.json").is_file():
                prerequisites.append("complete_scorecard")
        stages.append({"stage": stage, "prerequisites_missing": prerequisites,
                       "ready": not prerequisites})
    return {"campaign_id": config["campaign_id"], "stages": stages,
            "source_status": source_status, "evaluation_status": eval_status,
            "engineering_freeze": engineering.is_file(),
            "scientific_freeze": scientific.is_file()}


def prepare_engineering(config: dict[str, Any], root: Path, repo: Path, output: Path) -> dict[str, Any]:
    frozen = output / "engineering_freeze/manifests/freeze.json"
    if not frozen.is_file():
        raise FileNotFoundError("freeze the engineering manifest first")
    manifest = json.loads(frozen.read_text())
    if not manifest.get("engineering_only") or manifest["files"]["engineering"]["rows"] != 32:
        raise ValueError("engineering manifest is not the approved 32-row sample")
    template = repo / config["model"]["training_template"]
    training = yaml.safe_load(template.read_text())
    training.pop("pretokenized_cache_dir", None)
    training.pop("pretokenization_world_size", None)
    training["condition_name"] = config["campaign_id"] + "_engineering"
    training["model_name_or_path"] = config["model"]["name"]
    training["train_file"] = manifest["files"]["engineering"]["train"]
    training["validation_file"] = str(resolve(root, config["sources"]["flower"].get("valid")
        or "data/flower_natural_language_event_sft_v2/valid.jsonl"))
    training["test_file"] = None
    training["output_dir"] = str(output / "jobs/engineering_model")
    training["limit_examples"] = 0
    settings = training["training"]
    settings.update({"qlora": False, "bf16": True, "fp16": False,
                     "require_flash_sdp": False, "use_liger_kernel": False,
                     "max_steps": int(config["engineering_smoke"]["max_steps"]),
                     "num_train_epochs": 1.0, "per_device_train_batch_size": 1,
                     "per_device_eval_batch_size": 1,
                     "gradient_accumulation_steps": 8,
                     "validation_limit": 16, "save_steps": 100,
                     "dataloader_num_workers": 2,
                     "model_revision": config["model"]["revision"]})
    contract = training["contract"]
    contract["stable_id_manifest"] = str(frozen)
    contract["validation_report"] = str(frozen)
    contract["expected_train_rows"] = 32
    contract["expected_validation_rows"] = -1
    contract["expected_test_rows"] = -1
    contract["source_dataset"] = "autoresearch_frozen_engineering_smoke"
    config_path = output / "jobs/engineering_training.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        old = yaml.safe_load(config_path.read_text())
        if old != training:
            raise ValueError("existing engineering training config differs; no silent rewrite")
    else:
        config_path.write_text(yaml.safe_dump(training, sort_keys=False))
    command = [sys.executable, str(repo / "scripts/train_tool_sft.py"),
               "--config", str(config_path), "--dry-run"]
    dry = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
    if dry.returncode:
        raise RuntimeError("engineering trainer contract dry-run failed: " + dry.stderr[-1500:])
    report = json.loads(dry.stdout)
    if int(report["n_rows"]) != 32:
        raise ValueError("trainer did not validate 32 frozen rows")
    ready = {"campaign_id": config["campaign_id"], "prepared_at": timestamp(),
             "git_commit": git_commit(repo), "training_config": str(config_path),
             "training_config_sha256": digest(config_path),
             "manifest_sha256": digest(frozen), "trainer_dry_run": report,
             "command": ["python", "scripts/train_tool_sft.py", "--config", str(config_path)]}
    prepared = output / "jobs/engineering_prepared.json"
    if prepared.exists():
        previous = json.loads(prepared.read_text())
        for key in ("training_config_sha256", "manifest_sha256", "command"):
            if ready[key] != previous[key]:
                raise ValueError("prepared engineering job drifted")
        return previous
    save_new(prepared, ready)
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "reconcile", "freeze-engineering", "freeze-scientific",
                                          "prepare-engineering", "prepare-scientific",
                                          "render-engineering", "render-scientific", "submit-engineering",
                                          "poll-engineering", "render-heldout", "submit-heldout",
                                          "poll-heldout"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--gpu", choices=("H20", "A100", "V100"))
    parser.add_argument("--condition", choices=("base", "mech"))
    parser.add_argument("--task-flag")
    parser.add_argument("--donor-task")
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--job-config", type=Path)
    parser.add_argument("--retry-reason", choices=("model_cache_unavailable", "ceph_bootstrap",
                                                   "transient_runtime", "oom_equivalent",
                                                   "runtime_dependency_missing"))
    parser.add_argument("--client", type=Path, default=Path("/usr/local/bin/taiji_client"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(args.config.read_text())
    if args.action == "plan":
        result = plan(config, args.data_root, args.output)
    elif args.action == "reconcile":
        existing = []
        for kind in ("engineering", "scientific"):
            path = args.output / f"{kind}_freeze/manifests/freeze.json"
            if path.is_file():
                existing.append({"kind": kind, "path": str(path), "sha256": digest(path)})
        if not existing:
            raise FileNotFoundError("no frozen campaign artifact to reconcile")
        result = {"frozen": existing}
    elif args.action.startswith("freeze-"):
        engineering = args.action == "freeze-engineering"
        result = freeze(config, args.data_root,
                        args.output / ("engineering_freeze" if engineering else "scientific_freeze"),
                        engineering=engineering)
    elif args.action == "prepare-engineering":
        result = prepare_engineering(config, args.data_root, repo, args.output)
    elif args.action == "prepare-scientific":
        result = prepare_scientific(config, args.data_root, repo, args.output,
                                    model_cache=args.model_cache)
    elif args.action == "render-engineering":
        if not args.template or not args.gpu or not args.task_flag:
            parser.error("render-engineering requires template, gpu, task-flag")
        prepared = prepare_engineering(config, args.data_root, repo, args.output)
        job_config = args.job_config or args.output / "jobs/engineering_taiji.json"
        result = render_job(args.template, job_config,
                            task_flag=args.task_flag,
                            readable_name=f"meteor MechET PR69 engineering smoke ({args.gpu})",
                            repo=repo, training_config=Path(prepared["training_config"]),
                            gpu_name=args.gpu, model_cache=args.model_cache,
                            model_revision=config["model"]["revision"])
    elif args.action == "render-scientific":
        if not all((args.condition, args.template, args.gpu, args.task_flag, args.model_cache)):
            parser.error("render-scientific requires condition, template, gpu, task-flag, model-cache")
        result = render_scientific_job(
            config, args.data_root, repo, args.output,
            condition=args.condition, template=args.template,
            job_config=args.job_config or args.output / f"jobs/scientific_{args.condition}_taiji.json",
            task_flag=args.task_flag, gpu_name=args.gpu, model_cache=args.model_cache,
        )
    elif args.action == "render-heldout":
        if not args.template or not args.model_cache or not args.task_flag:
            parser.error("render-heldout requires template, model-cache, task-flag")
        result = render_heldout_job(
            args.template, args.job_config or args.output / "jobs/heldout_taiji.json",
            task_flag=args.task_flag, repo=repo, model_cache=args.model_cache,
            model_revision=config["model"]["revision"],
            data=args.data_root / "data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl",
            adapter=args.output / "jobs/engineering_model",
            evaluation_output=args.output / "jobs/engineering_heldout",
        )
    elif args.action == "submit-engineering":
        if not args.donor_task:
            parser.error("submit-engineering requires a proven donor task")
        state_path = args.output / "campaign_state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {"events": []}
        previous = [event for event in state["events"] if event["action"] == "submit-engineering"]
        if len(previous) >= 3:
            raise ValueError("engineering smoke exceeded two infrastructure retries")
        if previous:
            if not args.retry_reason:
                raise ValueError("repeat submission requires an infrastructure retry reason")
            polls = [event["evidence"] for event in state["events"]
                     if event["action"] == "poll-engineering"]
            if not polls or polls[-1]["state"] != "END":
                raise ValueError("previous Taiji instance is not confirmed terminal")
            if polls[-1]["adapter_exists"]:
                raise ValueError("previous engineering run has a checkpoint; do not retry")
            if args.retry_reason == "model_cache_unavailable" and not any(
                "couldn't connect to 'https://huggingface.co'" in line.lower()
                for line in polls[-1]["pod_log_tail"]
            ):
                raise ValueError("model-cache retry does not match the recorded failure")
        rendered = args.job_config or args.output / "jobs/engineering_taiji.json"
        if not rendered.is_file():
            raise FileNotFoundError("render-engineering before submission")
        result = {"submission_output": submit(repo, rendered, args.donor_task, args.client),
                  "job_config": str(rendered), "retry_reason": args.retry_reason,
                  "attempt": len(previous) + 1}
    elif args.action == "submit-heldout":
        if not args.donor_task:
            parser.error("submit-heldout requires a proven donor task")
        rendered = args.job_config or args.output / "jobs/heldout_taiji.json"
        if not rendered.is_file():
            raise FileNotFoundError("render-heldout before submission")
        state = json.loads((args.output / "campaign_state.json").read_text())
        previous = [event for event in state["events"] if event["action"] == "submit-heldout"]
        if len(previous) >= 3:
            raise ValueError("held-out smoke exceeded two infrastructure retries")
        if previous:
            polls = [event["evidence"] for event in state["events"]
                     if event["action"] == "poll-heldout"]
            if not args.retry_reason or not polls or polls[-1]["state"] != "END":
                raise ValueError("held-out retry requires a confirmed terminal infrastructure failure")
            if polls[-1]["evaluation_exists"]:
                raise ValueError("held-out evaluation exists; no retry")
            if args.retry_reason == "runtime_dependency_missing" and not any(
                "bitsandbytes is required" in line for line in polls[-1]["pod_log_tail"]
            ):
                raise ValueError("dependency retry does not match recorded failure")
        result = {"submission_output": submit(repo, rendered, args.donor_task, args.client),
                  "job_config": str(rendered), "retry_reason": args.retry_reason,
                  "attempt": len(previous) + 1}
    elif args.action == "poll-heldout":
        state = json.loads((args.output / "campaign_state.json").read_text())
        submissions = [event["evidence"] for event in state["events"]
                       if event["action"] == "submit-heldout"]
        if not submissions:
            raise ValueError("no recorded held-out submission")
        submission = submissions[-1]
        job = json.loads(Path(submission["job_config"]).read_text())
        found = re.search(r"instance_id:\s*([0-9a-f]{32})", submission["submission_output"])
        if not found:
            raise ValueError("held-out submission has no instance ID")
        result = poll(args.client, job["task_flag"], found.group(1),
                      args.output / "jobs/engineering_model")
        result["evaluation_exists"] = (args.output / "jobs/engineering_heldout/evaluation.json").is_file()
    else:
        state = json.loads((args.output / "campaign_state.json").read_text())
        submissions = [event["evidence"]
                       for event in state["events"] if event["action"] == "submit-engineering"]
        if not submissions:
            raise ValueError("no recorded engineering submission")
        submission = submissions[-1]
        job_path = Path(submission.get("job_config") or args.output / "jobs/engineering_taiji.json")
        job = json.loads(job_path.read_text())
        found = re.search(r"instance_id:\s*([0-9a-f]{32})", submission["submission_output"])
        if not found:
            raise ValueError("submission did not record a Taiji instance ID")
        result = poll(args.client, job["task_flag"], found.group(1),
                      args.output / "jobs/engineering_model")
    if args.action != "plan":
        stage = ("FREEZE_MANIFESTS" if args.action in {"reconcile", "freeze-engineering", "freeze-scientific"}
                 else "TRAIN_MECH_SMOKE" if args.action == "render-scientific" and args.condition == "mech"
                 else "TRAIN_BASE_SMOKE" if args.action in {"prepare-scientific", "render-scientific"}
                 else "ENGINEERING_SMOKE")
        append_ledger(args.output, campaign_id=config["campaign_id"],
                      config_path=args.config, repo=repo, action=args.action,
                      stage=stage, evidence=result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
