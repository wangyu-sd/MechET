#!/usr/bin/env python3
"""Full PR96 State-SFT -> frozen gate -> own-parent history SFT pipeline."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.pr96_artifacts import (validate_preflight, validate_tokens, validate_parent,
                                   verify_resume_checkpoint, training_identity)
from mechet.pr96_eval_runtime import sha, load_validation_report
from mechet.qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--node-cache", type=Path, required=True)
    args = p.parse_args()
    import yaml
    plan = json.loads(args.plan.read_text())
    run = Path(plan["run_dir"])
    lock = (run / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha(plan["reference_approval"]) != plan["reference_approval_sha256"] or sha(plan["base_manifest"]) != plan["base_manifest_sha256"]:
        raise ValueError("PR96_PIPELINE_INPUT_LINEAGE_CHANGED")
    args.node_cache.mkdir(parents=True, exist_ok=True)
    def emit(stage, **extra):
        value = {"stage": stage, "track": plan["track"],
                 "timestamp": datetime.now(timezone.utc).isoformat(), **extra}
        text = json.dumps(value)
        print(text, flush=True)
        with (run / "pipeline_progress.jsonl").open("a") as out:
            out.write(text + "\n")
        (run / "pipeline_status.json").write_text(text + "\n")
    def command(script, *options):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               str(ROOT / "scripts" / script), *map(str, options)]
        emit("launch", command=cmd)
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)  # inherit Taiji stdout/stderr; never redirect
        except Exception as exc:
            emit("subprocess-failed", command=cmd, error=f"{type(exc).__name__}:{exc}")
            raise
    def save(path, cfg):
        Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Node-local, per-track immutable base cache; no shared small-file imports.
    base = json.loads(Path(plan["base_manifest"]).read_text())
    if base["model_id"] != OFFICIAL_MODEL or base["revision"] != OFFICIAL_REVISION:
        raise ValueError("PR96_WRONG_PINNED_BASE")
    local_model = Path(os.environ["HF_HUB_CACHE"]) / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / OFFICIAL_REVISION
    local_model.mkdir(parents=True, exist_ok=True)
    for index, (name, record) in enumerate(base["files"].items(), 1):
        source, dest = Path(base["snapshot"]) / name, local_model / name
        if not dest.exists():
            shutil.copy2(source, dest)
        if sha(dest) != record["sha256"]:
            raise ValueError(f"PR96_LOCAL_BASE_HASH_MISMATCH:{name}")
        emit("base-file-staged", completed=index, total=len(base["files"]), file=name)

    configs = {stage: yaml.safe_load(Path(path).read_text()) for stage,path in plan["configs"].items()}
    for stage, cfg in configs.items():
        cache = run / "tokens" / stage
        expected = {split:(sha(cfg[key]), cfg["contract"][count]) for split,key,count in
                    (("train", "train_file", "expected_train_rows"), ("validation", "validation_file", "expected_validation_rows"))}
        manifest = cache / "manifest.json"
        if not manifest.is_file():
            cfg["pretokenized_cache_dir"] = str(cache)
            save(plan["configs"][stage], cfg)
            emit("tokenization-start", training_stage=stage, expected=expected)
            command("prepare_tool_sft_arrow.py", "--config", plan["configs"][stage])
        value = json.loads(manifest.read_text())
        validate_tokens(value, expected)
        local = args.node_cache / f"tokens_{stage}"
        local.mkdir(exist_ok=True)
        for record in value["splits"].values():
            for path in record["arrow_files"]:
                source, dest = Path(path), local / Path(path).name
                if not dest.exists():
                    shutil.copy2(source, dest)
                if sha(dest) != record["arrow_sha256"][dest.name]:
                    raise ValueError("PR96_LOCAL_TOKEN_CACHE_HASH_MISMATCH")
        shutil.copy2(manifest, local / "manifest.json")
        cfg["pretokenized_cache_dir"] = str(local)
        largest = value["splits"]["train"]["max_supervised_tokens"]
        cfg["pr96"]["max_new_tokens"] = max(512, ((largest + 128 + 255)//256)*256)
        cfg["pr96"]["token_cache_manifest_sha256"] = sha(manifest)
        cfg["pr96"]["config_file"] = plan["configs"][stage]
        save(plan["configs"][stage], cfg)
        emit("tokenization-audit-passed", training_stage=stage,
             rows=value["splits"]["train"]["n_rows"], zero_state_cropping=True,
             max_supervised_tokens=largest, generation_budget=cfg["pr96"]["max_new_tokens"])

    # Scratch preflight has no adapter output; formal State-SFT reloads the base.
    preflight = run / ("preflight_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    command("preflight_pr96.py", "--config", plan["configs"]["state"], "--output", preflight)
    preflight_manifest = preflight / "manifest.json"
    validate_preflight(json.loads(preflight_manifest.read_text()), plan["track"])
    for stage, cfg in configs.items():
        cfg["pr96"]["preflight_manifest"] = str(preflight_manifest)
        cfg["pr96"]["preflight_manifest_sha256"] = sha(preflight_manifest)
        status = {"training_allowed": True, "status": "pr96_reviewed_subset_tokens_and_deployed_preflight_verified",
                  "reference_approval_sha256": plan["reference_approval_sha256"],
                  "token_cache_manifest_sha256": cfg["pr96"]["token_cache_manifest_sha256"],
                  "deployed_preflight_sha256": sha(preflight_manifest), "stage": stage}
        (Path(cfg["train_file"]).parent / "ARTIFACT_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        save(plan["configs"][stage], cfg)
    emit("formal-training-authorized", preflight=str(preflight_manifest))

    def train(stage):
        cfg = configs[stage]
        model_dir = Path(cfg["output_dir"])
        finished = model_dir / "adapter_manifest.json"
        if finished.is_file():
            previous = json.loads(finished.read_text())
            if previous.get("pr96", {}).get("full_epochs_completed", 0) < cfg["training"]["num_train_epochs"]:
                raise ValueError("PR96_EXISTING_ROOT_ADAPTER_IS_NOT_FULL_STAGE")
            if (previous.get("base_model_revision") != OFFICIAL_REVISION
                    or previous["pr96"].get("reference_approval_sha256") != plan["reference_approval_sha256"]
                    or previous["pr96"].get("training_identity") != training_identity(cfg)
                    or previous["adapter_sha256"] != sha(model_dir / "adapter_model.safetensors")):
                raise ValueError("PR96_REFUSING_WRONG_EXISTING_STAGE_LINEAGE")
            emit("reuse-completed-stage", training_stage=stage, adapter_manifest=str(finished))
            return previous
        options = ["--config", plan["configs"][stage]]
        checkpoints = sorted(model_dir.glob("checkpoint-*"), key=lambda x:int(x.name.split("-")[-1]))
        if cfg["pr96"].get("performance_resume_anchor") or checkpoints:
            if cfg["pr96"].get("performance_resume_anchor"):
                from mechet.pr96_performance import select_performance_resume_checkpoint
                candidate = select_performance_resume_checkpoint(cfg)
            else:
                candidate = checkpoints[-1]
            verify_resume_checkpoint(cfg, candidate)
            options += ["--resume-from-checkpoint", candidate]
        emit("formal-training-start", training_stage=stage, epochs=cfg["training"]["num_train_epochs"],
             effective_batch=64, expected_rows=cfg["contract"]["expected_train_rows"])
        command("train_tool_sft.py", *options)
        result = json.loads(finished.read_text())
        if result["pr96"]["full_epochs_completed"] < cfg["training"]["num_train_epochs"] - 1e-5:
            raise ValueError("PR96_FORMAL_STAGE_DID_NOT_FINISH_FULL_EPOCHS")
        return result

    state = train("state")
    parent = Path(state["pr96"]["best_checkpoint"])
    gate_dir = run / "validation" / "state_best_paired"
    if load_validation_report(configs["state"], parent, gate_dir, "paired") is None:
        command("evaluate_pr96.py", "--config", plan["configs"]["state"], "--adapter", parent,
                "--output", gate_dir)
    gate_path = gate_dir / "report.json"
    gate = load_validation_report(configs["state"], parent, gate_dir, "paired")
    if not gate["gate"]["go"]:
        emit("stage-one-no-go", gate=gate["gate"], report=str(gate_path), stage_two_started=False)
        return
    parent_manifest = json.loads((parent / "pr96_checkpoint_manifest.json").read_text())
    validate_parent(parent_manifest, gate, track=plan["track"], weight_sha=sha(parent / "adapter_model.safetensors"),
                    cohort_sha=configs["state"]["pr96"]["cohorts_sha256"])
    configs["history"]["initial_adapter_path"] = str(parent)
    configs["history"]["pr96"]["stage_one_gate_report"] = str(gate_path)
    save(plan["configs"]["history"], configs["history"])
    emit("stage-one-go-start-stage-two", parent=str(parent), parent_weight_sha256=sha(parent / "adapter_model.safetensors"))
    hist = train("history")
    best = Path(hist["pr96"]["best_checkpoint"])
    final_reports = {}
    for cohort in ("paired", "independent"):
        target = run / "validation" / f"history_best_{cohort}"
        if cohort == "paired" and not target.exists():
            metadata = json.loads((best / "pr96_checkpoint_manifest.json").read_text())
            previous_report = metadata.get("epoch_validation_report")
            if previous_report:
                old = load_validation_report(configs["history"], best, Path(previous_report).parent, cohort)
                if old:
                    shutil.copytree(Path(previous_report).parent, target)
                    emit("reuse-exact-paired-epoch-validation", source=previous_report, target=str(target))
        if load_validation_report(configs["history"], best, target, cohort) is None:
            command("evaluate_pr96.py", "--config", plan["configs"]["history"], "--adapter", best,
                    "--output", target, "--cohort", cohort)
        final_reports[cohort] = load_validation_report(configs["history"], best, target, cohort)
    comparison = {key: {"state": gate["metrics"][key], "history": final_reports["paired"]["metrics"][key]}
                  for key in ("structural_hits", "full_hits", "root_import_fragment", "tf_flow_successor", "tf_finish")}
    (run / "paired_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    emit("two-stages-complete", comparison=comparison, state_checkpoint=str(parent),
         history_checkpoint=str(best), independent_report=str(run / "validation/history_best_independent/report.json"),
         complete_test_evaluation_executed=False)


if __name__ == "__main__":
    main()
