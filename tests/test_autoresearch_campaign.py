import json
from pathlib import Path

import pytest

from scripts.autoresearch.run_campaign import plan
from scripts.autoresearch.taiji_backend import poll, render_job
from scripts.autoresearch.evaluate_gate import promotion_evidence
from scripts.autoresearch.ledger import append


def test_plan_keeps_all_scientific_packages_visible(tmp_path: Path) -> None:
    config = {"campaign_id": "toy", "sources": {
        "flower": {"train": None}, "mech_uspto_31k": {"train": None},
        "curated": {"train": None}},
        "evaluation_sources": {
            "r1_multi_reference": None, "r2_plausibility": None,
            "r3_corruptions": None, "r4_pmechdb_challenging": None,
            "r4_pmechrp_pathways": None, "r4_literature_cycles": None,
            "r5_external_predictions": None}}
    result = plan(config, tmp_path, tmp_path / "out")
    stages = {row["stage"]: row for row in result["stages"]}
    assert all(f"RUN_R{i}" in stages for i in range(1, 6))
    assert "r4_literature_cycles" in stages["RUN_R4"]["prerequisites_missing"]
    assert "curated_augmentation_unavailable" in stages["TRAIN_MECH_SMOKE"]["prerequisites_missing"]


def test_taiji_render_enforces_single_gpu_meteor_and_stdout(tmp_path: Path) -> None:
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"GPUName": "A100", "host_num": 1,
        "host_gpu_num": 1, "business_flag": "group", "location": "qy",
        "init_cmd": "old-secret", "start_cmd": "old"}))
    output = tmp_path / "job.json"
    with pytest.raises(ValueError, match="meteor"):
        render_job(template, output, task_flag="wrong", readable_name="x",
                   repo=tmp_path, training_config=tmp_path / "train.yaml", gpu_name="A100")
    render_job(template, output, task_flag="meteor_toy", readable_name="meteor toy",
               repo=tmp_path, training_config=tmp_path / "train.yaml", gpu_name="A100")
    job = json.loads(output.read_text())
    assert job["init_cmd"] == "REPLACE_WITH_PRIVATE_INIT_CMD_FROM_SUCCESSFUL_TASK"
    assert "taiji_run_with_heartbeat.sh" in job["start_cmd"]
    assert "PYTHONUNBUFFERED=1" in job["start_cmd"]
    assert ">" not in job["start_cmd"]
    with pytest.raises(FileExistsError):
        render_job(template, output, task_flag="meteor_toy", readable_name="meteor toy",
                   repo=tmp_path, training_config=tmp_path / "train.yaml", gpu_name="A100")


def test_negative_result_does_not_cancel_other_packages() -> None:
    scorecard = {"packages": {f"r{i}": {"status": "complete", "data_contract_errors": 0}
                               for i in range(1, 6)},
                 "training": {"base_endpoint_top1": 0.15, "mech_endpoint_top1": 0.10,
                              "mech_comparison_identifiable": True},
                 "train_eval_overlap_count": 0}
    evidence = promotion_evidence(scorecard)
    assert evidence["checks"]["all_r1_to_r5_complete"]
    assert evidence["recommendation"] == "RECOMMEND_REVISE_DATA_OR_PROTOCOL"
    assert evidence["automatic_full_scale_launch_allowed"] is False


def test_ledger_resume_is_idempotent_and_config_bound(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 17\n")
    repo = Path(__file__).resolve().parents[1]
    first = append(tmp_path / "campaign", campaign_id="toy", config_path=config,
                   repo=repo, action="freeze-engineering", stage="FREEZE_MANIFESTS",
                   evidence={"sha256": "fixed"})
    second = append(tmp_path / "campaign", campaign_id="toy", config_path=config,
                    repo=repo, action="freeze-engineering", stage="FREEZE_MANIFESTS",
                    evidence={"sha256": "fixed"})
    assert len(first["events"]) == len(second["events"]) == 1
    config.write_text("seed: 18\n")
    with pytest.raises(ValueError, match="identity drift"):
        append(tmp_path / "campaign", campaign_id="toy", config_path=config,
               repo=repo, action="freeze-engineering", stage="FREEZE_MANIFESTS",
               evidence={"sha256": "fixed"})


def test_poll_requires_authoritative_instance_state(tmp_path: Path, monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = '{"state": "TRAINING_RUNNING"}'

    monkeypatch.setattr("scripts.autoresearch.taiji_backend.subprocess.run",
                        lambda *args, **kwargs: Result())
    result = poll(tmp_path / "client", "meteor_toy", "a" * 32, tmp_path / "model")
    assert result["state"] == "TRAINING_RUNNING"
    assert result["adapter_exists"] is False
