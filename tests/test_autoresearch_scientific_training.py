import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.autoresearch.scientific_training import (
    build_configs, prepare, render_scientific_job,
)
from scripts.autoresearch.stratified_manifest import digest
from scripts.autoresearch.run_campaign import check_scientific_submission
from scripts.autoresearch.audit_overlap import audit as audit_overlap


REPO = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path):
    root = tmp_path / "data_root"
    output = tmp_path / "campaign"
    root.mkdir()
    manifests = output / "scientific_freeze/manifests"
    manifests.mkdir(parents=True)
    evaluation_sources = {}
    evaluation_hashes = {}
    source_rows = {"r1_multi_reference": 220, "r2_plausibility": 800,
                   "r3_corruptions": 288, "r4_pmechdb_challenging": 1,
                   "r4_pmechrp_pathways": 350, "r4_literature_cycles": 12,
                   "r5_external_predictions": 200}
    for name in ("r1_multi_reference", "r2_plausibility", "r3_corruptions",
                 "r4_pmechdb_challenging", "r4_pmechrp_pathways",
                 "r4_literature_cycles", "r5_external_predictions"):
        directory = root / name
        directory.mkdir()
        path = directory / f"{name}.jsonl"
        path.write_text('{"product_smiles":"C"}\n' * source_rows[name])
        (directory / "manifest.json").write_text(json.dumps({
            "artifact_type": f"{name}_frozen_manifest",
            "cohort_sha256": digest(path),
        }))
        evaluation_sources[name] = str(path)
        evaluation_hashes[name] = digest(path)
    validation = root / "valid.jsonl"
    validation.write_text('{"id":"valid"}\n' * 256)
    flower_manifest = root / "flower_manifest.json"
    flower_manifest.write_text(json.dumps({"splits": {"valid": {
        "target": str(validation), "output_sha256": digest(validation)}}}))
    files = {}
    for condition in ("base", "mech"):
        train = manifests / f"{condition}_state_sft.jsonl"
        train.write_text("".join(json.dumps({"id": f"{condition}_{index}",
                                             "source_id": f"{condition}_reaction_{index}",
                                             "target_smiles": "CCO"}) + "\n"
                                 for index in range(4)))
        strata = manifests / f"{condition}_strata.jsonl"
        strata.write_text("".join(json.dumps({"stable_id": f"{condition}_{index}"}) + "\n"
                                  for index in range(4)))
        files[condition] = {"train": str(train), "train_sha256": digest(train),
                            "strata": str(strata), "strata_sha256": digest(strata),
                            "rows": 4}
    source = root / "source.jsonl"
    source.write_text('{"id":"source"}\n')
    frozen = {
        "campaign_id": "toy", "engineering_only": False,
        "evaluation_hashes": evaluation_hashes,
        "mech_comparison_identifiable": True, "curated_accepted_rows": 2,
        "resolved_quotas": {
            "base": {"flower": 2, "mech_uspto_31k": 2},
            "mech": {"flower": 1, "mech_uspto_31k": 1, "curated": 2},
        },
        "sources": {"curated": {"path": str(source), "sha256": digest(source)}},
        "files": files,
    }
    (manifests / "freeze.json").write_text(json.dumps(frozen))
    config = {
        "campaign_id": "toy", "seed": 17,
        "model": {"name": "Qwen/Qwen3-0.6B", "revision": "c" * 40,
                  "training_template": "configs/agent/natural_language_event_v2_qwen3_8b_h20.yaml"},
        "scientific_smoke": {"rows_per_condition": 4},
        "evaluation_sources": evaluation_sources,
        "sources": {"flower": {"manifest": str(flower_manifest)}},
    }
    return config, root, output


def test_scientific_conditions_have_matched_full_length_configs(tmp_path: Path) -> None:
    config, root, output = fixture(tmp_path)
    configs, report = build_configs(config, root, REPO, output)
    assert report["rows_per_condition"] == 4
    assert report["curated_accepted_rows"] == 2
    assert configs["base"]["training"] == configs["mech"]["training"]
    assert configs["base"]["train_file"] != configs["mech"]["train_file"]
    assert configs["base"]["validation_file"] == configs["mech"]["validation_file"]
    assert configs["base"]["training"]["max_steps"] == -1
    assert configs["base"]["training"]["num_train_epochs"] == 1.0
    assert configs["base"]["contract"]["expected_train_rows"] == 4
    assert "reaction_denominator" not in configs["base"]["contract"]


def test_independent_overlap_audit_detects_product_overlap(tmp_path: Path) -> None:
    config, root, output = fixture(tmp_path)
    clean = audit_overlap(config, root, output)
    assert clean["passed"] is True
    assert clean["overlap_count"] == 0
    (output / "manifests/train_eval_overlap_audit.json").unlink()
    frozen_path = output / "scientific_freeze/manifests/freeze.json"
    frozen = json.loads(frozen_path.read_text())
    selected = Path(frozen["files"]["mech"]["train"])
    rows = [json.loads(line) for line in selected.read_text().splitlines()]
    rows[0]["target_smiles"] = "C"
    selected.write_text("".join(json.dumps(row) + "\n" for row in rows))
    frozen["files"]["mech"]["train_sha256"] = digest(selected)
    frozen_path.write_text(json.dumps(frozen))
    compromised = audit_overlap(config, root, output)
    assert compromised["passed"] is False
    assert compromised["conditions"]["mech"]["product_overlap_count"] == 1
    with pytest.raises(ValueError, match="overlap detected"):
        prepare(config, root, REPO, output)
    assert not (output / "jobs/scientific_prepared.json").exists()


@pytest.mark.parametrize("change,reason", [
    ("engineering_only", "engineering rows"),
    ("zero_curated", "curated rows"),
    ("quota_mismatch", "source quotas"),
    ("eval_drift", "evaluation source hash differs"),
    ("train_drift", "training file hash drifted"),
])
def test_scientific_preparation_fails_closed(tmp_path: Path, change: str, reason: str) -> None:
    config, root, output = fixture(tmp_path)
    frozen_path = output / "scientific_freeze/manifests/freeze.json"
    frozen = json.loads(frozen_path.read_text())
    if change == "engineering_only":
        frozen["engineering_only"] = True
    elif change == "zero_curated":
        frozen["curated_accepted_rows"] = 0
    elif change == "quota_mismatch":
        frozen["resolved_quotas"]["base"]["flower"] = 1
    elif change == "eval_drift":
        Path(config["evaluation_sources"]["r2_plausibility"]).write_text("changed\n")
    elif change == "train_drift":
        Path(frozen["files"]["base"]["train"]).write_text("changed\n")
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises((ValueError, FileNotFoundError), match=reason):
        build_configs(config, root, REPO, output)
    assert not (output / "jobs/scientific_prepared.json").exists()


def test_scientific_prepare_audits_both_conditions_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, output = fixture(tmp_path)
    calls = []
    cache = tmp_path / "cache"
    snapshot = cache / "models--Qwen--Qwen3-0.6B/snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"toy")

    def run(command, **_kwargs):
        calls.append(command)
        cfg = Path(command[command.index("--config") + 1])
        import yaml
        training = yaml.safe_load(cfg.read_text())
        selected = Path(training["train_file"])
        if "--dry-run" in command:
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "n_rows": 4, "train_file_sha256": digest(selected),
                "validation_file_sha256": digest(Path(training["validation_file"])),
                "validation": {"n_rows": 256}, "max_steps": -1,
                "num_train_epochs": 1.0,
                "model_name_or_path": "Qwen/Qwen3-0.6B",
            }))
        assert _kwargs["env"]["HF_HUB_CACHE"] == str(cache)
        assert _kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        audit = Path(command[command.index("--output") + 1])
        audit.write_text(json.dumps({
            "passed": True, "rows": 4, "input_sha256": digest(selected),
            "resolved_model_revision": "c" * 40,
            "total_input_tokens": 100, "total_supervised_tokens": 20,
        }))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("scripts.autoresearch.scientific_training.subprocess.run", run)
    first = prepare(config, root, REPO, output, model_cache=cache)
    assert len(calls) == 4
    assert first["files"]["base"]["total_supervised_tokens"] == 20
    assert first["files"]["mech"]["total_supervised_tokens"] == 20
    assert prepare(config, root, REPO, output, model_cache=cache) == first
    assert len(calls) == 4
    Path(first["files"]["mech"]["token_audit"]).write_text("drift\n")
    with pytest.raises(ValueError, match="token audit drifted"):
        prepare(config, root, REPO, output, model_cache=cache)


def test_scientific_job_render_never_reuses_engineering_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, output = fixture(tmp_path)
    frozen = output / "scientific_freeze/manifests/freeze.json"
    training_config = output / "jobs/scientific_base_training.yaml"
    training_config.parent.mkdir(parents=True)
    training_config.write_text("condition_name: toy_base\n")
    prepared = {
        "scientific_freeze": str(frozen), "scientific_freeze_sha256": digest(frozen),
        "model_revision": "c" * 40, "rows_per_condition": 4,
        "expected_optimizer_updates_per_condition": 1,
        "files": {"base": {"training_config": str(training_config),
                           "training_config_sha256": digest(training_config),
                           "train_sha256": "a" * 64, "token_audit_sha256": "b" * 64}},
    }
    monkeypatch.setattr("scripts.autoresearch.scientific_training.prepare",
                        lambda *_args, **_kwargs: prepared)
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"GPUName": "A100", "host_num": 1,
                                    "host_gpu_num": 1, "business_flag": "group",
                                    "location": "qy"}))
    cache = tmp_path / "cache"
    snapshot = cache / "models--Qwen--Qwen3-0.6B/snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"toy")
    rendered = output / "jobs/scientific_base_taiji.json"
    result = render_scientific_job(
        config, root, REPO, output, condition="base", template=template,
        job_config=rendered, task_flag="meteor_toy_base", gpu_name="A100",
        model_cache=cache,
    )
    job = json.loads(rendered.read_text())
    assert result["scientific_freeze_sha256"] == digest(frozen)
    assert result["rows"] == 4
    assert "scientific base condition: 4" in job["task_description"]
    assert "engineering smoke" not in job["task_description"]
    assert "taiji_run_with_heartbeat.sh" in job["start_cmd"]
    assert "HF_HUB_OFFLINE=1" in job["start_cmd"]
    assert ">" not in job["start_cmd"]


def test_scientific_job_render_requires_freeze_before_template(
    tmp_path: Path,
) -> None:
    config, root, output = fixture(tmp_path)
    (output / "scientific_freeze/manifests/freeze.json").unlink()
    with pytest.raises(FileNotFoundError, match="scientific freeze"):
        render_scientific_job(
            config, root, REPO, output, condition="base",
            template=tmp_path / "not_used.json", job_config=tmp_path / "job.json",
            task_flag="meteor_toy", gpu_name="A100", model_cache=tmp_path,
        )
    assert not (tmp_path / "job.json").exists()


def test_scientific_submit_gate_requires_render_and_terminal_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, output = fixture(tmp_path)
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text("campaign_id: toy\n")
    (output / "jobs").mkdir(exist_ok=True)
    (output / "jobs/scientific_prepared.json").write_text("{}\n")
    prepared = {
        "scientific_freeze_sha256": digest(output / "scientific_freeze/manifests/freeze.json"),
        "model_revision": "c" * 40,
        "files": {"base": {"training_config_sha256": "a" * 64,
                           "token_audit_sha256": "b" * 64,
                           "train_sha256": "d" * 64}},
    }
    monkeypatch.setattr("scripts.autoresearch.run_campaign.prepare_scientific",
                        lambda *_args, **_kwargs: prepared)
    job = output / "jobs/scientific_base_taiji.json"

    def write_job(flag: str) -> dict:
        job.write_text(json.dumps({
            "task_flag": flag, "readable_name": "meteor scientific base",
            "start_cmd": "bash scripts/taiji_run_with_heartbeat.sh train",
        }))
        return {"scientific_condition": "base", "rendered_template": str(job),
                "task_flag": flag, "job_config_sha256": digest(job),
                "scientific_freeze_sha256": prepared["scientific_freeze_sha256"],
                "training_config_sha256": "a" * 64,
                "token_audit_sha256": "b" * 64}

    state = {"campaign_id": "toy", "config_sha256": digest(config_path),
             "events": []}
    first_render = write_job("meteor_base_01")
    (output / "campaign_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="no audited render"):
        check_scientific_submission(config, config_path, root, REPO, output,
                                    condition="base", job_config=job,
                                    retry_reason=None)
    state["events"].append({"action": "render-scientific", "evidence": first_render})
    (output / "campaign_state.json").write_text(json.dumps(state))
    first = check_scientific_submission(config, config_path, root, REPO, output,
                                        condition="base", job_config=job,
                                        retry_reason=None)
    assert first["attempt"] == 1
    state["events"].append({"action": "submit-scientific", "evidence": {
        **first, "instance_id": "1" * 32}})
    second_render = write_job("meteor_base_02")
    state["events"].append({"action": "render-scientific", "evidence": second_render})
    (output / "campaign_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="terminal instance"):
        check_scientific_submission(config, config_path, root, REPO, output,
                                    condition="base", job_config=job,
                                    retry_reason="transient_runtime")
    state["events"].append({"action": "poll-scientific", "evidence": {
        "scientific_condition": "base", "instance_id": "1" * 32,
        "state": "END", "adapter_exists": False}})
    (output / "campaign_state.json").write_text(json.dumps(state))
    second = check_scientific_submission(config, config_path, root, REPO, output,
                                         condition="base", job_config=job,
                                         retry_reason="transient_runtime")
    assert second["attempt"] == 2
    state["events"][-1]["evidence"]["adapter_exists"] = True
    (output / "campaign_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="checkpoint exists"):
        check_scientific_submission(config, config_path, root, REPO, output,
                                    condition="base", job_config=job,
                                    retry_reason="transient_runtime")
