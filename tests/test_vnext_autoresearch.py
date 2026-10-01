import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_vnext_autoresearch.py"
SPEC = importlib.util.spec_from_file_location("vnext_autoresearch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_campaign_loads_and_freezes_environment(monkeypatch):
    monkeypatch.setenv("MECHET_AUTORESEARCH_CODE_MIRROR", "/tmp/mechet-code-mirror")
    monkeypatch.setenv("MECHET_TAIJI_CLIENT", "/tmp/taiji-client")
    monkeypatch.setenv("MECHET_TAIJI_DONOR_TASK", "donor-success")
    campaign, digest = MODULE.load_campaign(
        ROOT / "configs" / "autoresearch" / "vnext_p0_20261001.yaml"
    )
    assert len(digest) == 64
    assert campaign["code_mirror"] == "/tmp/mechet-code-mirror"
    assert campaign["taiji"]["client"] == "/tmp/taiji-client"
    stages = {stage["id"]: stage for stage in campaign["stages"]}
    assert stages["pointer_valid256"]["depends_on"] == ["pointer_smoke16"]
    assert stages["pointer_valid1319"]["depends_on"] == ["pointer_valid256"]
    assert stages["runtime_benchmark"]["kind"] == "taiji"
    assert stages["packing_benchmark"]["kind"] == "taiji"


def test_campaign_rejects_unresolved_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_AUTORESEARCH_VARIABLE", raising=False)
    path = tmp_path / "campaign.yaml"
    unresolved = "$" + "{MISSING_AUTORESEARCH_VARIABLE}"
    path.write_text(
        "schema_version: 1\ncampaign: bad\ncode_mirror: " + unresolved + "\nstages: []\n"
    )
    with pytest.raises(ValueError, match="unresolved environment"):
        MODULE.load_campaign(path)


def test_gate_evaluation_reads_frozen_json_metrics(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"delta": 0.03125, "invalid": 0}))
    stage = {
        "required_globs": [str(report)],
        "metrics": {
            "delta": {"type": "json", "path": str(report), "field": "delta"},
            "invalid": {"type": "json", "path": str(report), "field": "invalid"},
        },
        "gates": [
            {"name": "positive", "lhs": "delta", "op": ">", "rhs": 0},
            {"name": "valid", "lhs": "invalid", "op": "==", "rhs": 0},
        ],
    }
    passed, details = MODULE.evaluate_stage(stage)
    assert passed is True
    assert details["metrics"]["delta"] == pytest.approx(0.03125)
    assert all(item["passed"] for item in details["gates"])


def test_scientific_stop_blocks_dependents_without_rewriting_campaign():
    campaign = {
        "stages": [
            {"id": "a", "kind": "artifact"},
            {"id": "b", "kind": "artifact", "depends_on": ["a"]},
        ]
    }
    ledger = {
        "stages": {
            "a": {"state": "SCIENTIFIC_STOP", "history": []},
            "b": {"state": "PENDING", "history": []},
        }
    }
    changed = MODULE.reconcile_blocked(campaign, ledger)
    assert changed is True
    assert ledger["stages"]["b"]["state"] == "BLOCKED"
    assert "SCIENTIFIC_STOP" in ledger["stages"]["b"]["history"][-1]["reason"]


def test_existing_ledger_rejects_campaign_or_code_drift(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "x").write_text("1")
    subprocess.run(["git", "add", "x"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    campaign = {"campaign": "x", "stages": [{"id": "s"}]}
    ledger_path = tmp_path / "ledger.json"
    ledger = MODULE.load_or_create_ledger(ledger_path, campaign, "a" * 64, repo)
    assert ledger["campaign_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="campaign YAML changed"):
        MODULE.load_or_create_ledger(ledger_path, campaign, "b" * 64, repo)
    (repo / "x").write_text("2")
    subprocess.run(["git", "add", "x"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "move"], cwd=repo, check=True, capture_output=True)
    with pytest.raises(ValueError, match="code revision changed"):
        MODULE.load_or_create_ledger(ledger_path, campaign, "a" * 64, repo)


def test_task_config_token_replacement_is_recursive():
    payload = {"start_cmd": "run __AUTORESEARCH_GIT_HEAD__", "nested": ["__AUTORESEARCH_GIT_HEAD__"]}
    replaced = MODULE.replace_token(payload, "__AUTORESEARCH_GIT_HEAD__", "abc123")
    assert replaced["start_cmd"] == "run abc123"
    assert replaced["nested"] == ["abc123"]


def test_retry_quarantines_partial_outputs(tmp_path):
    partial = tmp_path / "partial-output"
    partial.mkdir()
    (partial / "x").write_text("partial")
    stage = {
        "max_infra_retries": 1,
        "retry_cleanup_globs": [str(partial)],
    }
    record = {"state": "RUNNING", "attempt": 1, "history": []}
    MODULE.schedule_retry_or_fail(stage, record, event="taiji_failed")
    assert record["state"] == "PENDING"
    assert not partial.exists()
    moved = record["history"][-1]["quarantined_outputs"]
    assert len(moved) == 1
    quarantined = Path(moved[0])
    assert quarantined.is_dir()
    assert (quarantined / "x").read_text() == "partial"


def test_scientific_gate_failure_is_not_infrastructure_retry(tmp_path):
    report = tmp_path / "metric.json"
    report.write_text(json.dumps({"delta": -0.1}))
    stage = {
        "required_globs": [str(report)],
        "metrics": {"delta": {"type": "json", "path": str(report), "field": "delta"}},
        "gates": [{"lhs": "delta", "op": ">", "rhs": 0}],
    }
    record = {"state": "RUNNING", "attempt": 1, "history": []}
    MODULE.finalize_scientific_stage(stage, record)
    assert record["state"] == "SCIENTIFIC_STOP"
    assert record["attempt"] == 1


def test_existing_code_mirror_is_never_reset_or_cleaned(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked").write_text("frozen")
    subprocess.run(["git", "add", "tracked"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "frozen"], cwd=repo, check=True)
    sha = MODULE.git_head(repo)
    mirror = tmp_path / "mirror"
    MODULE.sync_code_mirror(repo, mirror, sha)
    assert (mirror / "tracked").read_text() == "frozen"

    (mirror / "private-note").write_text("preserve")
    with pytest.raises(ValueError, match="uncommitted changes"):
        MODULE.sync_code_mirror(repo, mirror, sha)
    assert (mirror / "private-note").read_text() == "preserve"


def test_taiji_query_failure_does_not_look_like_pending(monkeypatch, tmp_path):
    class Failed:
        returncode = 2
        stdout = ""

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Failed())
    with pytest.raises(RuntimeError, match="instance_list failed"):
        MODULE.taiji_rows(tmp_path / "client", "meteor_task")


def test_pointer_recovery_campaign_preserves_gates_and_isolates_tasks(monkeypatch):
    monkeypatch.setenv("MECHET_AUTORESEARCH_CODE_MIRROR", "/tmp/pointer-recovery-mirror")
    monkeypatch.setenv("MECHET_TAIJI_CLIENT", "/tmp/taiji-client")
    monkeypatch.setenv("MECHET_TAIJI_DONOR_TASK", "donor-success")
    original, _ = MODULE.load_campaign(ROOT / "configs/autoresearch/vnext_p0_20261001.yaml")
    recovery, _ = MODULE.load_campaign(
        ROOT / "configs/autoresearch/vnext_p0_pointer_recovery_20261001.yaml"
    )
    prior = {stage["id"]: stage for stage in original["stages"]}
    repaired = {stage["id"]: stage for stage in recovery["stages"]}
    for stage_id in ("pointer_smoke16", "pointer_valid256", "pointer_valid1319"):
        assert repaired[stage_id]["gates"] == prior[stage_id]["gates"]
        assert repaired[stage_id]["task_flag_suffix"] == "_repair1"
        config = json.loads((ROOT / repaired[stage_id]["taiji_config"]).read_text())
        for old, new in repaired[stage_id]["config_replacements"].items():
            config = MODULE.replace_token(config, old, new)
        assert "pointer-recovery" in config["start_cmd"]
        assert "repair1" in config["start_cmd"]
    for stage_id in ("runtime_benchmark", "packing_benchmark"):
        assert repaired[stage_id]["kind"] == "artifact"
        assert repaired[stage_id]["gates"] == prior[stage_id]["gates"]


def test_runtime_rankfix_preserves_scientific_gates(monkeypatch):
    monkeypatch.setenv("MECHET_AUTORESEARCH_CODE_MIRROR", "/tmp/runtime-rankfix-mirror")
    monkeypatch.setenv("MECHET_TAIJI_CLIENT", "/tmp/taiji-client")
    monkeypatch.setenv("MECHET_TAIJI_DONOR_TASK", "donor-success")
    original, _ = MODULE.load_campaign(ROOT / "configs/autoresearch/vnext_p0_20261001.yaml")
    rankfix, _ = MODULE.load_campaign(ROOT / "configs/autoresearch/vnext_runtime_rankfix_20261001.yaml")
    prior = next(stage for stage in original["stages"] if stage["id"] == "runtime_benchmark")
    current = rankfix["stages"][0]
    assert current["gates"] == prior["gates"]
    assert current["metrics"] == prior["metrics"]
    assert current["task_flag_suffix"] == "_rankfix1"
    config = json.loads((ROOT / current["taiji_config"]).read_text())
    for old, new in current["config_replacements"].items():
        config = MODULE.replace_token(config, old, new)
    assert "vnext-runtime-rankfix" in config["start_cmd"]


def test_runtime_cpufix_preserves_scientific_gates_and_caps_worker_threads(monkeypatch):
    monkeypatch.setenv("MECHET_AUTORESEARCH_CODE_MIRROR", "/tmp/runtime-cpufix-mirror")
    monkeypatch.setenv("MECHET_TAIJI_CLIENT", "/tmp/taiji-client")
    monkeypatch.setenv("MECHET_TAIJI_DONOR_TASK", "donor-success")
    original, _ = MODULE.load_campaign(ROOT / "configs/autoresearch/vnext_p0_20261001.yaml")
    cpufix, _ = MODULE.load_campaign(ROOT / "configs/autoresearch/vnext_runtime_cpufix_20261001.yaml")
    prior = next(stage for stage in original["stages"] if stage["id"] == "runtime_benchmark")
    current = cpufix["stages"][0]
    assert current["gates"] == prior["gates"]
    assert current["metrics"] == prior["metrics"]
    assert current["task_flag_suffix"] == "_cpufix1"
    config = json.loads((ROOT / current["taiji_config"]).read_text())
    for old, new in current["config_replacements"].items():
        config = MODULE.replace_token(config, old, new)
    assert "vnext-runtime-cpufix" in config["start_cmd"]
    launcher = (ROOT / "scripts/run_taiji_vnext_runtime_benchmark.sh").read_text()
    assert 'OMP_NUM_THREADS="${VNEXT_CPU_THREADS_PER_WORKER:-2}"' in launcher
    assert 'MKL_NUM_THREADS="$OMP_NUM_THREADS"' in launcher
    assert 'OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"' in launcher
