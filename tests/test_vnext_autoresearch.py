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
