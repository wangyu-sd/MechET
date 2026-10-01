#!/usr/bin/env python3
"""Finite-state AutoResearch controller for the MechET vNext experiment ladder.

The controller automates execution, monitoring, predeclared gates, and reporting.
It is deliberately not a self-modifying research agent: the campaign YAML,
thresholds, datasets, seeds, and commands are hashed into the ledger and cannot
change during a run.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import operator
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml


TERMINAL = {"PASSED", "SCIENTIFIC_STOP", "INFRA_FAILED", "BLOCKED", "SKIPPED"}
OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if "$" in expanded:
            raise ValueError(f"unresolved environment variable in: {value}")
        return expanded
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def replace_token(value: Any, token: str, replacement: str) -> Any:
    if isinstance(value, str):
        return value.replace(token, replacement)
    if isinstance(value, list):
        return [replace_token(item, token, replacement) for item in value]
    if isinstance(value, dict):
        return {key: replace_token(item, token, replacement) for key, item in value.items()}
    return value


def load_campaign(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported AutoResearch schema version")
    stages = payload.get("stages") or []
    ids = [str(stage["id"]) for stage in stages]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate stage id")
    known = set(ids)
    for stage in stages:
        missing = set(stage.get("depends_on") or []) - known
        if missing:
            raise ValueError(f"{stage['id']}: unknown dependencies {sorted(missing)}")
        if stage.get("uses_test") and not stage.get("requires_test_unlock"):
            raise ValueError(f"{stage['id']}: test stages require an explicit unlock")
    return expand(payload), sha256_bytes(raw)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def initial_ledger(campaign: dict[str, Any], digest: str, repo: Path) -> dict[str, Any]:
    return {
        "artifact_type": "mechet_vnext_autoresearch_ledger_v1",
        "campaign": campaign["campaign"],
        "campaign_sha256": digest,
        "git_head": git_head(repo),
        "created_at_unix": time.time(),
        "stages": {
            str(stage["id"]): {
                "state": "PENDING",
                "attempt": 0,
                "history": [],
            }
            for stage in campaign["stages"]
        },
    }


def load_or_create_ledger(
    path: Path, campaign: dict[str, Any], digest: str, repo: Path
) -> dict[str, Any]:
    if not path.exists():
        ledger = initial_ledger(campaign, digest, repo)
        atomic_json(path, ledger)
        return ledger
    ledger = json.loads(path.read_text())
    if ledger.get("campaign_sha256") != digest:
        raise ValueError("campaign YAML changed after the run started")
    if ledger.get("git_head") != git_head(repo):
        raise ValueError("code revision changed after the run started; start a new campaign")
    expected = {str(stage["id"]) for stage in campaign["stages"]}
    if set(ledger.get("stages", {})) != expected:
        raise ValueError("ledger stage set differs from frozen campaign")
    return ledger


def stage_map(campaign: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(stage["id"]): dict(stage) for stage in campaign["stages"]}


def append_history(record: dict[str, Any], event: str, **fields: Any) -> None:
    record.setdefault("history", []).append({"time": time.time(), "event": event, **fields})


def dependency_state(
    stage: Mapping[str, Any], ledger: Mapping[str, Any]
) -> tuple[bool, str | None]:
    for dep in stage.get("depends_on") or []:
        state = ledger["stages"][dep]["state"]
        if state == "PASSED":
            continue
        if state in {"SCIENTIFIC_STOP", "INFRA_FAILED", "BLOCKED", "SKIPPED"}:
            return False, f"dependency {dep} ended as {state}"
        return False, None
    return True, None


def required_artifacts(stage: Mapping[str, Any]) -> tuple[bool, dict[str, int]]:
    counts: dict[str, int] = {}
    for pattern in stage.get("required_globs") or []:
        matches = sorted(glob.glob(str(pattern)))
        counts[str(pattern)] = len(matches)
        if not matches:
            return False, counts
    return True, counts


def dot_get(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(key)
    return current


def _jsonl_values(pattern: str, field: str) -> list[float]:
    values: list[float] = []
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    for filename in files:
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = dot_get(json.loads(line), field)
                if isinstance(value, bool):
                    values.append(float(value))
                else:
                    values.append(float(value))
    if not values:
        raise ValueError(f"metric source has no rows: {pattern}")
    return values


def metric_value(spec: Mapping[str, Any]) -> float:
    kind = str(spec["type"])
    path = str(spec["path"])
    field = str(spec["field"])
    if kind == "json":
        value = dot_get(json.loads(Path(path).read_text()), field)
        return float(value)
    values = _jsonl_values(path, field)
    if kind == "jsonl_mean":
        return sum(values) / len(values)
    if kind == "jsonl_sum":
        return sum(values)
    if kind == "jsonl_count":
        return float(len(values))
    if kind == "jsonl_max":
        return max(values)
    if kind == "jsonl_min":
        return min(values)
    raise ValueError(f"unknown metric type: {kind}")


def collect_metrics(stage: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: metric_value(spec)
        for name, spec in (stage.get("metrics") or {}).items()
    }


def gate_result(
    gate: Mapping[str, Any], metrics: Mapping[str, float]
) -> tuple[bool, dict[str, Any]]:
    lhs_name = str(gate["lhs"])
    lhs = float(metrics[lhs_name])
    if "rhs_metric" in gate:
        rhs = float(metrics[str(gate["rhs_metric"])])
    else:
        rhs = float(gate["rhs"])
    rhs += float(gate.get("margin", 0.0))
    op_name = str(gate["op"])
    if op_name not in OPS:
        raise ValueError(f"unknown gate operator: {op_name}")
    passed = bool(OPS[op_name](lhs, rhs))
    return passed, {
        "name": gate.get("name") or f"{lhs_name}{op_name}{rhs}",
        "lhs_metric": lhs_name,
        "lhs": lhs,
        "op": op_name,
        "rhs": rhs,
        "passed": passed,
    }


def evaluate_stage(stage: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    artifacts_ok, artifact_counts = required_artifacts(stage)
    if not artifacts_ok:
        return False, {"artifacts_ready": False, "artifact_counts": artifact_counts}
    metrics = collect_metrics(stage)
    gates = [gate_result(gate, metrics)[1] for gate in stage.get("gates") or []]
    return all(item["passed"] for item in gates), {
        "artifacts_ready": True,
        "artifact_counts": artifact_counts,
        "metrics": metrics,
        "gates": gates,
    }


def run_command(command: list[str], repo: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            [str(item) for item in command],
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(process.returncode)


def taiji_rows(client: Path, task: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [str(client), "instance_list", task],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        ids = re.findall(r"\b[0-9a-f]{32}\b", line, re.I)
        if not ids:
            continue
        fields = [field.strip() for field in line.split("|")[1:-1]]
        state = fields[3] if len(fields) >= 4 else "UNKNOWN"
        success = fields[2].lower() == "true" if len(fields) >= 4 else False
        rows.append({"instance": ids[0], "state": state, "success": success})
    return rows


def sync_code_mirror(repo: Path, mirror: Path, sha: str) -> None:
    """Materialize the frozen campaign commit into a dedicated Taiji-visible clone."""
    if mirror.resolve() == repo.resolve():
        if git_head(repo) != sha:
            raise ValueError("campaign repository moved away from frozen git head")
        return
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--no-checkout", str(repo), str(mirror)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to create code mirror: {result.stderr.strip()}")
    if not (mirror / ".git").exists():
        raise ValueError(f"code mirror is not a git clone: {mirror}")
    fetch = subprocess.run(
        ["git", "fetch", str(repo), sha],
        cwd=mirror, capture_output=True, text=True, check=False,
    )
    if fetch.returncode != 0:
        raise RuntimeError(f"failed to fetch frozen commit into code mirror: {fetch.stderr.strip()}")
    subprocess.run(["git", "reset", "--hard", sha], cwd=mirror, check=True)
    subprocess.run(["git", "clean", "-fd"], cwd=mirror, check=True)
    if git_head(mirror) != sha:
        raise RuntimeError("code mirror did not land on frozen campaign commit")


def submit_taiji(
    stage: Mapping[str, Any],
    campaign: Mapping[str, Any],
    record: dict[str, Any],
    repo: Path,
    workdir: Path,
) -> None:
    base_config = Path(str(stage["taiji_config"]))
    if not base_config.is_absolute():
        base_config = repo / base_config
    if campaign.get("code_mirror"):
        sync_code_mirror(repo, Path(str(campaign["code_mirror"])), git_head(repo))
    config = json.loads(base_config.read_text())
    config = replace_token(config, "__AUTORESEARCH_GIT_HEAD__", git_head(repo))
    attempt = int(record["attempt"]) + 1
    base_flag = str(config["task_flag"])
    task_flag = f"{base_flag}_ar{attempt}"
    config["task_flag"] = task_flag
    rendered = workdir / "taiji" / f"{stage['id']}.attempt{attempt}.json"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_text(json.dumps(config, indent=2) + "\n")
    donor = str(stage.get("donor_task") or campaign["taiji"]["donor_task"])
    client = str(campaign["taiji"]["client"])
    command = [
        sys.executable,
        str(repo / "scripts/submit_taiji_with_donor_init.py"),
        "submit",
        "--config",
        str(rendered),
        "--donor-task",
        donor,
        "--client",
        client,
    ]
    log_path = workdir / "logs" / f"{stage['id']}.submit{attempt}.log"
    code = run_command(command, repo, log_path)
    if code != 0:
        raise RuntimeError(f"Taiji submission failed; see {log_path}")
    record["attempt"] = attempt
    record["task_flag"] = task_flag
    record["instance"] = None
    record["state"] = "RUNNING"
    append_history(record, "taiji_submitted", task_flag=task_flag, attempt=attempt)


def poll_taiji(
    stage: Mapping[str, Any],
    campaign: Mapping[str, Any],
    record: dict[str, Any],
) -> str:
    client = Path(str(campaign["taiji"]["client"]))
    rows = taiji_rows(client, str(record["task_flag"]))
    if record.get("instance"):
        rows = [row for row in rows if row["instance"] == record["instance"]]
    if not rows:
        return "SUBMITTED"
    if len(rows) > 1:
        active = [row for row in rows if row["state"] != "END"]
        if len(active) != 1:
            raise RuntimeError(f"{stage['id']}: ambiguous Taiji instances")
        rows = active
    row = rows[0]
    record["instance"] = row["instance"]
    if row["state"] == "END":
        return "PLATFORM_SUCCESS" if row["success"] else "PLATFORM_FAILURE"
    return str(row["state"])


def finalize_scientific_stage(
    stage: Mapping[str, Any],
    record: dict[str, Any],
) -> None:
    passed, details = evaluate_stage(stage)
    record["evaluation"] = details
    record["state"] = "PASSED" if passed else "SCIENTIFIC_STOP"
    append_history(record, "scientific_gate", passed=passed, details=details)


def process_stage(
    stage: Mapping[str, Any],
    campaign: Mapping[str, Any],
    ledger: dict[str, Any],
    repo: Path,
    workdir: Path,
    *,
    dry_run: bool,
) -> bool:
    record = ledger["stages"][stage["id"]]
    if record["state"] in TERMINAL:
        return False
    ready, blocked_reason = dependency_state(stage, ledger)
    if blocked_reason:
        record["state"] = "BLOCKED"
        append_history(record, "blocked", reason=blocked_reason)
        return True
    if not ready:
        return False
    if not stage.get("enabled", True):
        record["state"] = "SKIPPED"
        append_history(record, "disabled_by_frozen_campaign")
        return True
    if stage.get("uses_test"):
        unlock = Path(str(stage["requires_test_unlock"]))
        if not unlock.is_file():
            return False
    if stage.get("requires_unlock"):
        unlock = Path(str(stage["requires_unlock"]))
        if not unlock.is_file():
            return False

    kind = str(stage["kind"])
    if kind == "artifact":
        artifacts_ok, _ = required_artifacts(stage)
        if not artifacts_ok:
            return False
        finalize_scientific_stage(stage, record)
        return True

    if kind == "local":
        if record["state"] == "PENDING":
            if dry_run:
                append_history(record, "dry_run_local", command=stage["command"])
                return False
            record["state"] = "RUNNING"
            record["attempt"] += 1
            append_history(record, "local_started", attempt=record["attempt"])
            log_path = workdir / "logs" / f"{stage['id']}.log"
            code = run_command(list(stage["command"]), repo, log_path)
            if code != 0:
                record["state"] = "INFRA_FAILED"
                append_history(record, "local_failed", returncode=code, log=str(log_path))
                return True
            finalize_scientific_stage(stage, record)
            return True

    if kind == "taiji":
        if record["state"] == "PENDING":
            if dry_run:
                append_history(record, "dry_run_taiji", config=stage["taiji_config"])
                return False
            submit_taiji(stage, campaign, record, repo, workdir)
            return True
        platform = poll_taiji(stage, campaign, record)
        record["platform_state"] = platform
        append_history(record, "taiji_poll", platform_state=platform)
        if platform == "PLATFORM_SUCCESS":
            artifacts_ok, counts = required_artifacts(stage)
            if not artifacts_ok:
                record["state"] = "INFRA_FAILED"
                append_history(record, "platform_success_missing_artifacts", counts=counts)
            else:
                finalize_scientific_stage(stage, record)
            return True
        if platform == "PLATFORM_FAILURE":
            retries = int(stage.get("max_infra_retries", 0))
            if int(record["attempt"]) <= retries:
                record["state"] = "PENDING"
                append_history(record, "retry_scheduled", attempts=record["attempt"])
            else:
                record["state"] = "INFRA_FAILED"
                append_history(record, "taiji_failed", attempts=record["attempt"])
            return True
        return False

    raise ValueError(f"{stage['id']}: unknown stage kind {kind}")


def reconcile_blocked(campaign: Mapping[str, Any], ledger: dict[str, Any]) -> bool:
    changed = False
    for stage in campaign["stages"]:
        record = ledger["stages"][stage["id"]]
        if record["state"] != "PENDING":
            continue
        _, blocked = dependency_state(stage, ledger)
        if blocked:
            record["state"] = "BLOCKED"
            append_history(record, "blocked", reason=blocked)
            changed = True
    return changed


def summary(campaign: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    stages = []
    for stage in campaign["stages"]:
        record = ledger["stages"][stage["id"]]
        stages.append(
            {
                "id": stage["id"],
                "state": record["state"],
                "attempt": record.get("attempt", 0),
                "metrics": (record.get("evaluation") or {}).get("metrics"),
            }
        )
    return {
        "campaign": campaign["campaign"],
        "git_head": ledger["git_head"],
        "stages": stages,
        "complete": all(item["state"] in TERMINAL for item in stages),
        "all_passed_or_skipped": all(
            item["state"] in {"PASSED", "SKIPPED", "BLOCKED"} for item in stages
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=["plan", "step", "run", "status"], default="step")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    campaign, digest = load_campaign(args.campaign)
    repo = args.repo.resolve()
    ledger = load_or_create_ledger(args.ledger, campaign, digest, repo)
    stages = stage_map(campaign)

    if args.mode in {"plan", "status"}:
        print(json.dumps(summary(campaign, ledger), indent=2))
        return 0

    interval = max(int(campaign.get("poll_seconds", 300)), 30)
    while True:
        changed = False
        for stage in campaign["stages"]:
            changed |= process_stage(
                stages[stage["id"]], campaign, ledger, repo, args.workdir,
                dry_run=args.dry_run,
            )
            atomic_json(args.ledger, ledger)
        changed |= reconcile_blocked(campaign, ledger)
        atomic_json(args.ledger, ledger)
        current = summary(campaign, ledger)
        atomic_json(args.workdir / "campaign_summary.json", current)
        print(json.dumps(current, indent=2), flush=True)
        if args.mode == "step" or current["complete"] or args.dry_run:
            return 0
        if not changed:
            time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
