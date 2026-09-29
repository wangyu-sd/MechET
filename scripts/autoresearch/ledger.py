"""Append-only logical campaign ledger with immutable config identity."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from scripts.autoresearch.stratified_manifest import digest


def append(
    output: Path, *, campaign_id: str, config_path: Path, repo: Path,
    action: str, stage: str, evidence: dict[str, Any],
) -> dict[str, Any]:
    path = output / "campaign_state.json"
    config_hash = digest(config_path)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if path.exists():
        state = json.loads(path.read_text())
        if state["campaign_id"] != campaign_id or state["config_sha256"] != config_hash:
            raise ValueError("campaign/config identity drift; create a new campaign revision")
    else:
        state = {"artifact_type": "autoresearch_campaign_state_v1",
                 "campaign_id": campaign_id, "config_sha256": config_hash,
                 "created_at": datetime.now(timezone.utc).isoformat(), "events": []}
    event = {"event_index": len(state["events"]), "timestamp": datetime.now(timezone.utc).isoformat(),
             "git_commit": commit, "action": action, "stage": stage, "evidence": evidence}
    if state["events"] and state["events"][-1]["action"] == action and \
            state["events"][-1]["evidence"] == evidence:
        return state
    state["events"].append(event)
    output.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return state
