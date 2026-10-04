"""Refuse superseded artifacts without mutating frozen data manifests."""
from __future__ import annotations

import json
from pathlib import Path


def require_artifact_status(manifest_path: Path, *, operation: str) -> None:
    flags = {
        "training": "training_allowed",
        "headline_evaluation": "headline_evaluation_allowed",
    }
    if operation not in flags:
        raise ValueError(f"unknown artifact operation: {operation}")
    status_path = manifest_path.parent / "ARTIFACT_STATUS.json"
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get(flags[operation]) is False:
        raise SystemExit(f"artifact forbids {operation}: {status_path}")
