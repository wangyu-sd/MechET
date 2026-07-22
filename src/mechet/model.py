"""Resolve local Qwen model path."""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_qwen_model_path() -> str | None:
    env = os.environ.get("QWEN_MODEL_PATH") or os.environ.get("MECHET_MODEL_PATH")
    if env and Path(env).exists():
        return env
    search_roots = [
        Path(os.path.expanduser("~/models")),
        Path("/aaa/fionafyang/buddy1/whaleywang/models/orbit_qwen_base"),
        Path("/aaa/fionafyang/buddy1/whaleywang/models/Qwen3-8B"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        candidates = [root]
        if root.is_dir():
            candidates.extend(child for child in root.iterdir() if child.is_dir())
        for candidate in candidates:
            if not isinstance(candidate, Path):
                continue
            cfg = candidate / "config.json"
            if cfg.exists():
                try:
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    if "qwen" in json.dumps(data).lower():
                        return str(candidate)
                except Exception:
                    continue
    return None
