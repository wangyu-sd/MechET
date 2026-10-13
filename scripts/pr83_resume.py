"""Crash-safe, provenance-locked resume helpers for PR83 evaluation.

No RDKit/GPU imports; safe to exercise in lightweight CI. Never resumes a
different sample, checkpoint, code version or policy configuration.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_MANIFEST_NAME = "run_manifest.json"
RESULTS_NAME = "results.shard-00-of-01.jsonl"


class AuditJournal:
    """Append-only JSONL with SHA-authenticated checkpoint cursors.

    Checkpoints retain offsets/hashes, not all historical trajectories. After
    a crash, evidence written after the saved model cursor is quarantined before
    truncation; altered saved evidence fails closed without modifying the file.
    """
    def __init__(self, path: Path, *, restored_state=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.digest = hashlib.sha256()
        if restored_state is None:
            self.handle = self.path.open("xb")
            return
        offset = restored_state["offset"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("invalid audit cursor")
        if not self.path.exists():
            if offset:
                raise ValueError("audit prefix is missing")
            self.path.touch(exist_ok=False)
        if self.path.stat().st_size < offset:
            raise ValueError("audit prefix is shorter than saved cursor")
        with self.path.open("rb") as source:
            remaining = offset
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("audit prefix is incomplete")
                self.digest.update(block)
                remaining -= len(block)
        if self.digest.hexdigest() != restored_state["sha256"]:
            raise ValueError("audit prefix hash differs from checkpoint")
        if self.path.stat().st_size > offset:
            import shutil
            import time
            evidence = self.path.with_name(f"{self.path.stem}.rollback-{time.time_ns()}.jsonl")
            with self.path.open("rb") as source, evidence.open("xb") as sink:
                source.seek(offset)
                shutil.copyfileobj(source, sink)
                sink.flush()
                os.fsync(sink.fileno())
            with self.path.open("r+b") as handle:
                handle.truncate(offset)
        self.handle = self.path.open("ab")

    def append(self, record):
        raw = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        self.handle.write(raw)
        self.handle.flush()
        self.digest.update(raw)

    def checkpoint_state(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return {"offset": self.handle.tell(), "sha256": self.digest.hexdigest()}

    def close(self):
        self.handle.close()


def _bytes_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    staging = path.with_name(path.name + ".tmp")
    with staging.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def prepare_resume(
    output: Path,
    *,
    manifest: Mapping[str, Any],
    selected_ids: Sequence[str],
    resume: bool = False,
    manifest_name: str = RUN_MANIFEST_NAME,
    results_name: str = RESULTS_NAME,
) -> list[dict[str, Any]]:
    """Return validated complete saved rows, or fail closed on provenance drift.

    Repair is deliberately limited to an interrupted final JSONL append.
    Its raw trailing bytes are retained for independent forensic audit.
    """
    output = Path(output)
    ids = tuple(map(str, selected_ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("selected evaluation IDs must be unique and nonempty")
    if list(manifest.get("selected_ids", [])) != list(ids):
        raise ValueError("manifest/selection ID mismatch")

    if any(Path(name).name != name for name in (manifest_name, results_name)):
        raise ValueError("resume filenames must be basenames")
    manifest_file = output / manifest_name
    results_file = output / results_name
    existing = manifest_file.exists() or results_file.exists()
    if manifest_name == RUN_MANIFEST_NAME and results_name == RESULTS_NAME:
        existing = output.is_dir() and any(output.iterdir())
    if existing:
        if not resume:
            raise FileExistsError(
                f"{output} already has output; use --resume with the exact frozen run"
            )
        if not manifest_file.is_file():
            raise ValueError("cannot resume: original run manifest missing")
        previous = manifest_file.read_bytes()
        if previous != _bytes_json(manifest):
            raise ValueError("cannot resume: source, checkpoint, config, code, or IDs changed")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_atomic(manifest_file, _bytes_json(manifest))
    if not results_file.exists():
        return []
    raw = results_file.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not resume:
            raise ValueError("incomplete final JSONL record; resume is required")
        last_complete = raw.rfind(b"\n") + 1
        tail = raw[last_complete:]
        if tail:
            digest = hashlib.sha256(tail).hexdigest()[:16]
            prefix = "results" if results_name == RESULTS_NAME else results_name
            evidence = output / f"{prefix}.partial-tail-{digest}.bin"
            if not evidence.exists():
                _write_atomic(evidence, tail)
            # Preserve all valid complete rows; never discard an intact record.
            _write_atomic(output / (results_name + ".repaired"), raw[:last_complete])
            os.replace(output / (results_name + ".repaired"), results_file)
            raw = raw[:last_complete]

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        try:
            item = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt complete evaluation line {line_no}: {exc}") from exc
        if not isinstance(item, dict) or line_no > len(ids):
            raise ValueError("evaluation output contains unexpected rows")
        if str(item.get("id", "")) != ids[line_no - 1]:
            raise ValueError(f"evaluation ID mismatch at row {line_no}")
        if (
            item.get("policy") != manifest["policy"]
            or item.get("graph_checkpoint_sha256") != manifest["graph_sha256"]
            or item.get("qwen_adapter_sha256") != manifest["qwen_adapter_sha256"]
        ):
            raise ValueError(f"evaluation row provenance mismatch at row {line_no}")
        rows.append(item)
    return rows


def evaluation_manifest(
    *,
    selected_ids: Sequence[str],
    source_sha256: str,
    source_path: str,
    qwen_adapter_sha256: str,
    graph_sha256: str,
    graph_rl_head_sha256: str | None,
    model: str,
    model_revision: str,
    config: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    policy: str = "crn_jepa_qwen_routed_flow_v1",
) -> dict[str, Any]:
    return {
        "schema": 1,
        "policy": policy,
        "selected_ids": list(selected_ids),
        "source_path": str(Path(source_path).resolve()),
        "source_sha256": source_sha256,
        "qwen_adapter_sha256": qwen_adapter_sha256,
        "graph_sha256": graph_sha256,
        "graph_rl_head_sha256": graph_rl_head_sha256,
        "model": model,
        "model_revision": model_revision,
        "config": dict(config),
        "code_sha256": dict(code_sha256),
    }


def prepare_training_resume(
    output: Path, *, manifest: Mapping[str, Any], resume: bool = False,
) -> Path | None:
    """Bind a multi-update RL attempt to immutable training provenance.

    Returns the last checkpoint path on a same-contract resume, or None for a
    fresh attempt. Never silently adopts results from another source/config.
    """
    output = Path(output)
    path = output / "train_manifest.json"
    state = output / "latest_train_state.pt"
    existing = output.is_dir() and any(output.iterdir())
    if existing:
        if not resume:
            raise FileExistsError("training output already contains another attempt")
        if not path.is_file():
            raise ValueError("training resume manifest is missing")
        if path.read_bytes() != _bytes_json(manifest):
            raise ValueError("training resume source, code or frozen config changed")
        if not state.is_file():
            raise ValueError("training checkpoint missing; cannot resume an uncommitted update")
        return state
    if resume:
        raise ValueError("resume requested but no previous training output exists")
    output.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, _bytes_json(manifest))
    return None
