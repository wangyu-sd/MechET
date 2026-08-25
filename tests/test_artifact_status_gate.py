import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_tool_sft.py"
SPEC = importlib.util.spec_from_file_location("mechet_train_tool_sft_status", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
enforce_artifact_status = MODULE.enforce_artifact_status


def _artifact(tmp_path: Path, *, training_allowed: bool) -> Path:
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    (tmp_path / "ARTIFACT_STATUS.json").write_text(
        json.dumps(
            {
                "artifact_id": "test_artifact",
                "status": "validated" if training_allowed else "deprecated_pilot",
                "training_allowed": training_allowed,
                "reason": "test status",
            }
        ),
        encoding="utf-8",
    )
    return train


def test_artifact_status_allows_validated_data(tmp_path: Path) -> None:
    train = _artifact(tmp_path, training_allowed=True)
    status = enforce_artifact_status(train, {})
    assert status is not None
    assert status["artifact_id"] == "test_artifact"


def test_artifact_status_rejects_deprecated_data(tmp_path: Path) -> None:
    train = _artifact(tmp_path, training_allowed=False)
    with pytest.raises(ValueError, match="ARTIFACT_TRAINING_FORBIDDEN"):
        enforce_artifact_status(train, {})


def test_artifact_status_requires_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _artifact(tmp_path, training_allowed=False)
    monkeypatch.setenv("MECHET_ALLOW_DEPRECATED_ARTIFACT", "1")
    assert enforce_artifact_status(train, {}) is not None
