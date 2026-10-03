import json
from pathlib import Path

import pytest

from scripts.nmi_artifact_status import require_artifact_status


def test_absent_status_preserves_existing_manifest_contract(tmp_path: Path):
    require_artifact_status(tmp_path / "manifest.json", operation="training")
    require_artifact_status(tmp_path / "manifest.json", operation="headline_evaluation")


def test_superseded_open_flow_sidecar_blocks_both_operations(tmp_path: Path):
    (tmp_path / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "training_allowed": False,
        "headline_evaluation_allowed": False,
    }))
    for operation in ("training", "headline_evaluation"):
        with pytest.raises(SystemExit, match=f"forbids {operation}"):
            require_artifact_status(tmp_path / "manifest.json", operation=operation)


def test_status_can_forbid_training_without_forbidding_evaluation(tmp_path: Path):
    (tmp_path / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "training_allowed": False,
        "headline_evaluation_allowed": True,
    }))
    require_artifact_status(tmp_path / "manifest.json", operation="headline_evaluation")
    with pytest.raises(SystemExit, match="forbids training"):
        require_artifact_status(tmp_path / "manifest.json", operation="training")
