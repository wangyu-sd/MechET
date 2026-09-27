"""Legacy EARHO task labels must remain complete and unambiguous."""

import json
from pathlib import Path

from scripts.run_earho_v2 import load_earho_config


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/EARHO_HISTORICAL_RUN_REGISTRY.json"


def test_all_existing_earho_task_specs_are_registered():
    registry = json.loads(REGISTRY.read_text())
    rows = registry["task_configs"]
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "configs/taiji").glob("meteor_mechet_earho_*.json")
    }
    assert {row["path"] for row in rows} == expected
    assert len(rows) == len(expected)
    assert {row["classification"] for row in rows} == {
        "legacy_pre_v3", "confirmed_actor_prefix_mismatch"
    }
    for row in rows:
        spec = json.loads((ROOT / row["path"]).read_text())
        assert spec["task_flag"] == row["task_flag"]
    mismatches = [
        row for row in rows
        if row["classification"] == "confirmed_actor_prefix_mismatch"
    ]
    assert len(mismatches) == 1
    assert "prefixv2" in mismatches[0]["task_flag"]


def test_historical_artifacts_and_new_configs_are_separate():
    registry = json.loads(REGISTRY.read_text())
    artifacts = registry["artifacts"]
    legacy_paths = {row["path"] for row in artifacts}
    assert len(legacy_paths) == len(artifacts)
    assert sum(
        row["classification"] == "confirmed_actor_prefix_mismatch"
        for row in artifacts
    ) == 1
    for name in (
        "earho_paper_flower_strict_prefixv3_8h20.yaml",
        "earho_paper_flower_strict_k2_prefixv3_8a100.yaml",
        "earho_paper_flower_strict_k2_gt_smoke_prefixv3_8a100.yaml",
        "earho_paper_mech_uspto31k_prefixv3_8a100.yaml",
        "earho_paper_mech_uspto31k_prefixv3_8h20.yaml",
    ):
        cfg = load_earho_config(ROOT / "configs/agent" / name)
        assert cfg["prompt_prefix_contract"] == "qwen_sft_tool_and_text_prefix_v3"
        assert "prefixv3" in cfg["output_dir"]
        assert not any(cfg["output_dir"].endswith(path) for path in legacy_paths)


def test_local_historical_sidecars_match_registry_when_available():
    registry = json.loads(REGISTRY.read_text())
    shared_root = Path("/aaa/fionafyang/buddy1/whaleywang/MechET")
    for row in registry["artifacts"]:
        directory = shared_root / row["path"]
        if not directory.is_dir():
            continue
        marker = json.loads((directory / "HISTORICAL_STATUS.json").read_text())
        assert marker["status"] == row["classification"]
        assert marker["observed_prompt_prefix_contract"] == row[
            "observed_prompt_prefix_contract"
        ]
        assert marker["do_not_resume_as_prefixv3"]
        assert marker["do_not_report_as_prefixv3"]
