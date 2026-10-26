"""PR96 submission payload regression for the real -1100408 region failure."""
import json
from pathlib import Path


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "taiji"
PREFIX = "meteor_mechet_pr96_flower_qwen35_08b_2stage_8h20_zjk_20261023_"


def latest_payload():
    candidates = sorted(CONFIGS.glob(f"{PREFIX}*.json"))
    assert candidates, "Missing PR96 full FlowER submission payload"
    return json.loads(candidates[-1].read_text())


def test_latest_h20_payload_uses_proven_application_group_region_binding():
    """`zjk` is rejected before allocation; live tasks in this group use ''."""
    payload = latest_payload()
    assert payload["business_flag"] == "AI4LSLab_KEMO_ZJK_H20"
    assert payload["location"] == "", (
        "Use the proven application-group binding, not unsupported region alias zjk"
    )
    assert payload["GPUName"] == "H20"
    assert payload["host_num"] == 1 and payload["host_gpu_num"] == 8
    assert payload["is_elasticity"] is False


def test_region_replacement_preserves_full_training_and_deployment_payload():
    original = json.loads((CONFIGS / f"{PREFIX}01.json").read_text())
    # This specifically tests the routing-only _03 repair. Later _04 is the
    # explicitly authorized performance/runtime upgrade, not a region retry.
    replacement = json.loads((CONFIGS / f"{PREFIX}03.json").read_text())
    assert replacement["task_flag"] != original["task_flag"], (
        "Preserve the failed attempt; submit its replacement under a fresh flag"
    )
    changes = {"task_flag", "task_description", "location", "quota_type"}
    assert {k: v for k, v in replacement.items() if k not in changes} == {
        k: v for k, v in original.items() if k not in changes
    }


def test_latest_h20_payload_uses_business_quota_not_zero_public_quota():
    """H20 public quota is zero; successful same-group tasks use group defaults."""
    assert latest_payload()["quota_type"] == "", (
        "Do not copy A100's public pool override into the H20 business quota"
    )


def test_performance_replacement_retains_proven_image_ceph_and_default_logs():
    payload = latest_payload()
    assert payload["task_flag"].endswith("_04")
    assert payload["readable_name"].startswith("meteor ")
    assert payload["image_full_name"] == "mirrors.tencent.com/whaleywang/metabo:taiji6"
    assert payload["init_cmd"] == "REPLACE_WITH_PRIVATE_INIT_CMD_FROM_SUCCESSFUL_TASK"
    assert payload["is_mount_ceph"] is True
    assert "TAIJI_HEARTBEAT_SECONDS=45" in payload["start_cmd"]
    assert "TAIJI_MIRROR_PID1_STDOUT=1" in payload["start_cmd"]
    assert "scripts/taiji_run_with_heartbeat.sh" in payload["start_cmd"]
    assert "pr96_code_c42c94b_performance_20261023_01.tar.gz" in payload["start_cmd"]
    assert "pr96_runtime_tf550_causalconv170_Tr320_py311_20261023_01.tar.gz" in payload["start_cmd"]
    assert "/performance_v1/plan.json" in payload["start_cmd"]
    import re
    assert not re.search(r"(?:^|[;&|\s])(?:[12]?>>?|&>)", payload["start_cmd"])
