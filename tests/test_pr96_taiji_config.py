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
    replacement = latest_payload()
    assert replacement["task_flag"] != original["task_flag"], (
        "Preserve the failed attempt; submit its replacement under a fresh flag"
    )
    changes = {"task_flag", "task_description", "location"}
    assert {k: v for k, v in replacement.items() if k not in changes} == {
        k: v for k, v in original.items() if k not in changes
    }
