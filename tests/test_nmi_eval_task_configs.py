import json
from pathlib import Path

from scripts.submit_taiji_with_donor_init import validate_submission_policy


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "direct": ROOT / "configs/taiji/meteor_mechet_nmi_h2_direct_k10_8a100_qy_20261003.json",
    "open_flow": ROOT / "configs/taiji/meteor_mechet_nmi_h2_open_flow_k10_8a100_qy_20261003.json",
    "closed_loop": ROOT / "configs/taiji/meteor_mechet_nmi_h2_closed_loop_k10_8h20_zjk_20261003.json",
}


def test_h2_evaluation_tasks_are_frozen_and_not_prematurely_submitted():
    for condition, path in CONFIGS.items():
        cfg = json.loads(path.read_text())
        validate_submission_policy(cfg, path)
        command = cfg["start_cmd"]
        assert cfg["task_flag"].startswith("meteor_")
        assert cfg["task_type"] == "general_gpu_type"
        assert cfg["task_category"] == "fine_tuning"
        assert cfg["is_elasticity"] is False
        assert cfg["init_cmd"] == "REPLACE_WITH_PRIVATE_INIT_CMD_FROM_SUCCESSFUL_TASK"
        assert "SAMPLES_PER_TARGET=10" in command
        assert "MECHET_EXPECTED_ROWS=27104" in command
        assert "MECHET_EVALUATION_SCOPE=nmi_h2" in command
        assert f"MECHET_CONDITION_NAME=nmi_h2_{condition}_seed17_k10" in command
        assert f"/outputs/issue79/h2_{condition}_seed17" in command
        assert f"/nmi_matched_v1/{condition}/test.jsonl" in command
        if condition == "closed_loop":
            assert "MECHET_MAX_ITERATIONS=40" in command
            assert "run_taiji_flower_a7_inference.sh" in command
        else:
            assert f"MECHET_MANIFEST_TASK={condition}" in command
            assert "run_taiji_iclr_full_inference.sh" in command
