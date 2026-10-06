from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_reliable_mechet_three_stage_lineage_is_explicit():
    umbrella = load_yaml("configs/experiments/reliable_mechet_three_stage_v1.yaml")
    stage1 = load_yaml("configs/agent/natural_language_event_v2_qwen3_0_6b.yaml")
    stage2 = load_yaml("configs/agent/natural_language_history_v2_qwen3_0_6b.yaml")
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")

    expected_model = "Qwen/Qwen3-0.6B"
    expected_revision = "c1899de289a04d12100db370d81485cdf75e47ca"

    assert umbrella["model_policy"]["primary"]["model"] == expected_model
    assert umbrella["model_policy"]["primary"]["revision"] == expected_revision

    assert stage1["model_name_or_path"] == expected_model
    assert stage2["model_name_or_path"] == expected_model
    assert stage3["model_name_or_path"] == expected_model
    assert stage1["training"]["model_revision"] == expected_revision
    assert stage2["training"]["model_revision"] == expected_revision
    assert stage3["model_revision"] == expected_revision

    assert stage2["initial_adapter_path"] == stage1["output_dir"]
    assert stage3["initial_adapter_path"] == stage2["output_dir"]
    assert stage3["initial_adapter_model_sha256"] == "REPLACE_WITH_FROZEN_STAGE_II_SHA256"


def test_three_stages_keep_reverse_et_as_the_main_prediction_path():
    umbrella = load_yaml("configs/experiments/reliable_mechet_three_stage_v1.yaml")
    stage1 = load_yaml("configs/agent/natural_language_event_v2_qwen3_0_6b.yaml")
    stage2 = load_yaml("configs/agent/natural_language_history_v2_qwen3_0_6b.yaml")

    assert umbrella["stages"]["stage1_state_sft"]["output_decisions"] == [
        "import_fragment",
        "reverse_electron_flow",
        "finish",
    ]
    assert stage1["contract"]["electron_flow_model_generated"] is True
    assert stage2["contract"]["electron_flow_model_generated"] is True
    assert stage1["contract"]["terminal_tool"] == "finish_trace"
    assert stage2["contract"]["terminal_tool"] == "finish_trace"


def test_endpoint_and_process_denominators_stay_separate():
    umbrella = load_yaml("configs/experiments/reliable_mechet_three_stage_v1.yaml")
    assert umbrella["data"]["headline_endpoint_denominator"] == 28971
    assert umbrella["data"]["process_analysis_denominator"] == 28967
