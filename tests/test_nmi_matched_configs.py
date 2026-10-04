import json
from pathlib import Path

import pytest
import yaml

from scripts.create_nmi_matched_sft_configs import create_configs


BASE = Path(__file__).resolve().parents[1] / "configs/iclr/a4_open_flow_sft.yaml"


def _matched_fixture(tmp_path):
    matched = tmp_path / "matched"
    for condition in ("direct", "open_flow", "closed_loop"):
        path = matched / condition
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({
            "condition": condition,
            "parent_split_manifest_sha256": "same",
            "stable_reaction_ids": {"train": "a", "valid": "b", "test": "c"},
            "rows": {"train": 65, "valid": 2, "test": 3},
            "training_allowed": True,
        }))
        for split in ("train", "valid", "test"):
            (path / f"{split}.jsonl").write_text("{}\n")
    (matched / "verification.json").write_text(json.dumps({
        "passed": True,
        "parent_split_manifest_sha256": "same",
        "stable_reaction_ids": {"train": "a", "valid": "b", "test": "c"},
    }))
    return matched


def test_equal_update_configs_have_identical_core_hyperparameters(tmp_path):
    matched = _matched_fixture(tmp_path)
    output = tmp_path / "configs"
    report = create_configs(matched, output, BASE,
                            run_output_root=tmp_path / "runs", epochs_reference=3)
    assert report["equal_optimizer_updates"] == 6  # ceil(65 / 64) * 3
    configs = {condition: yaml.safe_load((output / f"{condition}.yaml").read_text())
               for condition in ("direct", "open_flow", "closed_loop")}
    assert len({json.dumps(config["training"], sort_keys=True) for config in configs.values()}) == 1
    assert len({json.dumps(config["lora"], sort_keys=True) for config in configs.values()}) == 1
    assert len({config["contract"]["h2_parent_split_manifest_sha256"] for config in configs.values()}) == 1
    assert configs["closed_loop"]["contract"]["require_trace_owned"] is True
    assert configs["direct"]["contract"]["require_trace_owned"] is False
    assert configs["closed_loop"]["environment"]["observation_mode"] == "compact_full_state"


def test_config_generator_rejects_id_drift(tmp_path):
    matched = _matched_fixture(tmp_path)
    manifest = matched / "closed_loop" / "manifest.json"
    record = json.loads(manifest.read_text())
    record["stable_reaction_ids"]["test"] = "changed"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="do not share frozen IDs"):
        create_configs(matched, tmp_path / "configs", BASE,
                       run_output_root=tmp_path / "runs")
