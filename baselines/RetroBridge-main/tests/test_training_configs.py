from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_full_training_configs_use_eight_gpu_global_batch_64():
    for filename in (
        "mechet_retrobridge_flower_full.yaml",
        "mechet_retrobridge_mech_uspto_31k_full.yaml",
    ):
        config = yaml.safe_load((ROOT / "configs" / filename).read_text())
        assert config["devices"] == 8
        assert config["strategy"] == "ddp"
        assert config["precision"] == "bf16-mixed"
        assert (
            config["devices"]
            * config["batch_size"]
            * config["accumulate_grad_batches"]
            == 64
        )
        assert config["resume"] is None
        assert config["sample_every_val"] == 0
        assert config["early_stopping_patience"] > 0
