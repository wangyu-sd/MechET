import json

import pytest

from scripts.eval_jev_style_successor import (
    ROOT,
    select_counts,
    verify_checkpoint_contract,
)
from scripts.train_system_one_electron_flow import file_sha256


def test_typed_successor_backoff_does_not_read_gold_flow_count():
    executable_two = {1: {"ok": True}, 2: {"ok": True}}
    invalid_two = {1: {"ok": True}, 2: {"ok": False}}
    for gold_count in (1, 2, 3):
        assert select_counts(gold_count, executable_two)["validity_backoff_2_to_1"] == 2
        assert select_counts(gold_count, invalid_two)["validity_backoff_2_to_1"] == 1
        assert select_counts(gold_count, executable_two)["oracle_count"] == gold_count


def test_typed_successor_requires_matching_preflight_code_and_source(tmp_path):
    manifest = {
        "train_source": {"sha256": "train"},
        "valid_source": {"sha256": "valid"},
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": "pinned",
        "input_contract": "shared_state_typed_source_and_atom_sink_composed_pairs_block_causal_v2",
        "selected_train_events": 19199,
        "selected_valid_events": 2543,
    }
    preflight = {
        **manifest,
        "artifact_type": "system_one_jev_typed_v2_factorized_preflight",
        "overlength_count": 0,
        "max_length": 8192,
        "max_observed_length": 2927,
        "trainer_sha256": file_sha256(ROOT / "scripts/train_jev_style_electron_flow.py"),
        "encoder_sha256": file_sha256(ROOT / "src/mechet/jev_style_decision.py"),
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(preflight))
    assert verify_checkpoint_contract(tmp_path, manifest) == preflight
    preflight["encoder_sha256"] = "old encoder"
    path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="encoder_sha256"):
        verify_checkpoint_contract(tmp_path, manifest)
    preflight["encoder_sha256"] = file_sha256(ROOT / "src/mechet/jev_style_decision.py")
    preflight["valid_source"] = {"sha256": "other"}
    path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="valid_source"):
        verify_checkpoint_contract(tmp_path, manifest)
    preflight["valid_source"] = manifest["valid_source"]
    preflight["max_length"] = 1024
    path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="input-length"):
        verify_checkpoint_contract(tmp_path, manifest)
