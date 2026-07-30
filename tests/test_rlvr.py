"""Tests for Self-MechVR RLVR helpers and adversarial process checks."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import torch

from mechet.rlvr import (
    compute_advantages,
    compute_mechvr_reward,
    grpo_advantages,
    mechvr_gate,
    normalize_sequence_log_prob,
    rloo_advantages,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data/samples/valid_mini.jsonl"


def _sample() -> tuple[dict, str]:
    row = json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])
    return row, str(row["messages"][-1]["content"])


def test_mechvr_gate_requires_executable_process():
    valid = {
        "format_ok": True,
        "target_state_matches_product": True,
        "reachability_ok": True,
        "state_maps_consistent": True,
        "local_transition_exact": True,
        "electron_conserved": True,
        "answer_state_agree": True,
    }
    assert mechvr_gate(valid)
    for key in valid:
        corrupted = dict(valid)
        corrupted[key] = False
        assert not mechvr_gate(corrupted)


def test_grpo_advantages_zero_mean():
    advantages = grpo_advantages([1.0, 2.0, 3.0, 4.0])
    assert abs(sum(advantages)) < 1e-5


def test_rloo_advantages():
    advantages = rloo_advantages([1.0, 3.0])
    assert advantages[1] > advantages[0]
    assert abs(sum(advantages)) < 1e-5


def test_compute_advantages_routing():
    assert len(compute_advantages([1.0, 2.0], method="grpo")) == 2
    assert len(compute_advantages([1.0, 2.0], method="rloo")) == 2


def test_length_normalized_log_prob():
    value = torch.tensor(-20.0)
    assert normalize_sequence_log_prob(value, 10).item() == -2.0
    assert normalize_sequence_log_prob(value, 10, length_normalize=False).item() == -20.0


def test_mechvr_reward_on_gold_sample():
    row, assistant = _sample()
    scored = compute_mechvr_reward(row, assistant)
    assert scored["gate_ok"]
    assert scored["verified"]["local_transition_exact"]
    assert scored["verified"]["answer_state_agree"]
    assert scored["rlvr_total"] > 0


def test_tampered_be_delta_is_rejected():
    row, assistant = _sample()
    tampered = assistant.replace("BOND 32 33 +1", "BOND 32 33 +2", 1)
    assert tampered != assistant
    scored = compute_mechvr_reward(row, tampered)
    assert not scored["gate_ok"]
    assert not scored["verified"]["local_transition_exact"]
    assert scored["failure_stage"] == "be_delta"


def test_answer_must_agree_with_precursor_state():
    row, assistant = _sample()
    tampered = re.sub(
        r"<answer>.*?</answer>",
        "<answer>\n[He:999]\n</answer>",
        assistant,
        count=1,
        flags=re.DOTALL,
    )
    scored = compute_mechvr_reward(row, tampered)
    assert not scored["gate_ok"]
    assert not scored["verified"]["answer_state_agree"]
    assert scored["failure_stage"] == "state_agree"


def test_gold_answer_is_outcome_reward_not_process_gate():
    row, assistant = _sample()
    wrong_gold = copy.deepcopy(row)
    wrong_gold.setdefault("metadata", {})["initial_reactants"] = "[He:999]"
    original = compute_mechvr_reward(row, assistant)
    scored = compute_mechvr_reward(wrong_gold, assistant)
    assert scored["gate_ok"]
    assert not scored["verified"]["answer_gold_exact"]
    assert scored["rlvr_total"] < original["rlvr_total"]
