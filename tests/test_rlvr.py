"""Tests for Self-MechVR RLVR helpers."""

from __future__ import annotations

import json
from pathlib import Path

from mechet.rlvr import (
    compute_advantages,
    compute_mechvr_reward,
    grpo_advantages,
    mechvr_gate,
    rloo_advantages,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data/samples/valid_mini.jsonl"


def test_mechvr_gate():
    assert mechvr_gate({"format_ok": True, "reachability_ok": True, "electron_conserved": True})
    assert not mechvr_gate({"format_ok": True, "reachability_ok": False, "electron_conserved": True})


def test_grpo_advantages_zero_mean():
    adv = grpo_advantages([1.0, 2.0, 3.0, 4.0])
    assert abs(sum(adv)) < 1e-5


def test_rloo_advantages():
    adv = rloo_advantages([1.0, 3.0])
    assert adv[1] > adv[0]
    assert abs(sum(adv)) < 1e-5


def test_compute_advantages_routing():
    assert len(compute_advantages([1.0, 2.0], method="grpo")) == 2
    assert len(compute_advantages([1.0, 2.0], method="rloo")) == 2


def test_mechvr_reward_on_gold_sample():
    row = json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])
    assistant = row["messages"][-1]["content"]
    scored = compute_mechvr_reward(row, assistant)
    assert scored["gate_ok"]
    assert scored["rlvr_total"] > 0
