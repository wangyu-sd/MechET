"""Tests for MechET eval metrics."""

from __future__ import annotations

import json
from pathlib import Path

from mechet.metrics import (
    aggregate_rates,
    canonical_species,
    score_mech_et_prediction,
    top1_strict_match,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data/samples/valid_mini.jsonl"


def test_canonical_species_order_independent():
    a = canonical_species("CCO.CC")
    b = canonical_species("CC.CCO")
    assert a == b


def test_top1_strict_match():
    assert top1_strict_match("CCO", "CCO")
    assert not top1_strict_match("CCO", "CC")


def test_gold_audit_on_mini_sample():
    row = json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])
    assistant = row["messages"][-1]["content"]
    case = score_mech_et_prediction(row, assistant, mode="gold_audit")
    assert case["format_ok"]
    assert case["answer_exact"]


def test_aggregate_rates_keys():
    agg = aggregate_rates([{"format_ok": True, "top1_strict": False}])
    assert "rates" in agg
    assert agg["rates"]["format_ok_rate"] == 1.0
