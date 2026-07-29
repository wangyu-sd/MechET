"""Tests for MechET eval metrics."""

from __future__ import annotations

import json
from pathlib import Path

from mechet.metrics import (
    aggregate_rates,
    build_eval_report,
    canonical_species,
    normalize_candidates,
    score_mech_et_prediction,
    top1_main_only_match,
    top1_strict_match,
    topk_strict_hit,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data/samples/valid_mini.jsonl"


def test_canonical_species_order_independent():
    a = canonical_species("CCO.CC")
    b = canonical_species("CC.CCO")
    assert a == b


def test_top1_strict_match():
    assert top1_strict_match("CCO", "CCO")
    assert not top1_strict_match("CCO", "CC")


def test_top1_main_only_match():
    assert top1_main_only_match("CC.CCO", "CCO")
    assert not top1_main_only_match("CCO", "CC")


def test_topk_strict_hit():
    gold = "CCO.CC"
    assert topk_strict_hit(["CCN", "CC.CCO"], gold, 1) is False
    assert topk_strict_hit(["CCN", "CC.CCO"], gold, 2) is True


def test_normalize_candidates_dedupes():
    texts = [
        "<answer>CCO</answer>",
        "<answer>CCO</answer>",
        "<answer>CC</answer>",
    ]
    assert normalize_candidates(texts[0], texts[1:]) == ["CCO", "CC"]


def test_gold_audit_on_mini_sample():
    row = json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])
    assistant = row["messages"][-1]["content"]
    case = score_mech_et_prediction(row, assistant, mode="gold_audit")
    assert case["format_ok"]
    assert case["answer_exact"]


def test_model_mode_topk_fields():
    row = json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])
    assistant = row["messages"][-1]["content"]
    case = score_mech_et_prediction(row, assistant, mode="model")
    assert case["top1_strict"]
    assert case["top5_strict"]
    assert case["top10_strict"]


def test_build_eval_report():
    report = build_eval_report(
        [{"format_ok": True, "top1_strict": True, "top5_strict": True}],
        mode="model",
        data_path="x.jsonl",
        predictions_path="y.jsonl",
    )
    assert report["rates"]["top1_strict_rate"] == 1.0


def test_aggregate_rates_keys():
    agg = aggregate_rates([{"format_ok": True, "top1_strict": False, "top5_strict": False}])
    assert "rates" in agg
    assert agg["rates"]["format_ok_rate"] == 1.0
