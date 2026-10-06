import json
from pathlib import Path

import pytest

from scripts.analyze_reliable_product_start import analyze, first_reference_divergence


def gold_decision(index: int, name: str, state: str) -> dict:
    result_key = "derived_precursor" if name == "finish_trace" else "current_state"
    return {
        "id": f"r::decision_{index}", "source_id": "r",
        "metadata": {"decision_index": index},
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "tool_calls": [{"function": {"name": name}}]},
            {"role": "tool", "content": json.dumps({result_key: state})},
        ],
    }


def test_first_divergence_distinguishes_fragment_from_event_failure():
    gold = [gold_decision(0, "import_fragments", "CO.O")]
    wrong_import = {"id": "r", "attempts": [{
        "depth": 0, "name": "import_fragments", "accepted": True,
        "state_after": "CO.N", "terminal": False,
    }]}
    assert first_reference_divergence(wrong_import, gold)["category"] == "fragment_proposal"
    bad_alias = {"id": "r", "attempts": [{
        "depth": 0, "name": "apply_electron_flow", "accepted": False,
        "error": "ValueError:unrecognized natural-language electron source: A99",
    }]}
    assert first_reference_divergence(bad_alias, gold)["category"] == "source_sink_grounding"


def test_reference_divergence_does_not_make_exact_alternative_a_failure(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    results = tmp_path / "results.jsonl"
    source.write_text(json.dumps({"id": "r", "source_id": "r"}) + "\n")
    decisions.write_text("\n".join(json.dumps(row) for row in [
        gold_decision(0, "import_fragments", "CO.O"),
        gold_decision(1, "finish_trace", "CO.O"),
    ]) + "\n")
    results.write_text(json.dumps({
        "id": "r", "source_id": "r", "top1_exact": True,
        "top_terminal": True, "top_policy_score": -0.3,
        "n_actions": 2,
        "attempts": [
            {"depth": 0, "name": "import_fragments", "accepted": True,
             "state_after": "CO.N", "terminal": False},
            {"depth": 1, "name": "finish_trace", "accepted": True,
             "state_after": "CO.O", "terminal": True},
        ],
    }) + "\n")
    report, cases = analyze(
        source=source, decisions=decisions, results=results,
        sample_reactions=1, seed=17,
    )
    assert report["endpoint_exact"] == 1
    assert cases[0]["failure_category"] is None
    assert cases[0]["recovered_after_reference_divergence"]


def test_analysis_refuses_missing_result_denominator(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    results = tmp_path / "results.jsonl"
    source.write_text(json.dumps({"id": "r", "source_id": "r"}) + "\n")
    decisions.write_text(json.dumps(gold_decision(0, "finish_trace", "CO")) + "\n")
    results.write_text("")
    with pytest.raises(ValueError, match="incomplete product-start result denominator"):
        analyze(
            source=source, decisions=decisions, results=results,
            sample_reactions=1, seed=17,
        )


def test_nonterminal_empty_path_cannot_look_more_confident_than_terminal(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    results = tmp_path / "results.jsonl"
    source.write_text("\n".join(json.dumps({"id": key, "source_id": key}) for key in ("r", "s")) + "\n")
    gold_r = gold_decision(0, "finish_trace", "CO")
    gold_s = {**gold_r, "id": "s::decision_0", "source_id": "s"}
    decisions.write_text(json.dumps(gold_r) + "\n" + json.dumps(gold_s) + "\n")
    exact = {
        "id": "r", "source_id": "r", "top1_exact": True,
        "top_terminal": True, "top_policy_score": -1.0, "n_actions": 1,
        "attempts": [{"depth": 0, "name": "finish_trace", "accepted": True,
                      "state_after": "CO", "terminal": True}],
    }
    incomplete = {
        "id": "s", "source_id": "s", "top1_exact": False,
        "top_terminal": False, "top_policy_score": 0.0, "n_actions": 0,
        "attempts": [{"depth": 0, "name": "", "accepted": False,
                      "error": "NO_PARSEABLE_TOOL_CALL", "state_after": "", "terminal": False}],
    }
    results.write_text(json.dumps(exact) + "\n" + json.dumps(incomplete) + "\n")
    report, cases = analyze(
        source=source, decisions=decisions, results=results,
        sample_reactions=2, seed=17,
    )
    assert next(case for case in cases if case["source_id"] == "s")["policy_score"] is None
    assert report["risk_coverage_by_policy_score"][2]["n"] == 1
    assert report["risk_coverage_by_policy_score"][2]["endpoint_miss_rate"] == 0.0
