import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.analyze_reliable_product_start import (
    analyze, attempt_confidence_report, first_reference_divergence,
)
from scripts.run_natural_language_value_search import Action, search_unlabeled


def test_product_start_records_per_action_mean_logprob() -> None:
    runtime = SimpleNamespace(
        pointer_invalid_handles=0,
        proposals=lambda *_args, **_kwargs: [
            Action("finish_trace", {}, "", logprob=-2.0, tokens=4)
        ],
        values=lambda *_args, **_kwargs: [0.0],
    )
    args = SimpleNamespace(
        max_decisions=1, max_imports=1, branching=1,
        max_new_tokens=8, compact_history=False, planning_sample=False,
        product_only_remap=True, vnext_v2_prefix=False,
        reject_target_retained_finish=False,
        early_depth=1, early_beam=1, late_beam=1,
        value_weight=0.0, pointer_weight=0.0,
    )
    result = search_unlabeled(runtime, "C", args)
    attempt = result["attempts"][0]
    assert attempt["accepted"] is True
    assert attempt["action_logprob"] == -2.0
    assert attempt["action_tokens"] == 4
    assert attempt["action_policy_score"] == -0.5
    runtime.proposals = lambda *_args, **_kwargs: []
    runtime.last_proposal_error = "NO_PARSEABLE_TOOL_CALL"
    unparseable = search_unlabeled(runtime, "C", args)["attempts"][0]
    assert unparseable["accepted"] is False
    assert unparseable["action_policy_score"] is None
    assert unparseable["action_tokens"] == 0


def test_attempt_confidence_separates_executable_rejection_from_legacy_schema() -> None:
    records = [{"attempts": [
        {"name": "apply_electron_flow", "accepted": False,
         "action_logprob": -0.2, "action_tokens": 2, "action_policy_score": -0.1},
        {"name": "finish_trace", "accepted": True,
         "action_logprob": -4.0, "action_tokens": 4, "action_policy_score": -1.0},
        {"name": "", "accepted": False, "action_policy_score": None},
    ]}]
    report = attempt_confidence_report(records)
    assert report["parsed_attempts"] == 2
    assert report["unparseable_attempts"] == 1
    assert report["executor_rejection_risk_coverage"][0]["executor_rejection_rate"] == 1.0
    assert report["executor_rejection_risk_coverage"][-1]["executor_rejection_rate"] == 0.5
    assert attempt_confidence_report([{"attempts": [
        {"name": "finish_trace", "accepted": True},
    ]}])["available"] is False
    assert attempt_confidence_report([{"attempts": [
        {"name": "", "accepted": False, "action_policy_score": None},
    ]}])["reason"] == "no parsed action proposals to rank"
    with pytest.raises(ValueError, match="mixed scored/unscored"):
        attempt_confidence_report([{"attempts": [
            records[0]["attempts"][0], {"name": "finish_trace", "accepted": True},
        ]}])
    with pytest.raises(ValueError, match="not mean token log-probability"):
        attempt_confidence_report([{"attempts": [
            {**records[0]["attempts"][0], "action_policy_score": 0.0},
        ]}])


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


def test_mapping_parity_audit_separates_reference_unstable_cases(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    results = tmp_path / "results.jsonl"
    audit = tmp_path / "mapping_audit.json"
    source.write_text("".join(json.dumps({"id": key, "source_id": key}) + "\n" for key in ("r", "s")))
    gold_r = gold_decision(0, "finish_trace", "CO")
    gold_s = {**gold_r, "id": "s::decision_0", "source_id": "s"}
    decisions.write_text(json.dumps(gold_r) + "\n" + json.dumps(gold_s) + "\n")
    exact = {
        "id": "r", "source_id": "r", "top1_exact": True,
        "top_terminal": True, "top_policy_score": -1.0, "n_actions": 1,
        "attempts": [{"depth": 0, "name": "finish_trace", "accepted": True,
                      "state_after": "CO", "terminal": True}],
    }
    missed = {
        "id": "s", "source_id": "s", "top1_exact": False,
        "top_terminal": False, "top_policy_score": 0.0, "n_actions": 0,
        "attempts": [{"depth": 0, "name": "", "accepted": False,
                      "error": "NO_PARSEABLE_TOOL_CALL", "state_after": "", "terminal": False}],
    }
    results.write_text(json.dumps(exact) + "\n" + json.dumps(missed) + "\n")
    audit.write_text(json.dumps({
        "artifact_type": "reliable_mechet_product_only_private_mapping_audit_v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "decisions_sha256": hashlib.sha256(decisions.read_bytes()).hexdigest(),
        "n_reactions": 2,
        "counts": {
            "root_prompt_exact": 2,
            "original_private_map_replay_ok": 2,
            "product_only_remap_replay_ok": 1,
        },
        "failures": [{"source_id": "s", "kind": "product_only_remap_replay_failed"}],
    }))
    report, cases = analyze(
        source=source, decisions=decisions, results=results,
        sample_reactions=2, seed=17, mapping_parity_report=audit,
    )
    assert report["endpoint_exact"] == 1
    assert report["denominator"] == 2
    assert report["reference_mapping_parity_failed"] == 1
    assert report["first_failure_category_counts"] == {"generation_or_budget": 1}
    assert report["first_failure_category_counts_parity_stable"] == {}
    assert next(case for case in cases if case["source_id"] == "s")[
        "reference_product_only_mapping_parity"
    ] is False
    audit_data = json.loads(audit.read_text())
    audit_data["source_sha256"] = "0" * 64
    audit.write_text(json.dumps(audit_data))
    with pytest.raises(ValueError, match="SHA mismatch"):
        analyze(
            source=source, decisions=decisions, results=results,
            sample_reactions=2, seed=17, mapping_parity_report=audit,
        )
