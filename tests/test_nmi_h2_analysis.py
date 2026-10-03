import hashlib
import json

import pytest
import numpy as np

from scripts.analyze_nmi_h2_results import analyze, _adjusted_paired_primitive_advantage


def _runtime_contract(condition, n, k, *, seed=17):
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    return {
        "n_rows": n,
        "n_unique_adapters": 1,
        "runtime_contract_consistent": True,
        "runtime_contract_complete": True,
        "runtime_contract_sha256": "test-contract",
        "runtime_contracts": [{
            "model_revision": revision,
            "tokenizer_revision": revision,
            "adapter_sha256": f"adapter-{condition}",
            "samples_per_target": k,
            "seed": seed,
            "temperature": 0.7,
            "top_p": 0.95,
        }],
    }


def _write_jsonl(path, rows):
    data = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_h2_analysis_requires_paired_ids_and_reports_reaction_delta(tmp_path):
    matched = tmp_path / "matched"
    evaluations = {}
    identifiers = [f"r{i}" for i in range(4)]
    for condition in ("direct", "open_flow", "closed_loop"):
        path = matched / condition
        path.mkdir(parents=True)
        reference = path / "test.jsonl"
        reference_sha = _write_jsonl(reference, [
            {"id": f"id_{i}", "source_id": identifier} for i, identifier in enumerate(identifiers)
        ])
        (path / "manifest.json").write_text(json.dumps({
            "rows": {"test": 4}, "output_sha256": {"test": reference_sha},
        }))
        evaluation = tmp_path / f"{condition}.json"
        evaluation_rows = tmp_path / f"{condition}.rows.jsonl"
        hits = {
            "direct": [0, 0, 1, 0],
            "open_flow": [0, 1, 0, 0],
            "closed_loop": [1, 1, 0, 0],
        }[condition]
        row_sha = _write_jsonl(evaluation_rows, [
            {"id": f"id_{i}", "source_id": identifier,
             "generation_order": [0, 1],
             **({"selected_candidate_index": 1, "selected_structural_exact": False,
                 "selected_formal_execute": True} if condition == "closed_loop" else
                {"nll_ranked_order": [1, 0]} if condition == "direct" else
                {"formal_nll_ranked_order": [1, 0]}),
             "candidates": [
                 {"candidate_index": 0, "structural_exact": bool(hits[i]),
                  "execute_ok": condition != "direct",
                  "trace_bound": condition == "closed_loop"},
                 {"candidate_index": 1, "structural_exact": False,
                  "execute_ok": condition != "direct",
                  "trace_bound": condition == "closed_loop"},
             ]} for i, identifier in enumerate(identifiers)
        ])
        report = {"reference_sha256": reference_sha, "n_reference_rows": 4,
                  "row_evaluation": str(evaluation_rows), "row_evaluation_sha256": row_sha,
                  "runtime_contract": _runtime_contract(condition, 4, 2)}
        if condition == "closed_loop":
            report.update({"candidate_count_min": 2, "candidate_count_max": 2})
        else:
            report["candidates_per_target"] = 2
        evaluation.write_text(json.dumps(report))
        evaluations[condition] = evaluation
    covariates = tmp_path / "covariates.jsonl"
    _write_jsonl(covariates, [
        {"source_id": identifier, "split": "test", "program_train_frequency": 0,
         "fraction_primitives_seen_in_train": 1.0,
         "minimum_primitive_train_frequency": 5 + i,
         "trajectory_steps": i + 1, "fragment_imports": i % 2,
         "structural_overlap": {"murcko_scaffold_seen_in_train": bool(i % 2),
                                "local_center_fraction_seen_in_train": 0.25 * i,
                                "near_duplicate_at_threshold": False}}
        for i, identifier in enumerate(identifiers)
    ])
    result = analyze(covariates, matched, evaluations, tmp_path / "result.json",
                     k=2, bootstrap_draws=100, regression_draws=20)
    assert result["n_reactions"] == 4
    assert result["paired_contrasts"]["closed_minus_open_flow"]["endpoint_at_1"]["estimate"] == 0.25
    assert result["methods"]["closed_loop"]["endpoint_at_1"]["estimate"] == 0.5
    assert result["methods"]["closed_loop"]["selected_endpoint_at_1"]["estimate"] == 0.0
    assert result["methods"]["closed_loop"]["trace_bound_at_1"]["estimate"] == 1.0
    assert "trace_bound_at_1" not in result["methods"]["open_flow"]
    assert "not exact agreement" in result["comparison_semantics"]["closed_loop_trace_bound"]
    assert result["lineage"]["direct"]["selected_candidate_rule"] == "assistant_mean_nll"
    assert result["lineage"]["direct"]["adapter_sha256"] == "adapter-direct"
    assert "not matched" in result["comparison_semantics"]["selected_top1"]
    assert result["adjusted_paired_primitive_advantage"]["closed_minus_open_flow"]["status"] == "unidentifiable_design_rank"
    support = result["primitive_frequency_strata"]["5_to_9"]
    assert support["n"] == 4
    assert support["generation_order_endpoint_at_1"]["closed_loop"] == 0.5
    assert support["paired_closed_minus_open_flow"]["endpoint_at_1"]["estimate"] == 0.25
    assert result["primitive_frequency_strata"]["1000_plus"]["n"] == 0


def test_h2_analysis_rejects_missing_ranked_order(tmp_path):
    from scripts.analyze_nmi_h2_results import _row_results

    matched = tmp_path / "matched" / "direct"
    matched.mkdir(parents=True)
    reference_sha = _write_jsonl(matched / "test.jsonl", [{"id": "id_0", "source_id": "r0"}])
    (matched / "manifest.json").write_text(json.dumps({
        "rows": {"test": 1}, "output_sha256": {"test": reference_sha},
    }))
    evaluation = tmp_path / "direct.json"
    _write_jsonl(tmp_path / "direct.rows.jsonl", [{
        "id": "id_0", "source_id": "r0", "candidates": [
            {"candidate_index": 0, "structural_exact": False},
            {"candidate_index": 1, "structural_exact": True},
        ],
    }])
    evaluation.write_text(json.dumps({
        "reference_sha256": reference_sha, "n_reference_rows": 1,
        "candidates_per_target": 2,
    }))
    with pytest.raises(ValueError, match="missing/invalid frozen ranking"):
        _row_results("direct", evaluation, matched.parent, expected_k=2)


def test_h2_analysis_rejects_closed_loop_selected_candidate_mismatch(tmp_path):
    from scripts.analyze_nmi_h2_results import _row_results

    matched = tmp_path / "matched" / "closed_loop"
    matched.mkdir(parents=True)
    reference_sha = _write_jsonl(matched / "test.jsonl", [
        {"id": "id_0", "source_id": "r0"},
    ])
    (matched / "manifest.json").write_text(json.dumps({
        "rows": {"test": 1}, "output_sha256": {"test": reference_sha},
    }))
    evaluation = tmp_path / "closed_loop.json"
    _write_jsonl(tmp_path / "closed_loop.rows.jsonl", [{
        "id": "id_0", "source_id": "r0", "selected_candidate_index": 1,
        "selected_structural_exact": True, "selected_formal_execute": True,
        "candidates": [
            {"candidate_index": 0, "structural_exact": True, "execute_ok": True},
            {"candidate_index": 1, "structural_exact": False, "execute_ok": True},
        ],
    }])
    evaluation.write_text(json.dumps({
        "reference_sha256": reference_sha, "n_reference_rows": 1,
        "candidate_count_min": 2, "candidate_count_max": 2,
    }))
    with pytest.raises(ValueError, match="selected endpoint disagrees"):
        _row_results("closed_loop", evaluation, matched.parent, expected_k=2)


def test_h2_analysis_requires_closed_loop_trace_bound_field(tmp_path):
    from scripts.analyze_nmi_h2_results import _row_results

    matched = tmp_path / "matched" / "closed_loop"
    matched.mkdir(parents=True)
    reference_sha = _write_jsonl(matched / "test.jsonl", [
        {"id": "id_0", "source_id": "r0"},
    ])
    (matched / "manifest.json").write_text(json.dumps({
        "rows": {"test": 1}, "output_sha256": {"test": reference_sha},
    }))
    rows = tmp_path / "closed_loop.rows.jsonl"
    row_sha = _write_jsonl(rows, [{
        "id": "id_0", "source_id": "r0", "selected_candidate_index": 0,
        "selected_structural_exact": False, "selected_formal_execute": True,
        "candidates": [
            {"candidate_index": 0, "structural_exact": False, "execute_ok": True},
            {"candidate_index": 1, "structural_exact": False, "execute_ok": True},
        ],
    }])
    evaluation = tmp_path / "closed_loop.json"
    evaluation.write_text(json.dumps({
        "reference_sha256": reference_sha, "n_reference_rows": 1,
        "row_evaluation": str(rows), "row_evaluation_sha256": row_sha,
        "candidate_count_min": 2, "candidate_count_max": 2,
    }))
    with pytest.raises(ValueError, match="missing trace-bound candidate"):
        _row_results("closed_loop", evaluation, matched.parent, expected_k=2)


def test_h2_analysis_rejects_missing_runtime_lineage(tmp_path):
    from scripts.analyze_nmi_h2_results import _row_results

    matched = tmp_path / "matched" / "direct"
    matched.mkdir(parents=True)
    reference_sha = _write_jsonl(matched / "test.jsonl", [
        {"id": "id_0", "source_id": "r0"},
    ])
    (matched / "manifest.json").write_text(json.dumps({
        "rows": {"test": 1}, "output_sha256": {"test": reference_sha},
    }))
    rows = tmp_path / "direct.rows.jsonl"
    row_sha = _write_jsonl(rows, [{
        "id": "id_0", "source_id": "r0", "nll_ranked_order": [0, 1],
        "candidates": [
            {"candidate_index": 0, "structural_exact": False},
            {"candidate_index": 1, "structural_exact": False},
        ],
    }])
    evaluation = tmp_path / "direct.json"
    evaluation.write_text(json.dumps({
        "reference_sha256": reference_sha, "n_reference_rows": 1,
        "row_evaluation": str(rows), "row_evaluation_sha256": row_sha,
        "candidates_per_target": 2,
    }))
    with pytest.raises(ValueError, match="missing/inconsistent inference runtime contract"):
        _row_results("direct", evaluation, matched.parent, expected_k=2)


def test_paired_familiarity_advantage_bootstraps_reaction_differences():
    features = np.array([
        [i / 10, i % 2, (i * 7 % 11) / 10, int(i % 5 == 0),
         np.log2(2 + i % 7), i % 4]
        for i in range(40)
    ], dtype=float)
    closed = np.array([int(i % 3 == 0) for i in range(40)], dtype=float)
    open_flow = np.array([int(i % 5 == 0) for i in range(40)], dtype=float)
    result = _adjusted_paired_primitive_advantage(
        features, closed, open_flow, np.random.default_rng(17), draws=100,
    )
    assert result["status"] == "estimated"
    assert np.isfinite(result["slope"])
    assert len(result["slope_ci95"]) == 2
    assert result["reaction_bootstrap_draws_used"] >= 50
