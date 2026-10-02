import hashlib
import json

from scripts.analyze_nmi_h2_results import analyze


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
             "candidates": [
                 {"structural_exact": bool(hits[i]), "execute_ok": condition != "direct"},
                 {"structural_exact": False, "execute_ok": condition != "direct"},
             ]} for i, identifier in enumerate(identifiers)
        ])
        report = {"reference_sha256": reference_sha, "n_reference_rows": 4,
                  "row_evaluation": str(evaluation_rows), "row_evaluation_sha256": row_sha}
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
