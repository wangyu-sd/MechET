import json

import pytest

from scripts.autoresearch.analyze_r3_repair_failures import analyze
from scripts.autoresearch.stratified_manifest import digest


def test_r3_failure_taxonomy_reconstructs_scored_totals(tmp_path):
    details = tmp_path / "repair_details.jsonl"
    rows = [
        {"case_id": "a", "stratum": "early/1", "generation_status": "completed",
         "exact": True, "repair_action_accepted": True, "error": None},
        {"case_id": "b", "stratum": "early/1", "generation_status": "completed",
         "exact": False, "repair_action_accepted": True, "error": None},
        {"case_id": "c", "stratum": "late/3+", "generation_status": "completed",
         "exact": False, "repair_action_accepted": False,
         "error": "ValueError:invalid bond"},
        {"case_id": "d", "stratum": "late/3+", "generation_status": "failed",
         "exact": False, "repair_action_accepted": False,
         "error": "missing_or_non_electron_replacement_action"},
    ]
    details.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "artifact_type": "r3_exposed_failure_one_action_repair_result_v1",
        "cases": 4, "details_sha256": digest(details), "endpoint_exact": 1,
        "repair_action_accepted": 2}))
    output = tmp_path / "analysis.json"
    report = analyze(result, details, output, expected_cases=4)
    assert report["categories"] == {
        "accepted_reference_relative_wrong_endpoint": 1,
        "generation_failed": 1,
        "oracle_suffix_endpoint_recovered": 1,
        "replacement_action_rejected": 1,
    }
    assert report["error_families"] == {
        "ValueError": 1, "missing_or_non_electron_replacement_action": 1}
    with pytest.raises(FileExistsError):
        analyze(result, details, output, expected_cases=4)
    output.unlink()
    details.write_text(details.read_text().replace("invalid bond", "changed bond"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        analyze(result, details, output, expected_cases=4)
