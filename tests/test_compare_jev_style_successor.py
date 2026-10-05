import json

import pytest

from scripts.compare_jev_style_successor import ARTIFACT_TYPES, compare_split


def _make_result(root, name, successes):
    directory = root / name
    directory.mkdir()
    cases = []
    for index, exact in enumerate(successes):
        selected = {
            "execute_ok": True,
            "successor_exact": exact,
            "successor": "CCO" if exact else "CC",
            "code": "PASS",
        }
        cases.append({
            "id": f"r{index}::event",
            "gold_successor": "CCO",
            "pair_targets": [0, 1] if index == 0 else [1],
            "gold_flow_count": 2 if index == 0 else 1,
            "ranked_top8": [1, 0],
            "policies": {
                "fixed1": selected,
                "fixed2": {"execute_ok": False, "successor_exact": False,
                           "successor": None, "code": "CHEMICAL_STATE_INVALID"},
                "validity_backoff_2_to_1": selected,
            },
        })
    report = {
        "artifact_type": ARTIFACT_TYPES[name],
        "scope": "reference_current_state_not_product_start",
        "split": "valid",
        "source": {"sha256": "frozen", "declared_sha256": "frozen", "event_decisions": 2},
        "evaluated_events": 2,
        "gold_replay_ok": 2,
        "checkpoint_manifest": name,
        "pair_recall": {"pair_r1": 0.5, "pair_all_r8": 1.0},
        "policies": {"validity_backoff_2_to_1": {"overall": {
            "n": 2, "execute_ok": 2, "successor_exact": sum(successes)
        }}},
        "elapsed_s": 2.0,
    }
    (directory / "report.json").write_text(json.dumps(report))
    with (directory / "cases.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    return directory


def test_three_model_comparison_uses_same_ids_and_backoff(tmp_path):
    directories = {
        "typed_v2": _make_result(tmp_path, "typed_v2", [True, True]),
        "marker_v1": _make_result(tmp_path, "marker_v1", [True, False]),
        "pr71_8b": _make_result(tmp_path, "pr71_8b", [False, False]),
    }
    result = compare_split("valid", directories)
    assert result["events"] == 2
    assert result["two_flow_events"] == 1
    assert result["models"]["typed_v2"]["successor_exact"] == 2
    assert result["models"]["typed_v2"]["two_flow_top2_gold_set"] == 1
    assert result["paired"]["typed_v2_minus_marker_v1"]["successor_exact_rate_difference"] == 0.5
    assert result["paired"]["typed_v2_minus_pr71_8b"]["successor_exact_rate_difference"] == 1.0


def test_three_model_comparison_rejects_reference_mismatch(tmp_path):
    directories = {
        "typed_v2": _make_result(tmp_path, "typed_v2", [True, True]),
        "marker_v1": _make_result(tmp_path, "marker_v1", [True, False]),
        "pr71_8b": _make_result(tmp_path, "pr71_8b", [False, False]),
    }
    cases_path = directories["marker_v1"] / "cases.jsonl"
    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    rows[0]["gold_successor"] = "different"
    cases_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="reference gold_successor"):
        compare_split("valid", directories)
