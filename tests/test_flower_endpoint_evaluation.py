import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_flower_endpoint_matched_subset import build_subset
from score_and_rank_predictions import _rank_key
from mechet.knowledge_ablation import extract_direct_prediction


def _row(identifier: str, source: str, target: str, precursor: str) -> dict:
    return {
        "id": identifier,
        "source_id": source,
        "target_smiles": target,
        "structural_precursor": precursor,
        "expected_precursor": precursor,
    }


def test_build_matched_subset_uses_trajectory_id_and_checks_endpoints():
    full = [
        _row("full:4", "4", "[CH3:1][OH:2]", "[CH3:1][Br:3].[OH-:2]"),
        _row("full:5", "5", "[CH3:1][NH2:2]", "[CH3:1][Cl:3].[NH2-:2]"),
    ]
    trace = [
        _row(
            "trace:4",
            "flower_mech_proof_test_4",
            "[CH3:1][OH:2]",
            "[OH-:20].[Br:30][CH3:10]",
        )
    ]
    result = build_subset(full, trace)
    assert [row["id"] for row in result] == ["full:4"]


def test_direct_rank_key_uses_mean_nll_without_trace_fields():
    candidate = {"sample_index": 0, "rollout_state": {}}
    low = {"assistant_mean_nll": 0.5, "score_status": "ok"}
    high = {"assistant_mean_nll": 1.5, "score_status": "ok"}
    assert _rank_key(low, candidate, direct=True) > _rank_key(
        high, candidate, direct=True
    )


def test_direct_endpoint_extraction_supports_iclr_answer_blocks():
    prediction = (
        "<mechanism>\nMECH_ET v3\n</mechanism>\n"
        "<answer>\n[OH-:2].[CH3:1][Br:3]\n</answer>"
    )
    assert extract_direct_prediction({"prediction": prediction}) == (
        "[OH-:2].[CH3:1][Br:3]"
    )


def test_endpoint_evaluator_reports_pass_and_nll_ranked_topk(tmp_path: Path):
    reference = tmp_path / "reference.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ranking_dir = tmp_path / "ranking"
    ranking_dir.mkdir()
    output = tmp_path / "evaluation.json"
    ref = _row("r1", "1", "[CH3:1][OH:2]", "[CH3:1][Br:3].[OH-:2]")
    pred = {
        "id": "r1",
        "artifact_type": "prediction",
        "prediction_mode": "direct",
        "candidates": [
            {"sample_index": 0, "prediction": "PRECURSOR: C"},
            {
                "sample_index": 1,
                "prediction": "PRECURSOR: [OH-:20].[Br:30][CH3:10]",
            },
        ],
    }
    ranking = {
        "id": "r1",
        "ranker": "formal_trace_gate__assistant_mean_nll_v1",
        "ranked_candidate_indices": [1, 0],
        "candidate_scores": [
            {"assistant_mean_nll": 2.0, "score_status": "ok"},
            {"assistant_mean_nll": 1.0, "score_status": "ok"},
        ],
    }
    reference.write_text(json.dumps(ref) + "\n")
    predictions.write_text(json.dumps(pred) + "\n")
    (ranking_dir / "nll_scores.shard-00.jsonl").write_text(json.dumps(ranking) + "\n")
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_endpoint_candidates.py",
            "--reference",
            str(reference),
            "--predictions",
            str(predictions),
            "--ranking-dir",
            str(ranking_dir),
            "--output",
            str(output),
            "--expected-rows",
            "1",
            "--expected-candidates",
            "2",
        ],
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["generation_order"]["structural_pass_at_1"] == 0.0
    assert report["generation_order"]["structural_pass_at_3"] == 1.0
    assert report["nll_ranked"]["structural_top_1"] == 1.0
    assert report["nll_ranked"]["selection_uses_ground_truth"] is False
