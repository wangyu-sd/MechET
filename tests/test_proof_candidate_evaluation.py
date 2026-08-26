import json
from pathlib import Path
import subprocess
import sys

from mechet.proof_program import format_proof_output

from test_proof_program import substitution_program


def test_complete_proof_candidates_use_execution_gate_then_nll(tmp_path: Path):
    proof = format_proof_output(substitution_program())
    reference = tmp_path / "reference.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ranking_dir = tmp_path / "rankings"
    output = tmp_path / "evaluation.json"
    ranking_dir.mkdir()
    reference.write_text(
        json.dumps(
            {
                "id": "r1",
                "target_smiles": "[CH3:1][OH:2]",
                "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
                "full_precursor_state": "[CH3:1][Br:3].[OH-:2]",
            }
        )
        + "\n"
    )
    predictions.write_text(
        json.dumps(
            {
                "id": "r1",
                "candidates": [
                    {"sample_index": 0, "prediction": "<proof>broken</proof>"},
                    {"sample_index": 1, "prediction": proof},
                ],
            }
        )
        + "\n"
    )
    (ranking_dir / "nll_scores.shard-00.jsonl").write_text(
        json.dumps(
            {
                "id": "r1",
                "candidate_scores": [
                    {"assistant_mean_nll": 0.1, "score_status": "ok"},
                    {"assistant_mean_nll": 2.0, "score_status": "ok"},
                ],
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_proof_candidates.py",
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
    assert report["generation_order"]["execute_ok_at_1"] == 0.0
    assert report["generation_order"]["structural_exact_at_5"] == 1.0
    assert report["formal_nll_ranked"]["execute_ok_at_1"] == 1.0
    assert report["formal_nll_ranked"]["structural_exact_at_1"] == 1.0
    assert report["formal_nll_ranked"]["selection_uses_ground_truth"] is False
