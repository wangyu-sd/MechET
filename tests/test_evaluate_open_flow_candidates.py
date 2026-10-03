import json
import sys

from scripts.evaluate_open_flow_candidates import main, score_candidate


def test_open_flow_candidate_scores_endpoint_only_after_execution():
    moves = [
        {"source": {"kind": "BOND", "atoms": [1, 2]},
         "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [3]},
         "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
    ]
    program = f"<flow>\nOPEN_FLOW v1\nIMPORT [Br-:3]\nSTEP 0 {json.dumps(moves)}\nEXECUTE\n</flow>"
    reference = {
        "id": "one",
        "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
        "full_precursor_state": "[CH3:1][Br:3].[OH-:2]",
    }
    result = score_candidate({"prediction": program, "sample_index": 0}, reference, 0)
    assert result["execute_ok"] is True
    assert result["structural_exact"] is True
    assert result["full_precursor_exact"] is True


def test_open_flow_invalid_candidate_is_not_repaired():
    reference = {"id": "one", "target_smiles": "[CH3:1][OH:2]",
                 "structural_precursor": "[CH3:1][Br:3]"}
    result = score_candidate({"prediction": "<flow>bad</flow>"}, reference, 0)
    assert result["execute_ok"] is False
    assert result["structural_exact"] is False
    assert result["failure_code"] == "FLOW_HEADER_OR_TERMINAL_INVALID"


def test_open_flow_report_carries_single_adapter_runtime_lineage(tmp_path, monkeypatch):
    reference = tmp_path / "reference.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "evaluation.json"
    reference.write_text(json.dumps({
        "id": "one", "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": "[CH3:1][Br:3]",
    }) + "\n")
    predictions.write_text(json.dumps({
        "id": "one",
        "model": {
            "base_model": "Qwen/Qwen3-8B",
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "adapter": "/tmp/frozen-adapter",
            "adapter_sha256": "frozen-adapter-hash",
            "temperature": 0.7, "top_p": 0.95,
            "max_new_tokens": 4096, "max_iterations": 12,
            "samples_per_target": 1, "seed": 17,
            "candidate_selector": "sample0_direct__formal_trace_reward_failures_v1",
        },
        "candidates": [{"sample_index": 0, "prediction": "<flow>bad</flow>"}],
    }) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "evaluate_open_flow_candidates.py", "--reference", str(reference),
        "--predictions", str(predictions), "--output", str(output),
        "--expected-rows", "1", "--expected-candidates", "1",
    ])
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["runtime_contract"]["runtime_contract_complete"] is True
    assert report["runtime_contract"]["adapter_ids"] == ["frozen-adapter-hash"]
