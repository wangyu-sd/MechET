import hashlib
import importlib.util
import json
from pathlib import Path

from mechet.agent_env import AgentEnvConfig
from mechet.knowledge_agent_env import KnowledgeAgentConfig, KnowledgeAugmentedAgentEnv
from mechet.prediction_metrics import prediction_runtime_contract
from mechet.strict_prediction_evaluation import endpoint_evaluation


def _load_infer_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "infer_mechet.py"
    spec = importlib.util.spec_from_file_location("infer_mechet_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_row():
    return {
        "id": "r1",
        "artifact_type": "prediction",
        "prediction_status": "completed",
        "prediction_mode": "trace",
        "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
    }


def test_unfinished_trace_never_receives_endpoint_credit():
    row = _reference_row()
    row.update(
        {
            "prediction": "PRECURSOR: [CH3:1][Br:3].[OH-:2]",
            "rollout_state": {
                "finalized": False,
                "flow_trace": {
                    "target_smiles": "[CH3:1][OH:2]",
                    "transitions": [],
                },
            },
            "messages": [
                {
                    "role": "assistant",
                    "content": "PRECURSOR: [CH3:1][Br:3].[OH-:2]",
                }
            ],
        }
    )
    result = endpoint_evaluation(row)
    assert result["prediction_present"] is False
    assert result["structural_exact"] is False
    assert result["prediction_source"] == "unfinished_trace"
    assert result["completion_failure"] == "TRACE_NOT_FINALIZED"


def test_trace_mode_never_falls_back_to_direct_answer():
    row = _reference_row()
    row.update(
        {
            "prediction": "PRECURSOR: [CH3:1][Br:3].[OH-:2]",
            "rollout_state": {"finalized": True, "abstained": False},
            "messages": [
                {
                    "role": "assistant",
                    "content": "PRECURSOR: [CH3:1][Br:3].[OH-:2]",
                }
            ],
        }
    )
    result = endpoint_evaluation(row)
    assert result["prediction_present"] is False
    assert result["completion_failure"] == "FINISH_TRACE_RESULT_REQUIRED"


def test_runtime_contract_rejects_consistently_empty_metadata():
    rows = [{"id": "a", "model": {}}, {"id": "b", "model": {}}]
    report = prediction_runtime_contract(rows, include_adapter=True)
    assert report["runtime_contract_consistent"] is True
    assert report["runtime_contract_complete"] is False
    assert report["runtime_contract_missing_fields_by_row"]


def test_scripted_runtime_contract_may_be_adapterless():
    model = {
        "base_model": "scripted",
        "model_revision": "scripted",
        "tokenizer_revision": "scripted",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 128,
        "max_iterations": 4,
        "samples_per_target": 1,
        "seed": 17,
        "candidate_selector": "selector-v1",
        "adapter": "/tmp/nonexistent-scripted-adapter",
        "adapter_sha256": None,
    }
    report = prediction_runtime_contract(
        [{"id": "a", "model": model}], include_adapter=True
    )
    assert report["runtime_contract_complete"] is True


def test_candidate_seed_is_stable_and_sample_specific():
    module = _load_infer_module()
    first = module._candidate_seed(17, "reaction-1", 0)
    assert first == module._candidate_seed(17, "reaction-1", 0)
    assert first != module._candidate_seed(17, "reaction-1", 1)
    assert first != module._candidate_seed(17, "reaction-2", 0)


def _write_corpus(path: Path):
    text = "A nucleophile displaces a leaving group."
    row = {
        "passage_id": "p1",
        "title": "Substitution",
        "text": text,
        "source_id": "source",
        "locator": "chapter/1",
        "revision": "r1",
        "license": "CC-BY-4.0",
        "source_url": "https://example.org",
        "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "topics": ["substitution"],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_textbook_assets_are_reused_across_environments(tmp_path):
    corpus = tmp_path / "passages.jsonl"
    _write_corpus(corpus)
    cfg = KnowledgeAgentConfig(
        textbook_corpus_path=str(corpus),
        require_textbook_corpus=True,
        max_tool_calls=4,
    )
    left = KnowledgeAugmentedAgentEnv(config=cfg)
    right = KnowledgeAugmentedAgentEnv(config=cfg)
    assert left.textbook_store is right.textbook_store
    assert left.textbook_retriever is right.textbook_retriever


def test_inference_budget_check_counts_supervision_calls():
    module = _load_infer_module()
    row = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "inspect_state"}},
                    {"id": "b", "function": {"name": "finish_trace"}},
                ],
            }
        ]
    }
    assert module._required_tool_calls(row) == 2
