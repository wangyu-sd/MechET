import json

import pytest

from scripts import eval_natural_language_event_local as local_eval
from scripts.eval_natural_language_event_local import validate_adapter_lineage


def test_local_and_suffix_evaluation_require_matching_frozen_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "base_model": "Qwen/Qwen3-0.6B",
                "base_model_revision": "frozen-revision",
            }
        )
    )
    validate_adapter_lineage(adapter, "Qwen/Qwen3-0.6B", "frozen-revision")
    with pytest.raises(ValueError, match="base model"):
        validate_adapter_lineage(adapter, "Qwen/Qwen3-8B", "frozen-revision")
    with pytest.raises(ValueError, match="revision"):
        validate_adapter_lineage(adapter, "Qwen/Qwen3-0.6B", "wrong-revision")


def test_stage_two_local_eval_uses_frozen_history_prompts_in_decision_order(monkeypatch):
    source = {
        "id": "source-1",
        "source_id": "reaction-1",
        "metadata": {"trace_plan": {"steps": [{}]}},
    }
    monkeypatch.setattr(local_eval, "stratified_sample", lambda rows, size, seed: [source])
    monkeypatch.setattr(
        local_eval,
        "_private_states",
        lambda row: [
            {"decision_type": "import", "private_state": "C", "reference_successor": "", "event_depth": 0},
            {"decision_type": "event", "private_state": "C.O", "reference_successor": "CO", "event_depth": 1},
        ],
    )

    def decision(index, kind):
        return {
            "id": f"decision-{index}::history_v2",
            "source_id": "reaction-1",
            "metadata": {"decision_index": index, "decision_type": kind},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"TRAJECTORY HISTORY: accepted={index}"},
                {"role": "assistant", "tool_calls": [{"function": {"name": kind, "arguments": {}}}]},
            ],
            "tools": [],
        }

    tasks, reaction_ids = local_eval.collect_tasks(
        [source], sample_reactions=1, seed=17,
        decision_rows=[decision(1, "event"), decision(0, "import")],
    )
    assert reaction_ids == ["reaction-1"]
    assert [task["key"] for task in tasks] == [
        "decision-0::history_v2", "decision-1::history_v2"
    ]
    assert [task["messages"][1]["content"] for task in tasks] == [
        "TRAJECTORY HISTORY: accepted=0", "TRAJECTORY HISTORY: accepted=1"
    ]
