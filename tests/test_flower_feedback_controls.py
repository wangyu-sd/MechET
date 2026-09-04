import importlib.util
import json
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_row():
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("inspect_state", "import_fragment", "finish_trace")
    ]
    return {
        "id": "r1",
        "source_id": "s1",
        "artifact_type": "supervision",
        "target_smiles": "[CH3:1][OH:2]",
        "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
        "expected_precursor": "[CH3:1][Br:3].[OH-:2]",
        "tools": tools,
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "Use inspect_state before referencing atom maps.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "import_fragment",
                            "arguments": {"fragment_smiles": "[Br-:3]"},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "import_fragment", "tool_call_id": "c1", "content": '{"state":1}'},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "function": {
                            "name": "import_fragment",
                            "arguments": {"fragment_smiles": "[Na+:4]"},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "import_fragment", "tool_call_id": "c2", "content": '{"state":2}'},
        ],
        "metadata": {},
    }


def test_b1_removes_enumeration_interface_and_instruction():
    module = _load("build_flower_feedback_controls.py")
    row = module.build_b1(_trace_row())
    assert "inspect_state" not in [item["function"]["name"] for item in row["tools"]]
    assert "Use inspect_state" not in row["messages"][1]["content"]


def test_b3_reuses_first_result_from_each_tool():
    module = _load("build_flower_feedback_controls.py")
    row = module.build_b3(_trace_row())
    results = [m["content"] for m in row["messages"] if m["role"] == "tool"]
    assert results == ['{"state":1}', '{"state":1}']
    assert row["metadata"]["stale_observation_replacements"] == 1


def test_b5_legal_summary_depends_only_on_product():
    module = _load("build_flower_feedback_controls.py")
    row = module.build_b5(_trace_row())
    assert row["task_type"] == "b5_direct_legal_actions"
    assert "LEGAL-ACTION SUMMARY" in row["messages"][1]["content"]
    assert row["structural_precursor"] not in row["messages"][1]["content"]
    assert json.loads(
        row["messages"][1]["content"].split("LEGAL-ACTION SUMMARY:\n", 1)[1]
    )["source"] == "deterministic_product_only_enumeration_v1"


class _FakeEnv:
    def __init__(self):
        self.state = {"finalized": True, "abstained": False, "final_result": {"formal_execute": True}}

    def _snapshot(self):
        return dict(self.state)


def test_a5_rollout_accepts_one_answer_after_terminal(monkeypatch):
    module = _load("infer_mechet.py")
    monkeypatch.setattr(module, "parse_tool_calls", lambda *args, **kwargs: [])
    rollout = module._TraceRollout(
        env=_FakeEnv(),
        messages=[],
        sample_index=0,
        seed=17,
        max_iterations=4,
        exchanges=[],
        allow_independent_answer=True,
    )
    rollout.awaiting_independent_answer = True
    rollout.advance("<answer>CCO</answer>", None, "", None)
    record = rollout.record()
    assert record["termination_reason"] == "independent_answer"
    assert record["prediction"] == "<answer>CCO</answer>"
