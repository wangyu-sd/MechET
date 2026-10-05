import pytest
import torch

from mechet.system_one_action_family import (
    ACTION_TO_INDEX,
    ActionFamilyHead,
    parse_action_atom_names,
    parse_action_family_example,
)
from scripts.train_system_one_action_family import metrics


@pytest.mark.parametrize(
    ("decision_type", "tool_name"),
    [
        ("event", "apply_electron_flow"),
        ("import", "import_fragments"),
        ("finish", "finish_trace"),
    ],
)
def test_action_family_parser_removes_answer_and_executor_result(decision_type, tool_name):
    row = {
        "id": f"rxn::decision_001_{decision_type}",
        "metadata": {"reaction_id": "rxn", "decision_type": decision_type,
                     "history_accepted_actions": 1},
        "tools": [],
        "messages": [
            {"role": "system", "content": "retrosynthesis"},
            {"role": "user", "content": "ANNOTATED CURRENT STATE: <A01>C<A02>O"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": tool_name, "arguments": {}
            }}]},
            {"role": "tool", "name": tool_name, "content": '{"ok":true}'},
        ],
    }
    example = parse_action_family_example(row)
    assert example.label == ACTION_TO_INDEX[tool_name]
    assert example.atom_names == ("A01", "A02")
    assert [msg["role"] for msg in example.messages] == ["system", "user"]
    assert example.history_accepted_actions == 1


def test_action_family_parser_rejects_contract_disagreement():
    row = {
        "id": "rxn::decision_001_import",
        "metadata": {"reaction_id": "rxn", "decision_type": "import",
                     "history_accepted_actions": 1},
        "tools": [],
        "messages": [
            {"role": "user", "content": "ANNOTATED CURRENT STATE: <A01>C"},
            {"role": "assistant", "tool_calls": [{"function": {
                "name": "finish_trace", "arguments": {}
            }}]},
        ],
    }
    with pytest.raises(ValueError, match="disagrees"):
        parse_action_family_example(row)


def test_action_family_head_scores_three_routes():
    head = ActionFamilyHead(16, width=8)
    scores = head(torch.zeros(4, 16))
    assert scores.shape == (4, 3)
    assert torch.isfinite(scores).all()


def test_action_inventory_preserves_explicit_hydrogen_atoms():
    names = parse_action_atom_names(
        "ANNOTATED CURRENT STATE: <A01>[H]<A02>O<A03>C", "finish-example"
    )
    assert names == ("A01", "A02", "A03")


def test_action_family_metrics_exposes_premature_finish_and_minority_recall():
    report = metrics([0, 1, 2, 0], [0, 2, 2, 0])
    assert report["accuracy"] == 0.75
    assert report["classes"]["import_fragments"]["recall"] == 0.0
    assert report["premature_finish"] == 1
    assert report["premature_finish_rate_among_nonfinish"] == 1 / 3
