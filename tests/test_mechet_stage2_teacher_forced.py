"""CPU-only regression gates for the frozen Stage-II teacher-forced evaluator."""
from __future__ import annotations

import argparse
import json

import pytest

from mechet.forward_expert import verify_electron_step
from mechet.natural_language_electron_flow import build_inventory
from scripts.eval_mechet_stage2_teacher_forced import (
    assert_visible_private_parity, prepare, prepare_task_rows, read_jsonl,
)


def _source(corpus: str, split: str = "valid") -> dict:
    state = "[O:1]=[C:2]([OH:3])[CH3:4].[O-:5][CH2:6][CH3:7]"
    # Same authentic coupled arrow fixture already tested by the local evaluator.
    moves = [
        {
            "source": {"kind": "BOND", "atoms": [1, 2]},
            "sink": {"kind": "ATOM", "atoms": [1]},
            "electrons": 2,
        },
        {
            "source": {"kind": "LP", "atoms": [5]},
            "sink": {"kind": "BOND", "atoms": [2, 5]},
            "electrons": 2,
        },
    ]
    result = verify_electron_step(state, moves)
    assert result["ok"], result
    successor = result["state_smiles"]
    return {
        "id": f"{corpus}:{split}:small",
        "source_id": f"{corpus}:{split}:small",
        "target_smiles": state,
        "full_precursor_state": successor,
        "expected_precursor": successor,
        "messages": [],
        "metadata": {
            "source_dataset": corpus,
            "source_split": split,
            "trace_plan": {
                "target_smiles": state,
                "expected_precursor": successor,
                "steps": [{
                    "state_before": state, "state_after": successor,
                    "moves": moves, "imports": [],
                }],
            },
        },
    }


def test_private_state_parity_is_not_reconstructed_from_mapless_smiles():
    mapped = "[CH3:1][OH:2]"
    annotated = build_inventory(mapped)
    prompt = (
        f"CURRENT STATE SMILES: {annotated.visible_smiles}\n"
        f"{annotated.prompt}\n"
    )
    assert_visible_private_parity(prompt, mapped)
    with pytest.raises(ValueError, match="private state differs"):
        assert_visible_private_parity(prompt, "[CH3:1][NH2:2]")


@pytest.mark.parametrize("corpus", ["flower_mech_proof", "mech_uspto_31k"])
def test_gold_state_next_action_replays_reference_with_stage2_history(corpus):
    tasks = prepare_task_rows(_source(corpus))
    assert [row["decision_type"] for row in tasks] == ["event", "finish"]
    assert all("TRAJECTORY HISTORY" in row["messages"][1]["content"] for row in tasks)
    assert tasks[0]["gold_name"] == "apply_electron_flow"
    assert tasks[1]["gold_name"] == "finish_trace"
    assert tasks[0]["reference_successor"]
    assert not tasks[1]["reference_successor"]
    assert all("expected_precursor" not in json.dumps(t["messages"]) for t in tasks)
    assert all("reference_successor" not in json.dumps(t["messages"]) for t in tasks)
    assert tasks[0]["event_depth"] == 1


@pytest.mark.parametrize("corpus,source_dataset", [
    ("flower_new_dataset", "flower_mech_proof"),
    ("mech_uspto_31k", "mech_uspto_31k"),
])
def test_prepare_works_for_both_corpora_without_gpu(tmp_path, corpus, source_dataset):
    file = tmp_path / "valid.jsonl"
    file.write_text(json.dumps(_source(source_dataset)) + "\n")
    destination = tmp_path / f"prepared_{corpus}"
    args = argparse.Namespace(
        data=file, corpus=corpus, expected_source_rows=1,
        sample_reactions=0, seed=17, output=destination,
    )
    assert prepare(args) == 0
    tasks = read_jsonl(destination / "tasks.jsonl")
    assert len(tasks) == 2
    assert json.loads((destination / "selection.json").read_text())[
        "gold_executor_parity"
    ] == "passed_all_selected_FLOW"


def test_test_and_train_rows_cannot_be_scored_as_validation(tmp_path):
    file = tmp_path / "test.jsonl"
    file.write_text(json.dumps(_source("mech_uspto_31k", split="test")) + "\n")
    args = argparse.Namespace(
        data=file, corpus="mech_uspto_31k", expected_source_rows=1,
        sample_reactions=0, seed=17, output=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="TRAIN/TEST"):
        prepare(args)


def test_fail_closed_on_reference_successor_mismatch():
    source = _source("flower_mech_proof")
    source["metadata"]["trace_plan"]["steps"][0]["state_after"] = source["target_smiles"]
    with pytest.raises(ValueError):
        prepare_task_rows(source)
