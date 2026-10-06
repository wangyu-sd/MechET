import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_natural_language_event_local as local_eval
from eval_natural_language_event_local import _import_role_summary, score_prediction
from mechet.natural_language_electron_flow import render_event_arguments
from scripts.build_natural_language_event_sft import _decision_row


def test_event_prediction_compiles_executes_and_matches_successor() -> None:
    state = "[O:1]=[C:2]([OH:3])[CH3:4].[O-:5][CH2:6][CH3:7]"
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
    arguments = render_event_arguments(state, moves)
    from mechet.forward_expert import verify_electron_step

    successor = verify_electron_step(state, moves)["state_smiles"]
    task = {
        "decision_type": "event",
        "gold_name": "apply_electron_flow",
        "gold_arguments": arguments,
        "private_state": state,
        "reference_successor": successor,
    }
    metrics = score_prediction(
        task,
        predicted_name="apply_electron_flow",
        predicted_arguments=arguments,
    )
    assert metrics["event_exact"] is True
    assert metrics["formal_execute"] is True
    assert metrics["successor_map_exact"] is True
    assert metrics["successor_chemical_exact"] is True


def test_import_scoring_is_order_invariant_but_checks_schedule() -> None:
    gold = {
        "fragments": [
            {"smiles": "[Na+]", "count": 1, "purpose": "endpoint_context"},
            {"smiles": "[OH-]", "count": 2, "purpose": "electron_participant"},
        ]
    }
    predicted = {"fragments": list(reversed(gold["fragments"]))}
    task = {
        "decision_type": "import",
        "gold_name": "import_fragments",
        "gold_arguments": gold,
    }
    metrics = score_prediction(
        task,
        predicted_name="import_fragments",
        predicted_arguments=predicted,
    )
    assert metrics["import_fragment_exact"] is True
    assert metrics["import_schedule_exact"] is True
    assert metrics["decision_exact"] is True
    assert metrics["import_participant_exact"] is True
    assert metrics["import_context_exact"] is True
    assert metrics["gold_import_participant_copies"] == 2
    assert metrics["gold_import_context_copies"] == 1


def test_import_role_breakdown_separates_participant_and_context_errors() -> None:
    gold = {"fragments": [
        {"smiles": "[Cl-]", "count": 1, "purpose": "electron_participant"},
        {"smiles": "[Na+]", "count": 1, "purpose": "endpoint_context"},
    ]}
    task = {
        "decision_type": "import", "gold_name": "import_fragments",
        "gold_arguments": gold,
    }
    prediction = {"fragments": [
        gold["fragments"][0],
        {"smiles": "[K+]", "count": 1, "purpose": "endpoint_context"},
    ]}
    metrics = score_prediction(
        task, predicted_name="import_fragments", predicted_arguments=prediction,
    )
    assert metrics["import_participant_exact"] is True
    assert metrics["import_context_exact"] is False
    assert metrics["import_fragment_exact"] is False
    assert metrics["import_schedule_exact"] is False
    wrong_tool = score_prediction(
        task, predicted_name="finish_trace", predicted_arguments={},
    )
    assert wrong_tool["gold_import_participant_copies"] == 1
    assert wrong_tool["gold_import_context_copies"] == 1
    assert wrong_tool["import_participant_exact"] is False
    assert wrong_tool["import_context_exact"] is False
    summary = _import_role_summary([metrics, wrong_tool])
    assert summary["n"] == 2
    assert summary["import_participant_exact_n"] == 2
    assert summary["import_context_exact_n"] == 2
    assert summary["import_participant_exact"] == 1
    assert summary["import_context_exact"] == 0


def test_role_breakdown_aggregation_uses_role_present_denominators(
    monkeypatch, tmp_path: Path,
) -> None:
    source = tmp_path / "valid.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test")
    output = tmp_path / "evaluation"
    output.mkdir()
    mixed = {"decision_type": "import", "gold_name": "import_fragments",
             "gold_arguments": {"fragments": [
                 {"smiles": "[Cl-]", "count": 1, "purpose": "electron_participant"},
                 {"smiles": "[Na+]", "count": 1, "purpose": "endpoint_context"},
             ]}}
    participant = {"decision_type": "import", "gold_name": "import_fragments",
                   "gold_arguments": {"fragments": [
                       {"smiles": "O", "count": 1, "purpose": "electron_participant"},
                   ]}}
    records = []
    for key, task, prediction in (
        ("a", mixed, {"fragments": [
            mixed["gold_arguments"]["fragments"][0],
            {"smiles": "[K+]", "count": 1, "purpose": "endpoint_context"},
        ]}),
        ("b", participant, participant["gold_arguments"]),
    ):
        metrics = score_prediction(
            task, predicted_name="import_fragments", predicted_arguments=prediction,
        )
        records.append({"key": key, "decision_type": "import",
                        "generated_tokens": 1, "generation_error": "", **metrics})
    (output / "decisions.shard-00-of-01.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8",
    )
    monkeypatch.setattr(
        local_eval, "validate_adapter_lineage",
        lambda *_args, **_kwargs: {"kind": "completed_adapter"},
    )
    monkeypatch.setattr(
        local_eval, "collect_tasks",
        lambda *_args, **_kwargs: ([{"key": "a"}, {"key": "b"}], ["r"]),
    )
    args = SimpleNamespace(
        adapter=adapter, model="test", model_revision="test",
        provisional_training_config=None, data=source, decision_data=None,
        output=output, sample_reactions=1, seed=17, batch_size=2,
        sft_aligned_prefix=True, dtype="bfloat16", no_4bit=True,
        import_role_breakdown=True,
    )
    assert local_eval.aggregate(args) == 0
    report = json.loads((output / "evaluation.json").read_text())
    assert report["decode_batch_size"] == 2
    by_role = report["by_import_role"]
    assert by_role["participant_present"]["n"] == 2
    assert by_role["participant_present"]["import_participant_exact_n"] == 2
    assert by_role["participant_present"]["import_participant_exact"] == 2
    assert by_role["context_present"]["n"] == 1
    assert by_role["context_present"]["import_context_exact_n"] == 1
    assert by_role["context_present"]["import_context_exact"] == 0
    assert by_role["mixed"]["n"] == 1
    assert by_role["participant_only"]["n"] == 1
    assert by_role["participant_only"]["import_context_exact_n"] == 0
    assert by_role["participant_only"]["import_context_exact_rate"] is None


def test_import_scoring_preserves_explicit_hydrogen_participants() -> None:
    gold = {
        "fragments": [
            {
                "smiles": "[H]OC(=O)[O-]",
                "count": 1,
                "purpose": "electron_participant",
            }
        ]
    }
    task = {
        "decision_type": "import",
        "gold_name": "import_fragments",
        "gold_arguments": gold,
    }
    exact = score_prediction(
        task, predicted_name="import_fragments", predicted_arguments=gold
    )
    collapsed = score_prediction(
        task,
        predicted_name="import_fragments",
        predicted_arguments={
            "fragments": [
                {
                    "smiles": "O=C([O-])O",
                    "count": 1,
                    "purpose": "electron_participant",
                }
            ]
        },
    )
    assert exact["import_fragment_exact"] is True
    assert collapsed["import_fragment_exact"] is False


def test_finish_requires_the_finish_tool_and_empty_arguments() -> None:
    task = {
        "decision_type": "finish",
        "gold_name": "finish_trace",
        "gold_arguments": {},
    }
    assert score_prediction(
        task, predicted_name="finish_trace", predicted_arguments={}
    )["finish_exact"]
    assert not score_prediction(
        task, predicted_name="import_fragments", predicted_arguments={}
    )["finish_exact"]


def test_every_action_type_uses_the_same_inventory_observation() -> None:
    source = {
        "source_id": "train_1",
        "target_smiles": "C",
        "expected_precursor": "C.O",
        "metadata": {},
    }
    for name, arguments in (
        (
            "import_fragments",
            {
                "fragments": [
                    {"smiles": "O", "count": 1, "purpose": "electron_participant"}
                ]
            },
        ),
        ("apply_electron_flow", {"direction": "retrosynthetic"}),
        ("finish_trace", {}),
    ):
        row = _decision_row(
            row=source,
            sequence_index=0,
            decision_type=name,
            mapped_state="[CH4:1]",
            name=name,
            arguments=arguments,
            result={"ok": True, "code": "PASS"},
        )
        prompt = row["messages"][1]["content"]
        assert "MOLECULAR INVENTORY" in prompt
        assert "ANNOTATED CURRENT STATE" in prompt
        assert row["metadata"]["decision_contract"] == "unified_inventory_tool_decision_v2"
