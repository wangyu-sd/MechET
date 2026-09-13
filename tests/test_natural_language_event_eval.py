from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_natural_language_event_local import score_prediction
from mechet.natural_language_electron_flow import render_event_arguments


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
