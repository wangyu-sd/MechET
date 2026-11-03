from copy import deepcopy

import pytest

from test_mech31k_main_product_root import forward_sn2
from mechet.inverse_trace_data import build_inverse_tool_sft_row
from scripts.build_main_product_training_data import prepare_reference, audit_reference


def state_rows():
    source = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")
    return audit_reference(prepare_reference(source))["decisions"]


def test_history_is_reconstructed_from_actual_prior_accepted_actions_only():
    from scripts.build_pr96_history import reaction_history
    rows = state_rows()
    history = reaction_history(rows)
    assert "accepted_actions: 0" in history[0]["messages"][1]["content"]
    assert "accepted_actions: 1" in history[1]["messages"][1]["content"]
    assert "accepted_actions: 2" in history[2]["messages"][1]["content"]
    assert "electron_events_committed: 0" in history[1]["messages"][1]["content"]
    for original, converted in zip(rows, history):
        assert original["messages"][2:] == converted["messages"][2:]
        assert converted["metadata"]["online_history_prefix_parity"] is True


def test_history_conversion_rejects_unreachable_teacher_forced_state():
    from scripts.build_pr96_history import reaction_history
    rows = state_rows()
    rows[1]["messages"][1]["content"] = rows[1]["messages"][1]["content"].replace("CURRENT STATE SMILES: CO.[Br-]", "CURRENT STATE SMILES: CBr")
    with pytest.raises(ValueError, match="STATE_PREFIX"):
        reaction_history(rows)
