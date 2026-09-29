"""The R3 oracle audit replays actions, not textually accepted answers."""

import pytest

from scripts.autoresearch.audit_r3_oracle_repair import (
    replay_repaired, require_frozen_rdkit,
)


def _row(action: dict) -> dict:
    return {"model_visible": {"target_smiles": "CC", "prefix_actions": []},
            "private_reference": {"first_failure_index": 0,
                                  "correct_action": action,
                                  "suffix_actions": [],
                                  "expected_successor": "CC",
                                  "expected_precursor": "CC"}}


def test_r3_oracle_replay_accepts_executed_terminal() -> None:
    assert replay_repaired(_row({"name": "finish_trace", "arguments": {}}),
                           "[CH3:1][CH3:2]")["exact"]


def test_r3_oracle_replay_rejects_invalid_tool() -> None:
    result = replay_repaired(_row({"name": "bad_tool", "arguments": {}}),
                             "[CH3:1][CH3:2]")
    assert not result["exact"]
    assert result["failure_index"] == 0


def test_r3_oracle_audit_rejects_rdkit_environment_drift() -> None:
    require_frozen_rdkit("2026.03.4")
    with pytest.raises(RuntimeError, match="requires RDKit 2026.03.4"):
        require_frozen_rdkit("2024.09.6")
