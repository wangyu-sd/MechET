import json

import pytest

from mechet.open_flow_program import parse_open_flow
from scripts.build_flower_a1_a4_a5_a6 import format_open_flow_program
from scripts.repair_nmi_open_flow_imports import repair_row


def _row(*, late_import=True):
    move = {"source": {"kind": "LP", "atoms": [1]},
            "sink": {"kind": "BOND", "atoms": [1, 2]}, "electrons": 2}
    plan = {"initial_imports": ["[OH:1]"],
            "steps": [{"moves": [move], "imports": ["[H:3]"] if late_import else []}]}
    old = "<flow>\nOPEN_FLOW v1\nIMPORT [OH:1]\nSTEP 0 " + json.dumps(
        [move], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ) + "\nEXECUTE\n</flow>"
    return {"source_id": "r1", "metadata": {"trace_plan": plan},
            "messages": [{"role": "assistant", "content": old}]}


def test_repair_declares_late_import_without_changing_electron_moves():
    row, changed = repair_row(_row())
    assert changed is True
    imports, steps = parse_open_flow(row["messages"][0]["content"])
    assert imports == ["[OH:1]", "[H:3]"]
    assert steps == [row["metadata"]["trace_plan"]["steps"][0]["moves"]]


def test_formatter_and_repair_keep_rows_without_late_import_byte_semantics():
    row = _row(late_import=False)
    assert format_open_flow_program(row["metadata"]["trace_plan"]) == row["messages"][0]["content"]
    repaired, changed = repair_row(row)
    assert changed is False
    assert repaired["messages"][0]["content"] == row["messages"][0]["content"]


def test_repair_rejects_changed_step_targets():
    row = _row()
    row["metadata"]["trace_plan"]["steps"][0]["moves"][0]["electrons"] = 1
    with pytest.raises(ValueError, match="changed electron moves"):
        repair_row(row)
