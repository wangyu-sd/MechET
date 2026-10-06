import json

import pytest

from scripts.smoke_reliable_stage2_parent import history_row


def test_stage2_smoke_selects_decision_with_actual_accepted_history(tmp_path):
    source = tmp_path / "history.jsonl"
    rows = [
        {"id": "r::history_v2", "messages": [{"role": "user", "content": "TRAJECTORY HISTORY\naccepted_actions: 0"}]},
        {"id": "r2::history_v2", "messages": [{"role": "user", "content": "TRAJECTORY HISTORY\naccepted_actions: 1"}]},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert history_row(source)["id"] == "r2::history_v2"


def test_stage2_smoke_rejects_unconditioned_first_decision(tmp_path):
    source = tmp_path / "history.jsonl"
    source.write_text(
        json.dumps({"id": "r::history_v2", "messages": [{"role": "user", "content": "TRAJECTORY HISTORY\naccepted_actions: 0"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one accepted historical action"):
        history_row(source)
