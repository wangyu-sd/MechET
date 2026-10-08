import json

import pytest

from scripts import run_earho_v2


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_full_executable_preparation_covers_every_reaction_once(tmp_path, monkeypatch):
    train = [{"id": f"train-{i}", "source_id": f"train-{i}"} for i in range(31)]
    valid = [{"id": "valid-0", "source_id": "valid-0"}]
    _write(tmp_path / "train.jsonl", train)
    _write(tmp_path / "valid.jsonl", valid)
    _write(tmp_path / "history.jsonl", [{"source_id": row["source_id"]} for row in train])
    _write(tmp_path / "valid_history.jsonl", [{"source_id": "valid-0"}])
    monkeypatch.setattr(run_earho_v2, "replay_reference", lambda *args, **kwargs: None)
    cfg = {
        "protocol_version": run_earho_v2.RELIABLE_PROTOCOL,
        "reaction_denominator": {"train": len(train)},
        "rounds": 4, "seed": 17,
        "train_file": str(tmp_path / "train.jsonl"),
        "validation_file": str(tmp_path / "valid.jsonl"),
        "history_file": str(tmp_path / "history.jsonl"),
        "history_validation_file": str(tmp_path / "valid_history.jsonl"),
        "validation_monitor_rows": 1,
        "rollout": {"max_imports": 32},
        "initial_adapter_model_sha256": "a" * 64,
    }
    output = tmp_path / "result"
    run_earho_v2._prepare_all_executable(cfg, output, "b" * 64)
    plan = json.loads((output / "plan.json").read_text())
    ids = []
    for index in range(4):
        rows = run_earho_v2.read_rows(output / f"round{index:02d}/source.jsonl")
        ids.extend(row["source_id"] for row in rows)
    assert len(ids) == len(set(ids)) == len(train)
    assert set(ids) == {row["source_id"] for row in train}
    assert sum(plan["round_counts"]) == len(train)
    assert plan["selected_train_reactions"] == len(train)
    assert len(plan["prepared_files"]) == 5


def test_full_executable_preparation_rejects_duplicate_source(tmp_path):
    _write(tmp_path / "train.jsonl", [
        {"id": "a", "source_id": "same"}, {"id": "b", "source_id": "same"},
    ])
    cfg = {
        "protocol_version": run_earho_v2.RELIABLE_PROTOCOL,
        "reaction_denominator": {"train": 2},
        "rounds": 1, "seed": 17,
        "train_file": str(tmp_path / "train.jsonl"),
    }
    with pytest.raises(ValueError, match="duplicate strict FlowER source_id"):
        run_earho_v2._prepare_all_executable(cfg, tmp_path / "result", "c" * 64)
