"""Permanent platform submission rules apply to PR69 jobs as well."""

from pathlib import Path

import pytest

from scripts.submit_taiji_with_donor_init import validate_submission_policy


def test_meteor_name_heartbeat_and_default_logs_are_required() -> None:
    config = {
        "task_flag": "meteor_mechet_pr69_test",
        "readable_name": "meteor MechET PR69 test",
        "start_cmd": "bash scripts/taiji_run_with_heartbeat.sh python -u task.py",
    }
    validate_submission_policy(config, Path("test.json"))
    for key, value in (
        ("task_flag", "mechet_pr69_test"),
        ("readable_name", "MechET PR69 test"),
        ("start_cmd", "python -u task.py"),
        ("start_cmd", "bash scripts/taiji_run_with_heartbeat.sh python task.py > task.log"),
    ):
        changed = {**config, key: value}
        with pytest.raises(ValueError):
            validate_submission_policy(changed, Path("test.json"))
