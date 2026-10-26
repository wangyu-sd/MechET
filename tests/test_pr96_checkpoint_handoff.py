"""CPU handoff safety: exact task identity and complete recovery artifacts."""
from copy import deepcopy
import json

import pytest
from scripts.replace_pr96_after_checkpoint import sole_source_instance, find_complete_anchor


def test_handoff_refuses_to_stop_a_task_with_another_active_instance():
    one, other = "a" * 32, "b" * 32
    line = lambda ident, state: f"| 2026-10-23 08:13:33 | {ident} | false | {state} | 1 | 2026-10-23 08:14:00 |"
    assert sole_source_instance(line(one, "TRAINING_RUNNING"), one)["state"] == "TRAINING_RUNNING"
    with pytest.raises(ValueError, match="EXACT_SOURCE"):
        sole_source_instance(line(other, "TRAINING_RUNNING"), one)
    with pytest.raises(ValueError, match="EXACT_SOURCE"):
        sole_source_instance(line(one, "TRAINING_RUNNING") + "\n" + line(other, "TRAINING_RUNNING"), one)
    assert sole_source_instance(line(one, "TRAINING_RUNNING") + "\n" + line(other, "END"), one)


def test_handoff_waits_for_manifest_then_checks_real_files_and_original_identity(tmp_path):
    from test_pr96_artifact_gates import checkpoint_fixture
    from mechet.pr96_performance import identity_sha256
    assert find_complete_anchor(tmp_path, "expected") is None
    cfg, checkpoint = checkpoint_fixture(tmp_path)
    value = json.loads((checkpoint / "pr96_checkpoint_manifest.json").read_text())
    expected = identity_sha256(value["lineage"]["training_identity"])
    assert find_complete_anchor(tmp_path, expected) == checkpoint
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_CHANGED"):
        find_complete_anchor(tmp_path, "wrong")
    (checkpoint / "optimizer.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="COMPONENT_HASH"):
        find_complete_anchor(tmp_path, expected)


def test_pipeline_resumes_authorized_anchor_not_a_newer_incomplete_old_directory(tmp_path):
    from test_pr96_artifact_gates import checkpoint_fixture
    from mechet.pr96_performance import select_performance_resume_checkpoint
    cfg, anchor = checkpoint_fixture(tmp_path)
    cfg["output_dir"] = str(tmp_path)
    cfg["pr96"]["performance_resume_anchor"] = str(anchor)
    (tmp_path / "checkpoint-13").mkdir()
    assert select_performance_resume_checkpoint(cfg) == anchor


def test_pipeline_prefers_complete_optimized_checkpoints_after_a_later_restart(tmp_path):
    import shutil
    from test_pr96_artifact_gates import checkpoint_fixture
    from mechet.pr96_performance import select_performance_resume_checkpoint
    from mechet.pr96_eval_runtime import sha
    cfg, anchor = checkpoint_fixture(tmp_path)
    cfg["output_dir"] = str(tmp_path)
    cfg["pr96"]["performance_resume_anchor"] = str(anchor)
    latest = tmp_path / "checkpoint-24"
    shutil.copytree(anchor, latest)
    state = {"global_step": 24, "epoch": 2}
    (latest / "trainer_state.json").write_text(json.dumps(state))
    value = json.loads((latest / "pr96_checkpoint_manifest.json").read_text())
    value.update(step=24, epoch=2)
    value["files"]["trainer_state.json"] = sha(latest / "trainer_state.json")
    (latest / "pr96_checkpoint_manifest.json").write_text(json.dumps(value))
    (tmp_path / "checkpoint-25").mkdir()
    assert select_performance_resume_checkpoint(cfg) == latest
    # Normal save_total_limit rotation may remove the old migration anchor.
    shutil.rmtree(anchor)
    assert select_performance_resume_checkpoint(cfg) == latest
    (latest / "optimizer.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="COMPONENT_HASH"):
        select_performance_resume_checkpoint(cfg)
