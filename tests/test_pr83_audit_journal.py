"""Resume verification for bounded-memory RL audit logs; no GPU jobs."""
import hashlib
from pathlib import Path

import pytest

from scripts import pr83_resume as resume


def journal(*args, **kwargs):
    factory = getattr(resume, "AuditJournal", None)
    if factory is None:
        pytest.fail("append-only audit journal is missing")
    return factory(*args, **kwargs)


def test_checkpoint_has_only_a_cursor_and_hash_not_the_growing_trace_history(tmp_path):
    log = journal(tmp_path / "groups.jsonl")
    log.append({"root": 0})
    state = log.checkpoint_state()
    assert state["offset"] == len(b'{"root":0}\n')
    assert state["sha256"] == hashlib.sha256(b'{"root":0}\n').hexdigest()
    log.close()


def test_resume_preserves_complete_prefix_and_quarantines_unsaved_tail(tmp_path):
    path = tmp_path / "groups.jsonl"
    log = journal(path)
    log.append({"root": 0})
    state = log.checkpoint_state()
    log.append({"root": 1})
    log.close()
    restored = journal(path, restored_state=state)
    restored.append({"root": 1, "retry": True})
    restored.close()
    assert path.read_bytes() == b'{"root":0}\n{"root":1,"retry":true}\n'
    tails = list(tmp_path.glob("groups.rollback-*.jsonl"))
    assert len(tails) == 1
    assert tails[0].read_bytes() == b'{"root":1}\n'


def test_mutated_prefix_is_not_truncated_or_accepted(tmp_path):
    path = tmp_path / "groups.jsonl"
    log = journal(path)
    log.append({"root": 0})
    state = log.checkpoint_state()
    log.close()
    path.write_bytes(b'{"root":9}\n{"root":1}\n')
    before = path.read_bytes()
    with pytest.raises(ValueError, match="hash"):
        journal(path, restored_state=state)
    assert path.read_bytes() == before


def test_fresh_run_cannot_overwrite_an_existing_journal(tmp_path):
    path = tmp_path / "groups.jsonl"
    path.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        journal(path)
    assert path.read_bytes() == b'preserve'
