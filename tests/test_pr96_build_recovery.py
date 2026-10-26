"""Storage recovery must never duplicate data, alter old files, or trust flags."""
from copy import deepcopy
import errno
import hashlib
import json
import os

import pytest


def test_multi_journal_commit_rolls_back_partial_eio_before_retry_without_duplicates(tmp_path, monkeypatch):
    from mechet.pr96_durable_io import DurableAppendSet
    paths = {k: tmp_path / f"{k}.jsonl" for k in ("inverse", "state")}
    events = []
    with DurableAppendSet(paths, on_retry=events.append) as writer:
        writer.commit({"inverse": b"old-inverse\n", "state": b"old-state\n"})
        real_sync = os.fsync
        failed = False

        def fail_once(fd):
            nonlocal failed
            # Error after both files have received bytes, as in the real flush failure.
            if not failed and os.readlink(f"/proc/self/fd/{fd}") == str(paths["state"]):
                failed = True
                raise OSError(errno.EIO, "injected shared-disk failure")
            return real_sync(fd)

        monkeypatch.setattr(os, "fsync", fail_once)
        offsets = writer.commit({"inverse": b"new-inverse\n", "state": b"new-state\n"})
    assert paths["inverse"].read_bytes() == b"old-inverse\nnew-inverse\n"
    assert paths["state"].read_bytes() == b"old-state\nnew-state\n"
    assert offsets == {"inverse": 24, "state": 20}
    assert len(events) == 1 and events[0]["errno"] == errno.EIO


def test_durable_append_handles_short_writes_and_refuses_existing_paths(tmp_path, monkeypatch):
    from mechet.pr96_durable_io import DurableAppendSet
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, data[:3]))
    path = tmp_path / "new.jsonl"
    with DurableAppendSet({"out": path}) as writer:
        writer.commit({"out": b"complete-line\n"})
    assert path.read_bytes() == b"complete-line\n"
    with pytest.raises(FileExistsError):
        DurableAppendSet({"out": path})
    assert path.read_bytes() == b"complete-line\n"


def test_retry_exhaustion_preserves_last_committed_prefix_and_does_not_retry_permission_error(tmp_path, monkeypatch):
    from mechet.pr96_durable_io import DurableAppendSet
    events = []
    with DurableAppendSet({"out": tmp_path / "out"}, max_attempts=2, on_retry=events.append) as writer:
        writer.commit({"out": b"committed\n"})
        def bad_sync(fd):
            raise OSError(errno.EIO, "persistent disk error")
        monkeypatch.setattr(os, "fsync", bad_sync)
        with pytest.raises(OSError, match="persistent disk"):
            writer.commit({"out": b"uncommitted\n"})
        assert (tmp_path / "out").read_bytes() == b"committed\n"
    assert len(events) == 1
    events.clear()
    with DurableAppendSet({"out": tmp_path / "permission"}, on_retry=events.append) as writer:
        def denied(fd):
            raise PermissionError(errno.EACCES, "permission denied")
        monkeypatch.setattr(os, "fsync", denied)
        with pytest.raises(PermissionError):
            writer.commit({"out": b"not-committed\n"})
    assert not events


def saved_prefix(tmp_path):
    from mechet.inverse_trace_data import build_inverse_tool_sft_row
    from scripts.build_main_product_training_data import prepare_reference, audit_reference
    from test_mech31k_main_product_root import forward_sn2
    source = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")
    accepted = audit_reference(prepare_reference(source))
    bad_source = deepcopy(source)
    bad_source["source_id"] = "bad-source"
    digest = hashlib.sha256(json.dumps(bad_source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bad = {"source_id": "bad-source", "error": "ValueError:known-reference-failure", "split": "train",
           "source_row": bad_source, "source_row_sha256": digest, "training_allowed": False}
    old = tmp_path / "old"
    old.mkdir()
    files = {"inverse": [accepted["inverse"]], "audits": [accepted["audit"]],
             "state_sft": accepted["decisions"], "unresolved": [bad]}
    paths = {}
    for k, rows in files.items():
        paths[k] = old / k
        # A partial corrupt tail may exist after the last progress-committed batch.
        paths[k].write_bytes(b"".join((json.dumps(r) + "\n").encode() for r in rows) + b'{"partial')
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("".join(json.dumps(r) + "\n" for r in (source, bad_source, source)))
    progress = {"processed": 2, "reference_pass": 1, "unresolved": 1, "counts": {
        "import_decisions": 1, "event_decisions": 1, "finish_decisions": 1,
        "first_import_fragments": 1, "error:known-reference-failure": 1}}
    return source_path, paths, progress


def test_recover_only_complete_verified_source_prefix_preserves_originals_and_all_decisions(tmp_path):
    from mechet.pr96_durable_io import DurableAppendSet
    from scripts.pr96_build_recovery import recover_prefix
    source, paths, progress = saved_prefix(tmp_path)
    before = {k: p.read_bytes() for k, p in paths.items()}
    outputs = {k: tmp_path / f"new-{k}" for k in paths}
    events = []
    with DurableAppendSet(outputs) as writer, source.open() as reader:
        accepted, unresolved, counts = recover_prefix(reader, paths, progress, writer, events.append)
        assert json.loads(next(reader))["source_id"] != "bad-source"
    assert accepted == {"sn2_main_only"} and unresolved == {"bad-source"}
    assert counts == progress["counts"]
    assert len(outputs["state_sft"].read_text().splitlines()) == 3
    assert all(b'partial' not in p.read_bytes() for p in outputs.values())
    assert all(p.read_bytes() == before[k] for k, p in paths.items())
    assert events[-1]["processed"] == 2


def test_prefix_recovery_rejects_changed_source_hash(tmp_path):
    from mechet.pr96_durable_io import DurableAppendSet
    from scripts.pr96_build_recovery import recover_prefix
    source, paths, progress = saved_prefix(tmp_path)
    rows = source.read_text().splitlines()
    value = json.loads(rows[0]); value["full_precursor_state"] = "wrong"
    source.write_text(json.dumps(value) + "\n" + "\n".join(rows[1:]) + "\n")
    with DurableAppendSet({k: tmp_path / f"bad-{k}" for k in paths}) as writer, source.open() as reader:
        with pytest.raises(ValueError, match="SOURCE_ID_OR_HASH_CHANGED"):
            recover_prefix(reader, paths, progress, writer, lambda event: None)


def test_prefix_recovery_rejects_missing_finish_and_never_consumes_partial_tail(tmp_path):
    from mechet.pr96_durable_io import DurableAppendSet
    from scripts.pr96_build_recovery import recover_prefix
    source, paths, progress = saved_prefix(tmp_path)
    lines = paths["state_sft"].read_bytes().splitlines(keepends=True)
    paths["state_sft"].write_bytes(b"".join(lines[:2]) + b'{"partial')
    with DurableAppendSet({k: tmp_path / f"bad-{k}" for k in paths}) as writer, source.open() as reader:
        with pytest.raises(ValueError, match="JOURNAL_INCOMPLETE"):
            recover_prefix(reader, paths, progress, writer, lambda event: None)


def test_builder_recovers_bound_prefix_and_completed_splits_then_processes_every_remaining_source(tmp_path, monkeypatch):
    import shutil
    from scripts import build_main_product_training_data as builder
    source_path, old_paths, progress = saved_prefix(tmp_path)
    source_dir, old, output = tmp_path / "source", tmp_path / "interrupted", tmp_path / "recovered"
    source_dir.mkdir(); old.mkdir()
    rows = [json.loads(l) for l in source_path.read_text().splitlines()]
    rows[-1]["source_id"] = "last-train"
    sources = {"train": rows}
    for split in ("valid", "test"):
        row = deepcopy(rows[0]); row["source_id"] = split + "-one"
        sources[split] = [row]
    manifest = {"splits": {}}
    for split, values in sources.items():
        path = source_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in values))
        manifest["splits"][split] = {"rows": len(values), "sha256": builder.sha(path)}
    (source_dir / "training_manifest.json").write_text(json.dumps(manifest))
    (source_dir / "ARTIFACT_STATUS.json").write_text('{"training_allowed": true}')
    for kind in ("inverse", "state_sft", "audits"):
        (old / kind).mkdir()
    for kind, path in old_paths.items():
        target = old / ("audits/train.unresolved.jsonl" if kind == "unresolved" else kind + "/train.jsonl")
        shutil.copy2(path, target)
    for split in ("valid", "test"):
        result = builder.audit_reference(builder.prepare_reference(sources[split][0]))
        values = {"inverse": [result["inverse"]], "state_sft": result["decisions"], "audits": [result["audit"]]}
        reports = {"original_count": 1, "reference_pass": 1, "unresolved": 0, "coverage": 1.0,
                   "accepted_ids": [split + "-one"], "unresolved_ids": [],
                   "source_sha256": manifest["splits"][split]["sha256"], "outputs": {},
                   "counts": {"import_decisions": 1, "event_decisions": 1, "finish_decisions": 1,
                              "first_import_fragments": 1}}
        for kind, records in values.items():
            path = old / kind / f"{split}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            reports["outputs"][kind] = {"path": str(path), "sha256": builder.sha(path)}
        bad = old / "audits" / f"{split}.unresolved.jsonl"; bad.write_text("")
        reports.update(unresolved_path=str(bad), unresolved_sha256=builder.sha(bad))
        (old / f"{split}.report.json").write_text(json.dumps(reports))
    (old / "progress.jsonl").write_text(json.dumps(dict(progress, stage="reference-replay", split="train")) + "\n")
    code = ["scripts/build_main_product_training_data.py", "scripts/build_natural_language_event_sft.py",
            "scripts/run_natural_language_value_search.py", "scripts/build_mech31k_main_product_only_sft.py",
            "src/mechet/inverse_trace_data.py", "src/mechet/in_place_grounded_flow.py",
            "src/mechet/natural_language_electron_flow.py", "src/mechet/forward_expert.py",
            "scripts/pr96_source_product_side.py"]
    for name in code:
        path = old / "code_snapshot" / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(builder.ROOT / name, path)
    originals = {str(p.relative_to(old)): builder.sha(p) for p in old.rglob("*") if p.is_file()}
    monkeypatch.setitem(builder.DENOMINATORS, "flower", {"train": 3, "valid": 1, "test": 1})
    monkeypatch.setattr("sys.argv", ["build", "--track", "flower", "--source-dir", str(source_dir),
                                  "--output-dir", str(output), "--recover-from", str(old), "--workers", "1"])
    assert builder.main() == 0
    result = json.loads((output / "manifest.json").read_text())
    assert result["reports"]["train"]["original_count"] == 3
    assert result["reports"]["train"]["reference_pass"] == 2
    assert result["reports"]["train"]["unresolved"] == 1
    assert result["recovery"]["recovered_train_prefix"] == 2
    assert result["training_allowed"] is False
    assert result["split_id_overlap"] == {"train_valid": 0, "train_test": 0, "valid_test": 0}
    assert result["reports"]["valid"]["outputs"]["state_sft"]["sha256"] == builder.sha(old / "state_sft/valid.jsonl")
    assert all(builder.sha(old / name) == digest for name, digest in originals.items())
