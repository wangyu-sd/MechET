import hashlib
import json

import pytest

from scripts.rebind_identical_tokenizer_cache import rebind_cache


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_cache(tmp_path):
    old, new = tmp_path / "old-tokenizer", tmp_path / "new-tokenizer"
    old.mkdir()
    new.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (old / name).write_text(name)
        (new / name).write_text(name)
    train, valid = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    train.write_text('{"id":"a"}\n')
    valid.write_text('{"id":"b"}\n')
    source = tmp_path / "source-cache"
    source.mkdir()
    for name in ("train.rank00.arrow", "validation.rank00.arrow"):
        (source / name).write_bytes(b"frozen arrow")
    manifest = {
        "model_name_or_path": "Qwen/Qwen3-8B",
        "model_revision": "old-revision",
        "max_length": 4096,
        "sources": {
            "train": {"bytes": train.stat().st_size, "sha256": digest(train)},
            "validation": {"bytes": valid.stat().st_size, "sha256": digest(valid)},
        },
        "splits": {
            "train": {
                "n_rows": 1,
                "truncation_count": 0,
                "arrow_files": [str(source / "train.rank00.arrow")],
            },
            "validation": {
                "n_rows": 1,
                "truncation_count": 0,
                "arrow_files": [str(source / "validation.rank00.arrow")],
            },
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    return {
        "source_cache": source,
        "target_cache": tmp_path / "target-cache",
        "source_tokenizer": old,
        "target_tokenizer": new,
        "source_model": "Qwen/Qwen3-8B",
        "source_revision": "old-revision",
        "target_model": "Qwen/Qwen3-0.6B",
        "target_revision": "new-revision",
        "expected_sources": {
            "train": (train, digest(train)),
            "validation": (valid, digest(valid)),
        },
        "max_length": 4096,
    }


def test_rebind_identical_tokenizer_cache_preserves_arrow_and_provenance(tmp_path):
    args = fixture_cache(tmp_path)
    rebound = rebind_cache(**args)
    target = args["target_cache"]
    assert rebound["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert rebound["model_revision"] == "new-revision"
    assert rebound["sources"]["train"]["sha256"] == digest(
        args["expected_sources"]["train"][0]
    )
    assert (target / "train.rank00.arrow").is_symlink()
    assert (target / "train.rank00.arrow").read_bytes() == b"frozen arrow"
    assert json.loads((target / "manifest.json").read_text()) == rebound
    with pytest.raises(FileExistsError):
        rebind_cache(**args)


def test_rebind_rejects_different_tokenizer_without_creating_target(tmp_path):
    args = fixture_cache(tmp_path)
    (args["target_tokenizer"] / "tokenizer.json").write_text("different")
    with pytest.raises(ValueError, match="tokenizer file differs"):
        rebind_cache(**args)
    assert not args["target_cache"].exists()


def test_rebind_rejects_modified_source_without_creating_target(tmp_path):
    args = fixture_cache(tmp_path)
    args["expected_sources"]["train"][0].write_text('{"id":"changed"}\n')
    with pytest.raises(ValueError, match="source file missing or size mismatch|source bytes differ"):
        rebind_cache(**args)
    assert not args["target_cache"].exists()
