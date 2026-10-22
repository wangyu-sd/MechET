import hashlib
from pathlib import Path

import pytest


def test_model_pin_rejects_mutable_revision_and_tampered_official_weights(tmp_path):
    from scripts.prepare_pr96_model import verify_snapshot
    from mechet.qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION
    p = tmp_path / "model.safetensors-00001-of-00001.safetensors"
    p.write_bytes(b"abc")
    metadata = {"id": OFFICIAL_MODEL, "sha": OFFICIAL_REVISION, "siblings": [
        {"rfilename": p.name, "size": 3, "lfs_sha256": hashlib.sha256(b"abc").hexdigest()}]}
    result = verify_snapshot(tmp_path, metadata, required_files=[p.name])
    assert result["files"][p.name]["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    metadata["sha"] = "main"
    with pytest.raises(ValueError, match="REVISION"):
        verify_snapshot(tmp_path, metadata, required_files=[p.name])
    metadata["sha"] = OFFICIAL_REVISION
    p.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="HASH"):
        verify_snapshot(tmp_path, metadata, required_files=[p.name])


def test_model_pin_rejects_a_different_repository_even_with_same_shaped_config(tmp_path):
    from scripts.prepare_pr96_model import verify_snapshot
    from mechet.qwen35_protocol import OFFICIAL_REVISION
    with pytest.raises(ValueError, match="MODEL_ID"):
        verify_snapshot(tmp_path, {"id": "Qwen/Qwen3-0.6B", "sha": OFFICIAL_REVISION, "siblings": []}, required_files=[])
