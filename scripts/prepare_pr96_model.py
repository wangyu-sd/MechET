#!/usr/bin/env python3
"""Download the pinned official base and verify authoritative file fingerprints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION

FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "vocab.json", "merges.txt", "model.safetensors.index.json",
         "model.safetensors-00001-of-00001.safetensors", "LICENSE", "README.md"]


def verify_snapshot(snapshot, metadata, *, required_files=FILES):
    if metadata.get("id") != OFFICIAL_MODEL:
        raise ValueError("OFFICIAL_MODEL_ID_MISMATCH")
    if metadata.get("sha") != OFFICIAL_REVISION:
        raise ValueError("OFFICIAL_MODEL_REVISION_MISMATCH")
    siblings = {r["rfilename"]: r for r in metadata["siblings"]}
    files = {}
    for name in required_files:
        path = snapshot / name
        record = siblings[name]
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as reader:
            for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                blob.update(chunk)
        observed = digest.hexdigest()
        if path.stat().st_size != record["size"]:
            raise ValueError(f"OFFICIAL_FILE_SIZE_MISMATCH:{name}")
        expected = record.get("lfs_sha256")
        if expected and observed != expected:
            raise ValueError(f"OFFICIAL_FILE_HASH_MISMATCH:{name}")
        if not expected and record.get("blob_id") and blob.hexdigest() != record["blob_id"]:
            raise ValueError(f"OFFICIAL_GIT_BLOB_HASH_MISMATCH:{name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": observed,
                       "official_lfs_sha256": expected, "official_git_blob": record.get("blob_id")}
        print(json.dumps({"stage": "official-file-verified", "file": name, **files[name]}), flush=True)
    return {"model_id": OFFICIAL_MODEL, "revision": OFFICIAL_REVISION,
            "snapshot": str(snapshot), "files": files,
            "compatibility_preflight_passed": False, "deployment_ready": False,
            "source": f"https://huggingface.co/{OFFICIAL_MODEL}/tree/{OFFICIAL_REVISION}",
            "timestamp": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(OFFICIAL_MODEL, revision=OFFICIAL_REVISION, files_metadata=True)
    metadata = {"id": info.id, "sha": info.sha, "siblings": [
        {"rfilename": s.rfilename, "size": s.size, "blob_id": s.blob_id,
         "lfs_sha256": s.lfs.sha256 if s.lfs else None} for s in info.siblings]}
    path = snapshot_download(OFFICIAL_MODEL, revision=OFFICIAL_REVISION,
                             cache_dir=str(args.cache_dir), allow_patterns=FILES)
    result = verify_snapshot(Path(path), metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "official-base-frozen", "manifest": str(args.output),
                      "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "deployment_ready": False}), flush=True)


if __name__ == "__main__":
    main()
