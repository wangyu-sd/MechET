#!/usr/bin/env python3
"""Fail unless an EditRetro preprocessed artifact is validated for training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    status_path = args.artifact / "ARTIFACT_STATUS.json"
    if not status_path.is_file():
        raise SystemExit(f"missing artifact gate: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "validated" or not bool(status.get("training_allowed")):
        raise SystemExit(f"artifact is not training-ready: {status_path}")
    audit_path = Path(str(status.get("audit") or ""))
    if not audit_path.is_file():
        raise SystemExit(f"missing preprocessing audit: {audit_path}")
    if sha256_file(audit_path) != str(status.get("audit_sha256") or ""):
        raise SystemExit(f"preprocessing audit hash mismatch: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("training_ready")):
        raise SystemExit(f"audit does not declare training_ready: {audit_path}")
    dictionary = Path(str(audit.get("dictionary") or ""))
    if not dictionary.is_file():
        raise SystemExit(f"missing audited dictionary: {dictionary}")
    if sha256_file(dictionary) != str(audit.get("dictionary_sha256") or ""):
        raise SystemExit(f"audited dictionary hash mismatch: {dictionary}")
    data_bin = args.artifact / "data-bin"
    if not data_bin.is_dir():
        raise SystemExit(f"missing Fairseq data-bin: {data_bin}")
    required = [
        data_bin / f"{split}.src-tgt.{language}.{suffix}"
        for split in ("train", "valid", "test")
        for language in ("src", "tgt")
        for suffix in ("bin", "idx")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing audited Fairseq binaries: {missing[:4]}")
    print(status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
