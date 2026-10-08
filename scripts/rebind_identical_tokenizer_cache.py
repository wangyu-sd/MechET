#!/usr/bin/env python3
"""Reuse frozen Arrow tokens only when two model tokenizers are byte-identical.

The generated cache has its own model lineage manifest and symlinks the immutable
Arrow shards.  It never modifies the source cache or silently accepts stale data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matching_tokenizer_files(source: Path, target: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in TOKENIZER_FILES:
        left, right = source / name, target / name
        if left.exists() != right.exists():
            raise ValueError(f"tokenizer file presence differs: {name}")
        if left.exists():
            left_hash, right_hash = sha256(left), sha256(right)
            if left_hash != right_hash:
                raise ValueError(f"tokenizer file differs: {name}")
            hashes[name] = left_hash
    if "tokenizer.json" not in hashes or "tokenizer_config.json" not in hashes:
        raise ValueError("both frozen tokenizers need tokenizer.json and tokenizer_config.json")
    return hashes


def rebind_cache(
    *,
    source_cache: Path,
    target_cache: Path,
    source_tokenizer: Path,
    target_tokenizer: Path,
    source_model: str,
    source_revision: str,
    target_model: str,
    target_revision: str,
    expected_sources: dict[str, tuple[Path, str]],
    max_length: int,
) -> dict:
    if target_cache.exists():
        raise FileExistsError(f"target cache already exists: {target_cache}")
    manifest_path = source_cache / "manifest.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("model_name_or_path") != source_model:
        raise ValueError("source cache model name mismatch")
    if source.get("model_revision") != source_revision:
        raise ValueError("source cache model revision mismatch")
    if int(source.get("max_length") or 0) != max_length:
        raise ValueError("source cache max_length mismatch")
    hashes = matching_tokenizer_files(source_tokenizer, target_tokenizer)

    for split, (path, expected_hash) in expected_sources.items():
        recorded = source.get("sources", {}).get(split, {})
        if recorded.get("sha256") != expected_hash:
            raise ValueError(f"{split} frozen source hash mismatch")
        if not path.is_file() or path.stat().st_size != recorded.get("bytes"):
            raise ValueError(f"{split} source file missing or size mismatch")
        if sha256(path) != expected_hash:
            raise ValueError(f"{split} source bytes differ from frozen hash")

    shards: dict[str, Path] = {}
    for split in ("train", "validation"):
        report = source.get("splits", {}).get(split, {})
        if int(report.get("truncation_count", -1)) != 0:
            raise ValueError(f"{split} cache has truncated rows")
        for recorded in report.get("arrow_files", []):
            name = Path(recorded).name
            if name in shards:
                raise ValueError(f"duplicate Arrow shard name: {name}")
            shard = source_cache / name
            if not shard.is_file():
                raise FileNotFoundError(shard)
            shards[name] = shard
        if not report.get("arrow_files"):
            raise ValueError(f"{split} has no Arrow shards")

    rebound = dict(source)
    rebound["model_name_or_path"] = target_model
    rebound["model_revision"] = target_revision
    rebound["derived_from_identical_tokenizer_cache"] = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "source_model": source_model,
        "source_revision": source_revision,
        "tokenizer_file_sha256": hashes,
        "reuse_policy": "byte_identical_tokenizer_assets_and_frozen_source_bytes_v1",
    }

    target_cache.mkdir(parents=True)
    for name, shard in shards.items():
        (target_cache / name).symlink_to(os.path.relpath(shard, target_cache))
    (target_cache / "manifest.json").write_text(
        json.dumps(rebound, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return rebound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--source-tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()
    result = rebind_cache(
        source_cache=args.source_cache,
        target_cache=args.target_cache,
        source_tokenizer=args.source_tokenizer,
        target_tokenizer=args.target_tokenizer,
        source_model=args.source_model,
        source_revision=args.source_revision,
        target_model=args.target_model,
        target_revision=args.target_revision,
        expected_sources={
            "train": (args.train_file, args.train_sha256),
            "validation": (args.validation_file, args.validation_sha256),
        },
        max_length=args.max_length,
    )
    print(
        json.dumps(
            {
                "target_cache": str(args.target_cache),
                "model": result["model_name_or_path"],
                "revision": result["model_revision"],
                "train_rows": result["splits"]["train"]["n_rows"],
                "validation_rows": result["splits"]["validation"]["n_rows"],
                "tokenizer_sha256": result["derived_from_identical_tokenizer_cache"][
                    "tokenizer_file_sha256"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
