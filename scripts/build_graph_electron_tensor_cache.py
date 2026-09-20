#!/usr/bin/env python3
"""Compile auditable graph-policy JSONL into reusable tensor-cache chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from torch.utils.data import DataLoader

from train_graph_electron_full_batched import (
    JsonlDecisionDataset,
    collate_prepared,
    write_tensor_chunk,
)


FORMAT = "graph_electron_tensor_cache_v2_aligned_product"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_shard(
    source: Path,
    output: Path,
    *,
    expected_rows: int,
    expected_sha256: str,
    workers: int,
    chunk_size: int,
) -> dict:
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source hash mismatch: {source}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"tensor-cache output is not empty: {output}")
    dataset = JsonlDecisionDataset(source)
    if len(dataset) != expected_rows:
        raise ValueError(f"source row mismatch: {len(dataset)} != {expected_rows}")
    loader = DataLoader(
        dataset,
        batch_size=chunk_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_prepared,
        multiprocessing_context="spawn",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
    )
    started = time.time()
    rows = 0
    chunks = []
    for index, batch in enumerate(loader):
        path = output / f"chunk{index:05d}.pt"
        write_tensor_chunk(batch, path)
        rows += len(batch)
        chunks.append({"file": path.name, "rows": len(batch), "bytes": path.stat().st_size})
        print(
            f"[tensor-cache] source={source.name} rows={rows}/{expected_rows} "
            f"chunks={len(chunks)} rate={rows / max(time.time() - started, 1e-6):.2f}decision/s",
            flush=True,
        )
    if rows != expected_rows:
        raise ValueError(f"compiled row mismatch: {rows} != {expected_rows}")
    manifest = {
        "format": FORMAT,
        "source": str(source),
        "source_sha256": actual_sha256,
        "rows": rows,
        "chunk_size": chunk_size,
        "chunks": chunks,
        "wall_seconds": time.time() - started,
    }
    temporary = output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "valid", "test"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise SystemExit("workers and chunk-size must be positive")
    source_manifest_path = args.data / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    root = {
        "format": FORMAT,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "splits": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        split_meta = source_manifest["splits"][split]
        split_rows = 0
        rank_outputs = []
        for rank, shard in enumerate(split_meta["shards"]):
            destination = args.output / f"{split}.rank{rank:02d}"
            result = build_shard(
                args.data / shard["file"],
                destination,
                expected_rows=int(shard["rows"]),
                expected_sha256=str(shard["sha256"]),
                workers=args.workers,
                chunk_size=args.chunk_size,
            )
            split_rows += int(result["rows"])
            rank_outputs.append(str(destination))
        if split_rows != int(split_meta["decisions"]):
            raise ValueError(f"{split}: cache denominator changed")
        root["splits"][split] = {"decisions": split_rows, "rank_outputs": rank_outputs}
    temporary = args.output / "manifest.json.partial"
    temporary.write_text(json.dumps(root, indent=2) + "\n")
    os.replace(temporary, args.output / "manifest.json")
    print(f"[tensor-cache] completed manifest={args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
