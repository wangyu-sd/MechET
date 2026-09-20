#!/usr/bin/env python3
"""Export the shared canonical decisions into state/trajectory LLM BC rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mechet.electron_policy_protocol import (
    PROTOCOL_VERSION,
    STAGE_STATE_BC,
    STAGE_TRAJECTORY_BC,
    llm_training_record,
)


SYSTEM = (
    "You are a retrosynthetic electron-policy model. Predict exactly one next "
    "inverse electron event from the executor-owned current state. Use temporary "
    "atom names from the current observation, never invent atom-map numbers, "
    "never enumerate alternatives, and emit only the canonical JSON action."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def export_shard(source: Path, output: Path, *, stage: str) -> dict:
    partial = output.with_suffix(output.suffix + ".partial")
    rows = 0
    with source.open() as reader, partial.open("w") as writer:
        for line in reader:
            decision = json.loads(line)
            record = llm_training_record(decision, stage=stage)
            writer.write(
                json.dumps(
                    {
                        "id": f"{record['id']}:{decision['decision_index']}",
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": record["input"]},
                            {"role": "assistant", "content": record["output"]},
                        ],
                        "metadata": {
                            "reaction_id": record["id"],
                            "decision_index": int(decision["decision_index"]),
                            "track": "llm",
                            "stage": stage,
                            "protocol": PROTOCOL_VERSION,
                            "action_family": str(decision["kind"]),
                        },
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            rows += 1
            if rows % 10000 == 0:
                print(f"[llm-bc-export] shard={source.name} rows={rows}", flush=True)
    os.replace(partial, output)
    return {"file": output.name, "rows": rows, "sha256": sha256_file(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=(STAGE_STATE_BC, STAGE_TRAJECTORY_BC), required=True
    )
    parser.add_argument("--splits", nargs="+", default=("train", "valid", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = args.data / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if int(source_manifest.get("schema_version") or 0) < 3:
        raise SystemExit("LLM trajectory protocol requires schema-v3 canonical decisions")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit(f"output must be empty: {args.output}")
    manifest = {
        "artifact_type": "electron_policy_llm_next_event_bc",
        "protocol": PROTOCOL_VERSION,
        "track": "llm",
        "stage": args.stage,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "reaction_denominator": source_manifest["reaction_denominator"],
        "candidate_enumeration": False,
        "splits": {},
    }
    for split in args.splits:
        shards = []
        total = 0
        for index, source_shard in enumerate(source_manifest["splits"][split]["shards"]):
            output = args.output / f"{split}.rank{index:02d}.jsonl"
            result = export_shard(
                args.data / source_shard["file"], output, stage=args.stage
            )
            if result["rows"] != int(source_shard["rows"]):
                raise ValueError(f"{split} rank {index}: decision denominator changed")
            shards.append(result)
            total += int(result["rows"])
        expected = int(source_manifest["splits"][split]["decisions"])
        if total != expected:
            raise ValueError(f"{split}: expected {expected} decisions, got {total}")
        manifest["splits"][split] = {"decisions": total, "shards": shards}
    temporary = args.output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, args.output / "manifest.json")
    print(f"[llm-bc-export] completed manifest={args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
