#!/usr/bin/env python3
"""Find an untouched official-valid primitive-unseen negative-control stratum.

This does not move any row in the composition-held-out training-pool split.
The official valid rows are held out from all #79 model selection and training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_control(
    train_source: Path, official_valid_source: Path, split_dir: Path,
    source_manifest: Path, output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing negative-control artifact: {output_dir}")
    split_manifest = json.loads((split_dir / "manifest.json").read_text())
    ids_bytes = (split_dir / "train.ids.txt").read_bytes()
    if hashlib.sha256(ids_bytes).hexdigest() != split_manifest["split_id_sha256"]["train"]:
        raise ValueError("H2 train ID hash mismatch")
    train_ids = set(ids_bytes.decode().splitlines())
    source_contract = json.loads(source_manifest.read_text())["splits"]
    train_primitives: set[str] = set()
    found_train_ids: set[str] = set()
    train_sha = hashlib.sha256()
    train_rows = 0
    with train_source.open("rb") as handle:
        for line in handle:
            train_sha.update(line)
            train_rows += 1
            row = json.loads(line)
            identifier = str(row.get("source_id") or "")
            if identifier in train_ids:
                found_train_ids.add(identifier)
                train_primitives.update(row["metadata"]["execution_primitive_signatures"])
    if train_rows != source_contract["train"]["rows"] or train_sha.hexdigest() != source_contract["train"]["sha256"]:
        raise ValueError("training source contract mismatch")
    if found_train_ids != train_ids:
        raise ValueError("some frozen H2 train IDs were not in the source")

    valid_sha = hashlib.sha256()
    valid_rows = 0
    control_ids: list[str] = []
    unseen_primitives: set[str] = set()
    with official_valid_source.open("rb") as handle:
        for line in handle:
            valid_sha.update(line)
            valid_rows += 1
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            if metadata.get("executor_replayed") is not True:
                raise ValueError("unreplayed official-valid source row")
            primitives = set(metadata.get("execution_primitive_signatures") or ())
            if not primitives:
                raise ValueError("official-valid source row lacks primitives")
            novel = primitives - train_primitives
            if novel:
                control_ids.append(str(row["source_id"]))
                unseen_primitives.update(novel)
    if valid_rows != source_contract["valid"]["rows"] or valid_sha.hexdigest() != source_contract["valid"]["sha256"]:
        raise ValueError("official-valid source contract mismatch")
    content = "".join(identifier + "\n" for identifier in control_ids)
    output_dir.mkdir(parents=True)
    (output_dir / "primitive_unseen.ids.txt").write_text(content)
    report = {
        "artifact_type": "nmi_primitive_unseen_official_valid_control_v1",
        "scope": "official_flowER_valid_strict_executable_negative_control_not_h2_headline",
        "h2_split_manifest_sha256": hashlib.sha256((split_dir / "manifest.json").read_bytes()).hexdigest(),
        "h2_train_ids_sha256": split_manifest["split_id_sha256"]["train"],
        "train_source_sha256": train_sha.hexdigest(),
        "official_valid_source_sha256": valid_sha.hexdigest(),
        "official_valid_reactions": valid_rows,
        "negative_control_reactions": len(control_ids),
        "unseen_primitive_types": len(unseen_primitives),
        "control_id_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "test_use_policy": "never select #79 models on this control",
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--official-valid-source", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_control(
        args.train_source, args.official_valid_source, args.split_dir,
        args.source_manifest, args.output_dir,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
