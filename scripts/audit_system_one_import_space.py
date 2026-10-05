#!/usr/bin/env python3
"""Audit open-vocabulary IMPORT support without defining an enumerated policy."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_system_one_electron_flow import verify_source


def normalize_batch(arguments: dict) -> tuple[tuple[str, int, str], ...]:
    fragments = arguments.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("IMPORT must contain at least one fragment")
    normalized = []
    for fragment in fragments:
        smiles = fragment.get("smiles")
        count = fragment.get("count")
        purpose = fragment.get("purpose")
        if not isinstance(smiles, str) or not smiles or type(count) is not int or count < 1:
            raise ValueError("IMPORT fragment has invalid SMILES or count")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError("IMPORT fragment has no purpose")
        normalized.append((smiles, count, purpose))
    return tuple(sorted(normalized))


def audit_split(path: Path, expected_rows: int, expected_imports: int) -> dict:
    batches: Counter[tuple[tuple[str, int, str], ...]] = Counter()
    fragments: Counter[str] = Counter()
    purposes: Counter[str] = Counter()
    rows = imports = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank source row in {path}")
            rows += 1
            row = json.loads(line)
            if row["metadata"]["decision_type"] != "import":
                continue
            imports += 1
            calls = [call for message in row["messages"]
                     if message.get("role") == "assistant"
                     for call in message.get("tool_calls", ())]
            if len(calls) != 1 or calls[0]["function"]["name"] != "import_fragments":
                raise ValueError(f"{row['id']}: IMPORT label disagrees with tool call")
            batch = normalize_batch(calls[0]["function"]["arguments"])
            batches[batch] += 1
            for smiles, count, purpose in batch:
                fragments[smiles] += count
                purposes[purpose] += count
            if imports % 50000 == 0:
                print(json.dumps({"phase": "audit", "split": path.stem,
                                  "source_rows": rows, "imports": imports}), flush=True)
    if rows != expected_rows or imports != expected_imports:
        raise ValueError(f"{path.stem}: row/import denominator mismatch")
    return {
        "source_rows": rows,
        "import_decisions": imports,
        "distinct_batches": len(batches),
        "distinct_fragment_smiles": len(fragments),
        "singleton_batches": sum(count == 1 for count in batches.values()),
        "purpose_counts": dict(purposes),
        "top_batches": [
            {"fragments": [{"smiles": smiles, "count": count, "purpose": purpose}
                           for smiles, count, purpose in batch], "decisions": frequency}
            for batch, frequency in batches.most_common(12)
        ],
        "top_fragments": fragments.most_common(12),
        "_batch_set": set(batches),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"),
                        default=("train", "valid", "test"))
    args = parser.parse_args()
    status = json.loads((args.data_dir / "ARTIFACT_STATUS.json").read_text())
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if (status.get("artifact_id") != manifest.get("artifact_type")
            or not status.get("training_allowed") or not manifest.get("training_allowed")):
        raise ValueError("source is not a validated train-ready trace view")
    reports = {}
    sources = {}
    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        sources[split] = verify_source(path)
        declared = manifest["splits"][split]
        reports[split] = audit_split(
            path, declared["decision_rows"], declared["import_decisions"]
        )
        print(json.dumps({"phase": "split_done", "split": split,
                          "imports": reports[split]["import_decisions"],
                          "distinct_batches": reports[split]["distinct_batches"]}), flush=True)
    if "train" in reports:
        for split in ("valid", "test"):
            if split in reports:
                reports[split]["unseen_batches_vs_train"] = len(
                    reports[split]["_batch_set"] - reports["train"]["_batch_set"]
                )
    for value in reports.values():
        del value["_batch_set"]
    output = {
        "artifact_type": "system_one_import_space_audit",
        "scope": "observed_fragment_diversity_not_an_enumerated_import_policy",
        "source_artifact": manifest["artifact_type"],
        "sources": sources,
        "splits": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
