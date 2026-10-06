#!/usr/bin/env python3
"""Separate electron-participant and endpoint-context fragment supervision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _role(fragments: list[dict[str, Any]]) -> str:
    purposes = {str(item["purpose"]) for item in fragments}
    if not purposes <= {"electron_participant", "endpoint_context"}:
        raise ValueError(f"unknown import purpose: {purposes}")
    if not purposes:
        raise ValueError("empty import decision")
    return "mixed" if len(purposes) > 1 else next(iter(purposes))


def audit(
    *, decisions: Path, manifest: Path, split: str,
    predictions: Path | None = None,
) -> dict[str, Any]:
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    info = json.loads(manifest.read_text(encoding="utf-8"))
    expected = info["splits"][split]
    digest = _sha256(decisions)
    if digest != expected["output_sha256"]:
        raise ValueError("decision source SHA-256 mismatch")
    selected: dict[str, dict[str, Any]] = {}
    if predictions is not None:
        for row in _rows(predictions):
            if row["decision_type"] != "import":
                continue
            key = str(row["key"])
            if key in selected:
                raise ValueError(f"duplicate predicted decision: {key}")
            selected[key] = row
    counts: Counter[str] = Counter()
    reactions: dict[str, set[str]] = defaultdict(set)
    model: Counter[str] = Counter()
    for row in _rows(decisions):
        counts["decision_rows"] += 1
        if counts["decision_rows"] % 100000 == 0:
            print({"split": split, "decision_rows_checked": counts["decision_rows"]}, flush=True)
        function = row["messages"][2]["tool_calls"][0]["function"]
        if function["name"] != "import_fragments":
            continue
        fragments = list(function["arguments"]["fragments"])
        role = _role(fragments)
        counts["import_decisions"] += 1
        counts[f"import_decisions_{role}"] += 1
        source_id = str(row["source_id"])
        reactions[source_id].add(role)
        for fragment in fragments:
            purpose = str(fragment["purpose"])
            copies = int(fragment["count"])
            if copies < 1:
                raise ValueError(f"nonpositive fragment count: {row['id']}")
            counts[f"fragment_copies_{purpose}"] += copies
        key = str(row["id"])
        if key in selected:
            prediction = selected.pop(key)
            if prediction["decision_type"] != "import":
                raise ValueError(f"predicted decision type disagrees: {key}")
            model[f"{role}_n"] += 1
            model[f"{role}_correct_tool"] += bool(prediction["correct_tool"])
            model[f"{role}_fragment_exact"] += bool(prediction["import_fragment_exact"])
    if selected:
        raise ValueError(f"predictions absent from import supervision: {list(selected)[:3]}")
    if counts["decision_rows"] != int(info["decision_rows"][split]):
        raise ValueError("decision-row denominator mismatch")
    counts["reactions_with_import"] = len(reactions)
    counts["reactions_with_endpoint_context"] = sum(
        "endpoint_context" in kinds or "mixed" in kinds
        for kinds in reactions.values()
    )
    counts["reactions_with_electron_participant"] = sum(
        "electron_participant" in kinds or "mixed" in kinds
        for kinds in reactions.values()
    )
    return {
        "artifact_type": "reliable_mechet_import_supervision_audit_v1",
        "split": split, "decisions": str(decisions), "decisions_sha256": digest,
        "reaction_denominator": int(info["reaction_denominator"][split]),
        "counts": dict(counts),
        "prediction_file": str(predictions) if predictions else None,
        "prediction_sha256": _sha256(predictions) if predictions else None,
        "prediction_by_gold_role": dict(model) if predictions else None,
        "interpretation": (
            "Electron-participant imports may be required for a reverse-electron-flow "
            "transition. Endpoint-context imports can include solvents, salts and other "
            "spectators; their exact identity is not generally identifiable from a "
            "product-only input. These supervision roles must be reported separately."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        decisions=args.decisions, manifest=args.manifest,
        split=args.split, predictions=args.predictions,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
