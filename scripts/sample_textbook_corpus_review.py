#!/usr/bin/env python3
"""Create a deterministic, stratified human-review sheet for a textbook corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_store import TextbookStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("knowledge/corpus_v2_spec.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    store = TextbookStore.load(args.corpus)
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    selected: dict[str, object] = {}
    strata: dict[str, list[str]] = {}

    def select(label: str, candidates: list[object]) -> None:
        candidates = [row for row in candidates if row.passage_id not in selected]
        if not candidates:
            return
        row = rng.choice(sorted(candidates, key=lambda item: item.passage_id))
        selected[row.passage_id] = row
        strata.setdefault(row.passage_id, []).append(label)

    for source_id in sorted({row.source_id for row in store.passages}):
        select(f"source:{source_id}", [row for row in store.passages if row.source_id == source_id])
    for topic in (spec.get("required_topic_coverage") or {}):
        select(f"topic:{topic}", [row for row in store.passages if topic in row.topics])
    for phase in (spec.get("required_phase_coverage") or {}):
        select(f"phase:{phase}", [row for row in store.passages if phase in row.phases])
    remaining = [row for row in store.passages if row.passage_id not in selected]
    rng.shuffle(remaining)
    for row in remaining[: max(args.sample_size - len(selected), 0)]:
        selected[row.passage_id] = row
        strata.setdefault(row.passage_id, []).append("random")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for passage_id in sorted(selected):
        passage = selected[passage_id]
        rows.append(
            {
                "passage_id": passage.passage_id,
                "review_strata": strata[passage_id],
                "source_id": passage.source_id,
                "title": passage.title,
                "locator": passage.locator,
                "topics": list(passage.topics),
                "phases": list(passage.phases),
                "modalities": list(passage.modalities),
                "text": passage.text,
                "review": {
                    "chemically_correct": None,
                    "self_contained": None,
                    "clean_extraction": None,
                    "tag_correct": None,
                    "keep": None,
                    "notes": "",
                },
            }
        )
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    markdown = [
        "# Corpus v2 stratified review sheet",
        "",
        f"Sample size: {len(rows)}; seed: {args.seed}",
        "",
    ]
    for row in rows:
        markdown.extend(
            [
                f"## {row['passage_id']} — {row['title']}",
                "",
                f"Source: `{row['source_id']}`; strata: {', '.join(row['review_strata'])}",
                "",
                str(row["text"]),
                "",
                "Review: correct [ ] self-contained [ ] clean [ ] tags [ ] keep [ ]",
                "",
            ]
        )
    args.output.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_sampled": len(rows),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "status": "awaiting_human_review",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
