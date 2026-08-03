#!/usr/bin/env python3
"""Validate a textbook corpus and write a deterministic BM25 index manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_retriever import TextbookRetriever
from mechet.textbook_store import TextbookStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("knowledge/corpus/passages.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("knowledge/corpus/bm25_index.json"))
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--state-weight", type=float, default=0.35)
    args = parser.parse_args()

    store = TextbookStore.load(args.corpus)
    retriever = TextbookRetriever(
        store,
        k1=args.k1,
        b=args.b,
        state_weight=args.state_weight,
    )
    value = {
        **retriever.manifest(),
        "document_frequency": retriever.document_frequency,
        "document_lengths": retriever.lengths,
        "passage_ids": [item.passage_id for item in store.passages],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(retriever.manifest(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
