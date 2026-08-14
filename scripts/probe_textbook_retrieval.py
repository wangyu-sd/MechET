#!/usr/bin/env python3
"""Run transparent topic-recall smoke probes against the frozen retriever."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_retriever import TextbookRetriever
from mechet.textbook_store import TextbookStore

PROBES = (
    ("SN2 backside attack inversion polar aprotic solvent", {"substitution"}),
    ("E2 anti-periplanar beta hydrogen elimination", {"elimination"}),
    ("nucleophilic addition to aldehyde carbonyl", {"carbonyl", "addition"}),
    ("Diels-Alder cycloaddition orbital symmetry", {"pericyclic"}),
    ("organic acid base equilibrium pKa", {"acid_base"}),
    ("radical bromination chain propagation", {"radical"}),
    ("photochemical organic reaction excited state", {"photochemistry"}),
    ("electrochemical oxidation organic molecule electrode", {"electrochemistry"}),
    ("electron ionization molecular ion alpha cleavage McLafferty rearrangement", {"mass_spectrometry", "fragmentation"}),
    ("ESI tandem mass spectrometry collision induced dissociation", {"mass_spectrometry", "ionization"}),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    retriever = TextbookRetriever(TextbookStore.load(args.corpus))
    probes = []
    for query, expected in PROBES:
        results = retriever.retrieve(query, top_k=args.top_k, max_per_source=args.top_k)
        first_hit = None
        serialized = []
        for rank, result in enumerate(results, 1):
            topics = set(result.passage.topics)
            if first_hit is None and topics & expected:
                first_hit = rank
            serialized.append(
                {
                    "rank": rank,
                    "passage_id": result.passage.passage_id,
                    "title": result.passage.title,
                    "source_id": result.passage.source_id,
                    "topics": list(result.passage.topics),
                    "score": result.score,
                    "preview": result.passage.text[:240],
                }
            )
        probes.append(
            {
                "query": query,
                "expected_topics": sorted(expected),
                "first_topic_hit": first_hit,
                "pass_at_k": first_hit is not None,
                "results": serialized,
            }
        )
    report = {
        "status": "automatic_smoke_probe_not_human_relevance_evaluation",
        "top_k": args.top_k,
        "n_probes": len(probes),
        "passed": sum(row["pass_at_k"] for row in probes),
        "empty_query_returns": len(retriever.retrieve("", top_k=args.top_k)),
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "n_probes", "passed", "empty_query_returns"]}, indent=2))
    return int(report["passed"] != report["n_probes"] or report["empty_query_returns"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
