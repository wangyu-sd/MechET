#!/usr/bin/env python3
"""Audit organic textbook corpus quality, diversity and declared coverage gates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_cleaning import quality_flags
from mechet.textbook_store import TextbookStore


def _normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    return sorted(values)[round((len(values) - 1) * q)]


def audit(store: TextbookStore, spec: dict[str, Any]) -> dict[str, Any]:
    source = Counter()
    license_counts = Counter()
    topics = Counter()
    phases = Counter()
    modalities = Counter()
    groups = Counter()
    flags = Counter()
    section_kinds = Counter()
    normalized = Counter()
    lengths: list[int] = []
    source_topics: dict[str, Counter[str]] = defaultdict(Counter)
    for passage in store.passages:
        source[passage.source_id] += 1
        license_counts[passage.license] += 1
        topics.update(passage.topics)
        phases.update(passage.phases)
        modalities.update(passage.modalities)
        groups.update(passage.functional_groups)
        observed_flags = quality_flags(passage.text)
        flags.update(observed_flags)
        section_kinds[str((passage.metadata or {}).get("section_kind") or "UNKNOWN")] += 1
        normalized[_normalized(passage.text)] += 1
        lengths.append(len(passage.text))
        for topic in passage.topics:
            source_topics[passage.source_id][topic] += 1
    exact_duplicate_rows = sum(count - 1 for count in normalized.values() if count > 1)
    n = len(store.passages)
    required_topics = dict(spec.get("required_topic_coverage") or {})
    required_phases = dict(spec.get("required_phase_coverage") or {})
    coverage = {
        "topics": {
            key: {
                "observed": topics.get(key, 0),
                "required": int(required),
                "pass": topics.get(key, 0) >= int(required),
            }
            for key, required in required_topics.items()
        },
        "phases": {
            key: {
                "observed": phases.get(key, 0),
                "required": int(required),
                "pass": phases.get(key, 0) >= int(required),
            }
            for key, required in required_phases.items()
        },
    }
    gates = dict(spec.get("quality_gates") or {})
    gate_results = {
        "minimum_passages": {
            "observed": n,
            "required": int(gates.get("minimum_passages", 0)),
            "pass": n >= int(gates.get("minimum_passages", 0)),
        },
        "minimum_source_count": {
            "observed": len(source),
            "required": int(gates.get("minimum_source_count", 1)),
            "pass": len(source) >= int(gates.get("minimum_source_count", 1)),
        },
        "maximum_passages_with_markup": {
            "observed": flags.get("wiki_markup", 0),
            "maximum": int(gates.get("maximum_passages_with_markup", 0)),
            "pass": flags.get("wiki_markup", 0)
            <= int(gates.get("maximum_passages_with_markup", 0)),
        },
        "maximum_passages_with_page_furniture": {
            "observed": flags.get("page_furniture", 0),
            "maximum": int(gates.get("maximum_passages_with_page_furniture", 0)),
            "pass": flags.get("page_furniture", 0)
            <= int(gates.get("maximum_passages_with_page_furniture", 0)),
        },
        "maximum_passages_with_urls": {
            "observed": flags.get("url_heavy", 0),
            "maximum": int(gates.get("maximum_passages_with_urls", 0)),
            "pass": flags.get("url_heavy", 0)
            <= int(gates.get("maximum_passages_with_urls", 0)),
        },
        "maximum_exact_duplicate_fraction": {
            "observed": exact_duplicate_rows / max(n, 1),
            "maximum": float(gates.get("maximum_exact_duplicate_fraction", 0.0)),
            "pass": exact_duplicate_rows / max(n, 1)
            <= float(gates.get("maximum_exact_duplicate_fraction", 0.0)),
        },
    }
    all_coverage = [
        item["pass"] for family in coverage.values() for item in family.values()
    ]
    automatic_gates_pass = all(all_coverage) and all(
        item["pass"] for item in gate_results.values()
    )
    corpus_status = str(spec.get("status") or "development")
    return {
        "corpus": store.manifest(),
        "n_characters": sum(lengths),
        "passage_length_characters": {
            "min": min(lengths, default=0),
            "p25": _quantile(lengths, 0.25),
            "median": _quantile(lengths, 0.5),
            "p75": _quantile(lengths, 0.75),
            "p95": _quantile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "distributions": {
            "sources": dict(source),
            "licenses": dict(license_counts),
            "topics": dict(topics),
            "phases": dict(phases),
            "modalities": dict(modalities),
            "functional_groups": dict(groups),
            "section_kinds": dict(section_kinds),
            "quality_flags": dict(flags),
        },
        "source_topic_matrix": {
            key: dict(value) for key, value in source_topics.items()
        },
        "exact_duplicate_rows": exact_duplicate_rows,
        "coverage": coverage,
        "quality_gates": gate_results,
        "automatic_gates_pass": automatic_gates_pass,
        "human_review_required": True,
        # A development build is never promoted by automatic counts alone.
        "headline_ready": automatic_gates_pass and corpus_status == "frozen",
        "audit_digest": hashlib.sha256(
            json.dumps(
                {
                    "corpus": store.manifest(),
                    "spec": spec,
                    "coverage": coverage,
                    "gates": gate_results,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Organic textbook corpus audit",
        "",
        f"- Passages: {report['corpus']['n_passages']:,}",
        f"- Characters: {report['n_characters']:,}",
        f"- Sources: {len(report['distributions']['sources'])}",
        f"- Automatic gates pass: **{report['automatic_gates_pass']}**",
        f"- Human review required: **{report['human_review_required']}**",
        f"- Headline ready (requires status=frozen): **{report['headline_ready']}**",
        "",
        "## Sources",
        "",
        "| Source | Passages |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {key} | {value:,} |"
        for key, value in sorted(report["distributions"]["sources"].items())
    )
    for family, title in (("topics", "Topic coverage"), ("phases", "Phase coverage")):
        lines.extend(["", f"## {title}", "", "| Label | Observed | Required | Pass |", "|---|---:|---:|:---:|"])
        lines.extend(
            f"| {key} | {item['observed']:,} | {item['required']:,} | {item['pass']} |"
            for key, item in report["coverage"][family].items()
        )
    lines.extend(["", "## Quality gates", "", "| Gate | Observed | Limit | Pass |", "|---|---:|---:|:---:|"])
    for key, item in report["quality_gates"].items():
        limit = item.get("required", item.get("maximum"))
        lines.append(f"| {key} | {item['observed']} | {limit} | {item['pass']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("knowledge/corpus_v2_spec.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gates", action="store_true")
    args = parser.parse_args()
    store = TextbookStore.load(args.corpus)
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    report = audit(store, spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "automatic_gates_pass": report["automatic_gates_pass"],
                "headline_ready": report["headline_ready"],
            },
            indent=2,
        )
    )
    return int(args.fail_on_gates and not report["automatic_gates_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
