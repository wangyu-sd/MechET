#!/usr/bin/env python3
"""Build evidence-linked primitive-extraction tasks from downloaded Web sources.

This script does not call an LLM and does not create released chemistry rules.
It turns revision/hash-tracked source artifacts into bounded evidence spans with
a strict extraction contract. Downstream models or reviewers must preserve the
source span and mark unsupported fields UNKNOWN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

KEYWORDS = {
    "mechanism", "nucleophile", "nucleophilic", "electrophile", "electrophilic",
    "substitution", "elimination", "addition", "leaving group", "carbonyl",
    "proton transfer", "rearrangement", "migration", "aromatic", "oxidation",
    "reduction", "electron pair", "curved arrow", "intermediate", "selectivity",
}

EXTRACTION_SCHEMA = {
    "candidate_name": "string or UNKNOWN",
    "level": "electron_move | motif | family | UNKNOWN",
    "explicit_claims": ["claims directly supported by the span"],
    "participants": {"semantic_role": "text description or UNKNOWN"},
    "electron_moves": [{
        "source_kind": "LP | BOND | ATOM | UNKNOWN",
        "source_roles": ["semantic roles"],
        "sink_kind": "LP | BOND | ATOM | UNKNOWN",
        "sink_roles": ["semantic roles"],
        "support": "explicit | inferred | absent",
    }],
    "preconditions": ["only conditions stated in the span"],
    "warnings_or_exceptions": ["only stated boundaries"],
    "competing_pathways": ["only stated competitors"],
    "stereochemical_effects": ["only stated effects"],
    "uncertain_fields": ["fields requiring chemistry review"],
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flatten_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        if value.strip(): yield prefix, value.strip()
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from flatten_strings(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten_strings(item, f"{prefix}[{index}]")


def artifact_text(path: Path, media_type: str) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".json" or "json" in media_type:
        value = json.loads(path.read_text(encoding="utf-8"))
        preferred = []
        if isinstance(value, dict):
            for key in ("wikitext", "text", "definition", "term", "name"):
                if isinstance(value.get(key), str) and value[key].strip():
                    preferred.append((key, value[key].strip()))
        return preferred or list(flatten_strings(value))
    if path.suffix.lower() in {".txt", ".md", ".html", ".htm", ".xml", ".owl", ".rdf"}:
        return [("document", path.read_text(encoding="utf-8", errors="replace"))]
    return []


def split_spans(text: str, max_chars: int, min_chars: int) -> list[str]:
    blocks = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n{2,}", re.sub(r"\r\n?", "\n", text))]
    spans, current = [], ""
    for block in filter(None, blocks):
        for sentence in re.split(r"(?<=[.!?])\s+", block) if len(block) > max_chars else [block]:
            candidate = sentence if not current else current + " " + sentence
            if len(candidate) <= max_chars: current = candidate
            else:
                if len(current) >= min_chars: spans.append(current)
                current = sentence[:max_chars]
    if len(current) >= min_chars: spans.append(current)
    return spans


def prompt(span: str) -> str:
    return (
        "Extract only mechanistic knowledge explicitly supported by the evidence span. "
        "Do not invent arrows, conditions, scope, stereochemistry, or exceptions. "
        "Use UNKNOWN for missing fields and return JSON matching extraction_schema.\n\n"
        "EVIDENCE SPAN:\n" + span
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=2400)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.download_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keywords, written, skipped = set(args.keyword) or set(KEYWORDS), 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for artifact in manifest.get("artifacts") or []:
            if artifact.get("status") not in (None, "downloaded"):
                skipped += 1; continue
            source_id, relative = str(artifact.get("source_id") or ""), str(artifact.get("path") or "")
            path = args.download_root / source_id / relative
            if not path.exists(): skipped += 1; continue
            for field_path, text in artifact_text(path, str(artifact.get("media_type") or "")):
                for span_index, span in enumerate(split_spans(text, args.max_chars, args.min_chars)):
                    hits = sorted(x for x in keywords if x in span.lower())
                    if not args.include_all and not hits: continue
                    identity = f"{source_id}|{relative}|{field_path}|{span_index}|{sha256_text(span)}"
                    row = {
                        "candidate_id": "primitive-extract:" + hashlib.sha1(identity.encode()).hexdigest()[:20],
                        "source_id": source_id, "source_path": relative,
                        "source_url": artifact.get("url"), "license": artifact.get("license"),
                        "revision": artifact.get("revision_id") or artifact.get("term_id") or "",
                        "artifact_sha256": artifact.get("sha256"), "field_path": field_path,
                        "span_index": span_index, "span_sha256": sha256_text(span),
                        "keyword_hits": hits, "evidence_span": span,
                        "extraction_schema": EXTRACTION_SCHEMA, "prompt": prompt(span),
                        "status": "unreviewed_candidate", "released_primitive": False,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n"); written += 1
                    if args.limit and written >= args.limit: break
                if args.limit and written >= args.limit: break
            if args.limit and written >= args.limit: break
    summary = {
        "download_manifest": str(manifest_path),
        "download_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "output": str(args.output), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "written": written, "skipped_artifacts": skipped, "candidate_evidence_only": True,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
