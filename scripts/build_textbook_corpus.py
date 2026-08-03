#!/usr/bin/env python3
"""Build a provenance-preserving natural-language textbook passage corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_store import TextbookPassage, TextbookStore


_TEXT_KEYS = {
    "text",
    "wikitext",
    "definition",
    "definitions",
    "description",
    "content",
    "body",
    "term",
    "name",
    "title",
}
_TOPIC_KEYWORDS = {
    "substitution": ("substitution", "nucleophilic displacement", "leaving group"),
    "elimination": ("elimination", "beta hydrogen", "e2", "e1"),
    "carbonyl": ("carbonyl", "aldehyde", "ketone", "acyl"),
    "addition": ("addition", "nucleophilic attack", "electrophilic attack"),
    "aromatic": ("aromatic", "rearomatization", "arenium"),
    "proton_transfer": ("proton transfer", "protonation", "deprotonation"),
    "oxidation_reduction": ("oxidation", "reduction", "hydride"),
    "rearrangement": ("rearrangement", "migration", "migratory aptitude"),
    "stereochemistry": ("stereochemistry", "inversion", "retention", "enantio"),
}


def _collect_strings(value: Any, *, key: str = "") -> Iterable[str]:
    if isinstance(value, str):
        if key.lower() in _TEXT_KEYS or len(value) >= 160:
            yield value
        return
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _collect_strings(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _collect_strings(child, key=key)


def _artifact_text(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        parts = [item.strip() for item in _collect_strings(payload) if item.strip()]
        title = str(payload.get("title") or payload.get("term", {}).get("name") or path.stem) if isinstance(payload, dict) else path.stem
        return title, "\n\n".join(dict.fromkeys(parts))
    return path.stem, raw


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunks(text: str, *, minimum: int, maximum: int, overlap: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    output: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > maximum:
            if current:
                output.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                chunk = paragraph[start : start + maximum].strip()
                if len(chunk) >= minimum:
                    output.append(chunk)
                start += max(maximum - overlap, 1)
            continue
        candidate = paragraph if not current else current + "\n\n" + paragraph
        if len(candidate) <= maximum:
            current = candidate
        else:
            if len(current) >= minimum:
                output.append(current)
            current = paragraph
    if len(current) >= minimum:
        output.append(current)
    return output


def _topics(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        topic
        for topic, phrases in _TOPIC_KEYWORDS.items()
        if any(phrase in lowered for phrase in phrases)
    )


def build(
    download_root: Path,
    *,
    minimum: int,
    maximum: int,
    overlap: int,
) -> TextbookStore:
    manifest_path = download_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    passages: list[TextbookPassage] = []
    for artifact in payload.get("artifacts") or []:
        if artifact.get("status") not in (None, "downloaded", "offline_import"):
            continue
        source_id = str(artifact.get("source_id") or "")
        relative = str(artifact.get("path") or "")
        path = download_root / source_id / relative
        if not source_id or not relative or not path.exists():
            continue
        if path.suffix.lower() not in {".json", ".txt", ".xml", ".md"}:
            continue
        try:
            title, text = _artifact_text(path)
        except Exception:
            continue
        text = _normalize(text)
        for index, chunk in enumerate(
            _chunks(text, minimum=minimum, maximum=maximum, overlap=overlap)
        ):
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            passage_id = f"{source_id}:{digest[:16]}:{index}"
            passages.append(
                TextbookPassage(
                    passage_id=passage_id,
                    title=title,
                    text=chunk,
                    source_id=source_id,
                    locator=str(artifact.get("canonical_url") or artifact.get("url") or relative),
                    revision=str(
                        artifact.get("revision_id")
                        or artifact.get("revision")
                        or artifact.get("canonical_term_id")
                        or ""
                    ),
                    license=str(artifact.get("license") or "unknown"),
                    source_url=str(artifact.get("url") or ""),
                    evidence_sha256=digest,
                    topics=_topics(chunk),
                    metadata={
                        "artifact_path": relative,
                        "artifact_sha256": artifact.get("sha256"),
                        "retrieval_backend": artifact.get("retrieval_backend"),
                    },
                )
            )
    return TextbookStore(passages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, default=Path("knowledge/raw"))
    parser.add_argument("--output", type=Path, default=Path("knowledge/corpus/passages.jsonl"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=160)
    args = parser.parse_args()

    store = build(
        args.download_root,
        minimum=args.min_chars,
        maximum=args.max_chars,
        overlap=args.overlap_chars,
    )
    store.save(args.output)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                **store.manifest(),
                "download_root": str(args.download_root),
                "source_manifest_sha256": hashlib.sha256(
                    (args.download_root / "manifest.json").read_bytes()
                ).hexdigest(),
                "chunking": {
                    "min_chars": args.min_chars,
                    "max_chars": args.max_chars,
                    "overlap_chars": args.overlap_chars,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(store.manifest(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
