#!/usr/bin/env python3
"""Extract a complete revisioned Wikibooks subtree from an official XML dump."""
from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.source_health import source_quality_metadata


DEFAULT_DUMP = (
    "https://dumps.wikimedia.org/enwikibooks/latest/"
    "enwikibooks-latest-pages-articles.xml.bz2"
)
DEFAULT_AGENT = "MechET/0.2 (https://github.com/wangyu-sd/MechET; research corpus)"
DEFAULT_EXCLUDES = re.compile(
    r"/(?:Authors|Book_outline|Cover|Doubts_and_Discussions|External_[Ll]inks?|"
    r"Foreword|Glossary|Places_to_buy_organic_chemistry_models|Print_[Vv]ersion|"
    r"Questions_and_Discussions|Templates|To-Do_List)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, output: Path, *, user_agent: str, timeout: float) -> dict[str, Any]:
    if output.exists() and output.stat().st_size:
        return {
            "url": url,
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "reused": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept-Encoding": "identity"},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
        metadata = {
            "url": response.url,
            "last_modified": response.headers.get("Last-Modified"),
            "etag": response.headers.get("ETag"),
        }
    partial.replace(output)
    return {
        **metadata,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "reused": False,
    }


def _child_text(node: ET.Element, name: str) -> str:
    for child in node.iter():
        if child.tag.rsplit("}", 1)[-1] == name:
            return child.text or ""
    return ""


def _revision(page: ET.Element) -> ET.Element | None:
    return next(
        (
            child
            for child in page
            if child.tag.rsplit("}", 1)[-1] == "revision"
        ),
        None,
    )


def extract_dump(
    dump_path: Path,
    *,
    output_root: Path,
    source_id: str,
    source: dict[str, Any],
    prefix: str,
    minimum_characters: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prefix = prefix.replace("_", " ")
    source_root = output_root / source_id / "pages"
    source_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    counts = {
        "pages_seen": 0,
        "prefix_matched": 0,
        "excluded_administrative": 0,
        "redirects": 0,
        "short_or_empty": 0,
        "written": 0,
    }
    with bz2.open(dump_path, "rb") as stream:
        for _, page in ET.iterparse(stream, events=("end",)):
            if page.tag.rsplit("}", 1)[-1] != "page":
                continue
            counts["pages_seen"] += 1
            title = _child_text(page, "title")
            namespace = _child_text(page, "ns")
            if namespace != "0" or not (title == prefix or title.startswith(prefix + "/")):
                page.clear()
                continue
            counts["prefix_matched"] += 1
            if DEFAULT_EXCLUDES.search(title.replace(" ", "_")):
                counts["excluded_administrative"] += 1
                page.clear()
                continue
            if any(child.tag.rsplit("}", 1)[-1] == "redirect" for child in page):
                counts["redirects"] += 1
                page.clear()
                continue
            revision = _revision(page)
            text = _child_text(revision, "text") if revision is not None else ""
            if len(text.strip()) < minimum_characters:
                counts["short_or_empty"] += 1
                page.clear()
                continue
            page_id = next(
                (
                    child.text
                    for child in page
                    if child.tag.rsplit("}", 1)[-1] == "id"
                ),
                "",
            )
            revision_id = _child_text(revision, "id") if revision is not None else ""
            timestamp = _child_text(revision, "timestamp") if revision is not None else ""
            canonical_url = "https://en.wikibooks.org/wiki/" + quote(
                title.replace(" ", "_"), safe="/:,()'-"
            )
            payload = {
                "title": title.replace("_", " "),
                "page_id": page_id,
                "canonical_url": canonical_url,
                "revision_id": revision_id,
                "revision_timestamp": timestamp,
                "license": source.get("license"),
                "wikitext": text,
                "retrieval_backend": "wikimedia_xml_dump",
                "retrieval_url": str(dump_path),
            }
            name = hashlib.sha256(title.encode()).hexdigest()[:16] + ".json"
            path = source_root / name
            encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode()
            path.write_bytes(encoded)
            quality = source_quality_metadata(source, title=title.replace("_", " "))
            artifacts.append(
                {
                    "source_id": source_id,
                    "path": str(path.relative_to(output_root / source_id)),
                    "url": canonical_url,
                    "canonical_url": canonical_url,
                    "license": source.get("license"),
                    "redistribution": source.get("redistribution"),
                    "media_type": "application/json",
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "status": "downloaded",
                    "revision_id": revision_id,
                    "revision_timestamp": timestamp,
                    "retrieval_backend": "wikimedia_xml_dump",
                    "configured_title": title.replace("_", " "),
                    "resolved_title": title.replace("_", " "),
                    "content_characters": len(text),
                    **quality,
                }
            )
            counts["written"] += 1
            page.clear()
    return artifacts, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("knowledge/source_registry.yaml"))
    parser.add_argument("--source-id", default="wikibooks_organic_chemistry")
    parser.add_argument("--prefix", default="Organic Chemistry")
    parser.add_argument("--dump-url", default=DEFAULT_DUMP)
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=Path("knowledge/downloads/enwikibooks-latest-pages-articles.xml.bz2"),
    )
    parser.add_argument("--output", type=Path, default=Path("knowledge/raw_corpus_v2"))
    parser.add_argument("--minimum-characters", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--user-agent", default=DEFAULT_AGENT)
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    source = dict(registry["sources"][args.source_id])
    dump = _download(
        args.dump_url, args.dump_path, user_agent=args.user_agent, timeout=args.timeout
    )
    artifacts, counts = extract_dump(
        args.dump_path,
        output_root=args.output,
        source_id=args.source_id,
        source=source,
        prefix=args.prefix,
        minimum_characters=args.minimum_characters,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    retained = [
        row
        for row in existing.get("artifacts") or []
        if row.get("source_id") != args.source_id
    ]
    manifest = {
        "schema_version": 2,
        "registry": str(args.registry),
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sorted(set([*(existing.get("sources") or []), args.source_id])),
        "artifacts": [*retained, *artifacts],
        "source_snapshots": {
            **dict(existing.get("source_snapshots") or {}),
            args.source_id: {"dump": dump, "extraction": counts},
        },
        "candidate_evidence_only": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "dump": dump, "extraction": counts},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
