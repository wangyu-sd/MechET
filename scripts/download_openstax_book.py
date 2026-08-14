#!/usr/bin/env python3
"""Download and structurally extract a complete OpenStax book snapshot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag
import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.source_health import source_quality_metadata
from mechet.textbook_cleaning import clean_page_furniture, normalize_prose


DEFAULT_AGENT = "MechET/0.2 (https://github.com/wangyu-sd/MechET; research corpus)"
_STATE_PREFIX = "window.__PRELOADED_STATE__ = "
_CONTENT_TAGS = {"p", "li", "figcaption", "dt", "dd"}


def _state(document: BeautifulSoup) -> dict[str, Any]:
    for script in document.find_all("script"):
        text = script.get_text()
        if _STATE_PREFIX in text:
            return json.loads(text.split(_STATE_PREFIX, 1)[1].strip().rstrip(";"))
    raise ValueError("OpenStax page has no preloaded state")


def _plain_html(value: Any) -> str:
    return normalize_prose(BeautifulSoup(str(value or ""), "html.parser").get_text(" "))


def _walk_tree(value: Any, chapter: str = "") -> Iterable[dict[str, str]]:
    if isinstance(value, dict):
        current_chapter = chapter
        if value.get("toc_type") == "chapter":
            current_chapter = _plain_html(value.get("title"))
        if value.get("toc_type") == "book-content" and value.get("slug"):
            yield {
                "slug": str(value["slug"]),
                "title": _plain_html(value.get("title")),
                "toc_target_type": str(value.get("toc_target_type") or "content"),
                "chapter": current_chapter,
            }
        for child in value.get("contents") or []:
            yield from _walk_tree(child, current_chapter)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_tree(child, chapter)


def _page_role(row: dict[str, str]) -> str:
    target, slug = row["toc_target_type"], row["slug"]
    if target == "answer-key":
        return "evaluation_answer_key"
    if target == "index":
        return "excluded_index"
    if "additional-problems" in slug or "review-questions" in slug:
        return "evaluation_questions"
    if slug.endswith("key-terms") or slug.endswith("glossary"):
        return "excluded_glossary_duplicate"
    return "corpus"


def _node_text(node: Tag) -> str:
    clone_document = BeautifulSoup(str(node), "html.parser")
    clone = clone_document.find()
    if not isinstance(clone, Tag):
        return ""
    for hidden in clone.select(
        "script,style,noscript,svg,.os-caption-title-label,.os-number,.os-title-label"
    ):
        hidden.decompose()
    return clean_page_furniture(clone.get_text(" ", strip=True))


def extract_sections(main: Tag, *, page_title: str, section_kind: str) -> list[dict[str, Any]]:
    headings: list[tuple[int, str]] = []
    current_heading = page_title
    buckets: list[dict[str, Any]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        text = normalize_prose("\n\n".join(dict.fromkeys(current)))
        if text:
            buckets.append(
                {
                    "heading": current_heading,
                    "heading_path": [item[1] for item in headings],
                    "text": text,
                    "section_kind": section_kind,
                }
            )
        current = []

    for node in main.find_all(["h1", "h2", "h3", "h4", *_CONTENT_TAGS]):
        if any(parent.name in _CONTENT_TAGS for parent in node.parents if isinstance(parent, Tag)):
            continue
        if node.find_parent(attrs={"data-type": ["exercise-question", "question-solution"]}):
            continue
        if node.name and node.name.startswith("h"):
            flush()
            level = int(node.name[1])
            label = _node_text(node)
            headings = [item for item in headings if item[0] < level]
            if label:
                headings.append((level, label))
                current_heading = label
            continue
        text = _node_text(node)
        if len(text) >= 20:
            current.append(text)
    flush()
    return buckets


def extract_assessment_items(main: Tag, *, role: str) -> list[dict[str, Any]]:
    """Extract stable problem IDs without pretending images are plain text."""

    if role in {"evaluation_questions", "corpus"}:
        selector, item_kind = '[data-type="exercise-question"]', "question"
    elif role == "evaluation_answer_key":
        selector, item_kind = '[data-type="question-solution"]', "answer"
    else:
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in main.select(selector):
        number_node = node.select_one(".os-number")
        number = normalize_prose(number_node.get_text(" ", strip=True) if number_node else "")
        match = re.search(r"\b(\d+[-.]\d+)\b", number)
        problem_id = match.group(1).replace(".", "-") if match else ""
        if not problem_id or problem_id in seen:
            continue
        seen.add(problem_id)
        clone_document = BeautifulSoup(str(node), "html.parser")
        clone = clone_document.find()
        if not isinstance(clone, Tag):
            continue
        image_alts = [
            normalize_prose(str(image.get("alt") or ""))
            for image in clone.select("img")
            if normalize_prose(str(image.get("alt") or ""))
        ]
        has_figure = bool(clone.select("figure,img,svg,.os-figure,.os-figure-container"))
        for hidden in clone.select(
            "script,style,noscript,.os-prefix,.os-number,.os-title-label,.os-caption-title-label"
        ):
            hidden.decompose()
        for image in clone.select("img"):
            alt = normalize_prose(str(image.get("alt") or ""))
            image.replace_with(f" [IMAGE: {alt or 'untranscribed chemical figure'}] ")
        text = clean_page_furniture(clone.get_text(" ", strip=True))
        href_node = node.select_one("a.os-prefix[href]")
        output.append(
            {
                "problem_id": problem_id,
                "chapter_number": int(problem_id.split("-", 1)[0]),
                "kind": item_kind,
                "text": text,
                "has_figure": has_figure,
                "image_alts": image_alts,
                "source_anchor": str(href_node.get("href") or "") if href_node else "",
                "dom_id": str(node.get("id") or ""),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    return output


def _fetch(
    url: str,
    *,
    session: requests.Session,
    timeout: float,
    retries: int,
) -> tuple[bytes, dict[str, str]]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content, dict(response.headers)
        except requests.RequestException as exc:
            last = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last}") from last


def _extract_one(
    row: dict[str, str],
    *,
    base_url: str,
    output_root: Path,
    source_id: str,
    source: dict[str, Any],
    timeout: float,
    retries: int,
    user_agent: str,
    refresh_evaluation: bool,
    refresh_all: bool,
) -> dict[str, Any]:
    role = _page_role(row)
    url = f"{base_url}/{row['slug']}"
    path = output_root / source_id / "pages" / f"{row['slug']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    refresh = refresh_all or (refresh_evaluation and role.startswith("evaluation_"))
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        encoded = path.read_bytes()
    else:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "identity"})
        raw, headers = _fetch(url, session=session, timeout=timeout, retries=retries)
        document = BeautifulSoup(raw, "html.parser")
        state = _state(document)
        content = state["content"]
        book, page = content["book"], content["page"]
        main = document.select_one("#main-content")
        if main is None:
            raise ValueError(f"OpenStax page lacks #main-content: {url}")
        page_title = str(page.get("title") or row.get("title") or row["slug"])
        payload = {
            "title": page_title,
            "slug": row["slug"],
            "chapter": row.get("chapter"),
            "canonical_url": url,
            "page_id": page.get("id"),
            "book_id": book.get("id"),
            "book_version": book.get("version"),
            "content_version": book.get("contentVersion"),
            "archive_version": book.get("archiveVersion"),
            "revised": book.get("revised"),
            "license": book.get("license"),
            "toc_target_type": row["toc_target_type"],
            "page_role": role,
            "assessment_items": extract_assessment_items(main, role=role),
            "sections": extract_sections(
                main, page_title=page_title, section_kind=row["toc_target_type"]
            ),
            "raw_html_sha256": hashlib.sha256(raw).hexdigest(),
            "response_last_modified": headers.get("Last-Modified"),
        }
        encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode()
        path.write_bytes(encoded)
    status = "downloaded" if role == "corpus" else role
    quality = source_quality_metadata(source)
    return {
        "source_id": source_id,
        "path": str(path.relative_to(output_root / source_id)),
        "url": url,
        "canonical_url": url,
        "license": source.get("license"),
        "redistribution": source.get("redistribution"),
        "media_type": "application/json",
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "status": status,
        "revision": str(payload.get("content_version") or payload.get("book_version") or ""),
        "retrieval_backend": "openstax_rex_html",
        "configured_title": row.get("title"),
        "resolved_title": payload.get("title"),
        "chapter": row.get("chapter"),
        "toc_target_type": row["toc_target_type"],
        "content_characters": sum(
            len(str(section.get("text") or "")) for section in payload.get("sections") or []
        ),
        **quality,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("knowledge/source_registry.yaml"))
    parser.add_argument("--source-id", default="openstax_organic_chemistry_10e")
    parser.add_argument("--output", type=Path, default=Path("knowledge/raw_corpus_v2"))
    parser.add_argument("--accept-noncommercial", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--user-agent", default=DEFAULT_AGENT)
    parser.add_argument(
        "--refresh-evaluation",
        action="store_true",
        help="refetch exercise and answer pages to refresh problem-level extraction",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="refetch every page (needed after structural extractor changes)",
    )
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    source = dict(registry["sources"][args.source_id])
    if source.get("requires_noncommercial_acceptance") and not args.accept_noncommercial:
        raise PermissionError(
            f"{args.source_id} is {source.get('license')}; pass --accept-noncommercial "
            "for a separately marked non-commercial research snapshot"
        )
    seed = str(source["seed_url"])
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept-Encoding": "identity"})
    raw, _ = _fetch(seed, session=session, timeout=args.timeout, retries=args.retries)
    seed_document = BeautifulSoup(raw, "html.parser")
    seed_state = _state(seed_document)
    book = seed_state["content"]["book"]
    upstream_license = str((book.get("license") or {}).get("url") or "")
    if "by-nc-sa/4.0" not in upstream_license.lower():
        raise ValueError(f"unexpected OpenStax license: {book.get('license')}")
    rows = list(_walk_tree(book["tree"]))
    unique = {row["slug"]: row for row in rows}
    rows = [unique[key] for key in sorted(unique)]
    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]
    base_url = seed.rsplit("/", 1)[0]
    artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {
            pool.submit(
                _extract_one,
                row,
                base_url=base_url,
                output_root=args.output,
                source_id=args.source_id,
                source=source,
                timeout=args.timeout,
                retries=args.retries,
                user_agent=args.user_agent,
                refresh_evaluation=args.refresh_evaluation,
                refresh_all=args.refresh_all,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                artifacts.append(future.result())
            except Exception as exc:
                failures.append({"slug": row["slug"], "error": str(exc)})
    artifacts.sort(key=lambda row: str(row.get("path")))

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
            args.source_id: {
                "book_id": book.get("id"),
                "book_version": book.get("version"),
                "content_version": book.get("contentVersion"),
                "archive_version": book.get("archiveVersion"),
                "license": book.get("license"),
                "pages_discovered": len(rows),
                "pages_written": len(artifacts),
                "failures": failures,
            },
        },
        "candidate_evidence_only": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "pages_discovered": len(rows),
                "pages_written": len(artifacts),
                "failures": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
