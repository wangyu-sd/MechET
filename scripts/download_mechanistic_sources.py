#!/usr/bin/env python3
"""Download license-gated, provenance-tracked mechanistic Web references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install mechet[knowledge]") from exc

USER_AGENT = "MechET-Primitive-Library/0.1 (+https://github.com/wangyu-sd/MechET)"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_registry(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value.get("sources"), dict):
        raise ValueError("source registry requires a sources mapping")
    return value


def fetch(url: str, params: dict | None = None) -> tuple[bytes, dict[str, str]]:
    if params:
        url += ("&" if "?" in url else "?") + urlencode(params)
    try:
        with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as response:
            return response.read(), {k.lower(): v for k, v in response.headers.items()}
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc


def fetch_json(url: str, params: dict | None = None) -> dict:
    return dict(json.loads(fetch(url, params)[0]))


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:100] or "item"


def html_text(payload: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install mechet[knowledge]") from exc
    soup = BeautifulSoup(payload, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    return "\n".join(x.strip() for x in soup.get_text("\n").splitlines() if x.strip())


def gate(source_id: str, row: dict, noncommercial: bool, restricted: bool) -> None:
    if row.get("requires_noncommercial_acceptance") and not noncommercial:
        raise PermissionError(f"{source_id} requires --accept-noncommercial")
    if row.get("requires_restricted_acceptance") and not restricted:
        raise PermissionError(f"{source_id} requires --accept-restricted")


def write(root: Path, relative: str, payload: bytes, source_id: str, row: dict, url: str, media_type: str, extra: dict | None = None) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    record = {
        "source_id": source_id,
        "path": str(path.relative_to(root)),
        "url": url,
        "license": row.get("license"),
        "redistribution": row.get("redistribution"),
        "media_type": media_type,
        "bytes": len(payload),
        "sha256": sha(payload),
        "downloaded_at": now(),
    }
    record.update(extra or {})
    return record


def goldbook(source_id: str, row: dict, root: Path) -> list[dict]:
    output = []
    for term_id in row.get("terms") or []:
        url = str(row["base_url"]).format(term_id=term_id)
        payload = json.dumps(fetch_json(url), indent=2, ensure_ascii=False).encode()
        output.append(write(root, f"terms/{term_id}.json", payload, source_id, row, url, "application/json", {"term_id": str(term_id)}))
    return output


def urls(source_id: str, row: dict, root: Path) -> list[dict]:
    output = []
    for index, url in enumerate(row.get("urls") or []):
        payload, headers = fetch(str(url))
        name = Path(urlparse(str(url)).path).name or f"download_{index}"
        output.append(write(root, name, payload, source_id, row, str(url), headers.get("content-type", "application/octet-stream")))
    return output


def mediawiki(source_id: str, row: dict, root: Path) -> list[dict]:
    output = []
    for title in row.get("pages") or []:
        response = fetch_json(str(row["api_url"]), {
            "action": "query", "prop": "revisions|info", "rvprop": "ids|timestamp|content",
            "rvslots": "main", "inprop": "url", "format": "json", "formatversion": 2,
            "titles": title,
        })
        page = ((response.get("query") or {}).get("pages") or [{}])[0]
        if page.get("missing"):
            output.append({"source_id": source_id, "title": title, "status": "missing", "license": row.get("license")})
            continue
        revision = (page.get("revisions") or [{}])[0]
        value = {
            "title": page.get("title") or title,
            "page_id": page.get("pageid"),
            "canonical_url": page.get("canonicalurl"),
            "revision_id": revision.get("revid"),
            "revision_timestamp": revision.get("timestamp"),
            "license": row.get("license"),
            "wikitext": ((revision.get("slots") or {}).get("main") or {}).get("content", ""),
        }
        payload = json.dumps(value, indent=2, ensure_ascii=False).encode()
        output.append(write(root, f"pages/{slug(str(title))}.json", payload, source_id, row, str(page.get("canonicalurl") or row["api_url"]), "application/json", {"revision_id": revision.get("revid")}))
    return output


def html_pages(source_id: str, row: dict, root: Path) -> list[dict]:
    expected = [str(x).lower() for x in row.get("expected_license_markers") or []]
    excluded = [str(x).lower() for x in row.get("exclude_markers") or []]
    output = []
    for index, url in enumerate(row.get("pages") or []):
        raw, headers = fetch(str(url))
        text = html_text(raw)
        lowered = text.lower()
        verified = not expected or any(x in lowered for x in expected)
        hits = [x for x in excluded if x in lowered]
        status = "downloaded" if verified and not hits else ("excluded_marker_detected" if hits else "license_marker_missing")
        value = {
            "url": url, "license": row.get("license"), "status": status,
            "license_marker_verified": verified, "excluded_marker_hits": hits,
            "raw_html_sha256": sha(raw), "text": text if status == "downloaded" else "",
        }
        payload = json.dumps(value, indent=2, ensure_ascii=False).encode()
        output.append(write(root, f"pages/{index:03d}_{slug(urlparse(str(url)).path)}.json", payload, source_id, row, str(url), "application/json", {"status": status, "content_type": headers.get("content-type")}))
    return output


def manual(source_id: str, row: dict, root: Path) -> list[dict]:
    text = (
        f"# Manual download required: {row.get('title', source_id)}\n\n"
        f"License: {row.get('license', '')}\n\nRequest URL: {row.get('request_url', '')}\n\n"
        "MechET does not automate acceptance of this license or redistribute modified derivatives. "
        "Download the unmodified archive through the upstream request flow and record its local hash.\n"
    )
    return [write(root, "MANUAL_DOWNLOAD_REQUIRED.md", text.encode(), source_id, row, str(row.get("request_url") or ""), "text/markdown", {"status": "manual_required"})]


def download_one(source_id: str, row: dict, output: Path) -> list[dict]:
    mode, root = str(row.get("downloader") or ""), output / source_id
    return {
        "iupac_goldbook_terms": goldbook,
        "url": urls,
        "mediawiki_pages": mediawiki,
        "html_pages": html_pages,
        "manual_gate": manual,
    }[mode](source_id, row, root)


def verify(root: Path) -> dict:
    manifest = root / "manifest.json"
    rows = (json.loads(manifest.read_text()) or {}).get("artifacts") or []
    failures, checked = [], 0
    for row in rows:
        if not row.get("path") or not row.get("sha256"): continue
        path = root / str(row.get("source_id") or "") / str(row["path"])
        checked += 1
        if not path.exists() or sha(path.read_bytes()) != row["sha256"]:
            failures.append(str(path))
    return {"checked": checked, "failed": len(failures), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("knowledge/source_registry.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").add_argument("--json", action="store_true")
    download = sub.add_parser("download")
    download.add_argument("--source", action="append", default=[])
    download.add_argument("--output", type=Path, default=Path("knowledge/raw"))
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--accept-noncommercial", action="store_true")
    download.add_argument("--accept-restricted", action="store_true")
    sub.add_parser("verify").add_argument("--output", type=Path, default=Path("knowledge/raw"))
    args = parser.parse_args()
    registry = load_registry(args.registry)
    sources = dict(registry["sources"])
    if args.command == "list":
        rows = {k: {x: v.get(x) for x in ("title", "source_type", "license", "downloader", "redistribution")} for k, v in sources.items()}
        print(json.dumps(rows, indent=2, ensure_ascii=False) if args.json else "\n".join(f"{k}\t{v['license']}\t{v['downloader']}\t{v['title']}" for k, v in rows.items()))
        return 0
    if args.command == "verify":
        result = verify(args.output); print(json.dumps(result, indent=2)); return int(result["failed"] > 0)
    selected = list(args.source or sources)
    unknown = sorted(set(selected) - set(sources))
    if unknown: raise ValueError(f"unknown source IDs: {unknown}")
    plan = []
    for source_id in selected:
        row = sources[source_id]
        gate(source_id, row, args.accept_noncommercial, args.accept_restricted)
        plan.append({"source_id": source_id, "title": row.get("title"), "license": row.get("license"), "downloader": row.get("downloader"), "output": str(args.output / source_id)})
    if args.dry_run:
        print(json.dumps({"plan": plan}, indent=2, ensure_ascii=False)); return 0
    artifacts = []
    for item in plan: artifacts.extend(download_one(item["source_id"], sources[item["source_id"]], args.output))
    manifest = {"schema_version": 1, "registry": str(args.registry), "registry_sha256": sha(args.registry.read_bytes()), "created_at": now(), "sources": selected, "artifacts": artifacts, "candidate_evidence_only": True}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"written": len(artifacts), "manifest": str(args.output / 'manifest.json')}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
