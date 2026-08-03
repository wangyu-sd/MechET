#!/usr/bin/env python3
"""Download license-gated, provenance-tracked mechanistic Web references."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import quote, urlparse
import xml.etree.ElementTree as ET

try:
    import requests
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install mechet[knowledge]") from exc

DEFAULT_USER_AGENT = "MechET-Primitive-Library/0.2 (+https://github.com/wangyu-sd/MechET)"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MEDIAWIKI_BACKENDS = ("rest", "action_api", "export", "raw")


@dataclass(frozen=True)
class NetworkOptions:
    timeout: float = 60.0
    retries: int = 3
    backoff: float = 1.0
    proxy: str = ""
    user_agent: str = DEFAULT_USER_AGENT

    @property
    def proxies(self) -> dict[str, str] | None:
        return {"http": self.proxy, "https": self.proxy} if self.proxy else None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:100] or "item"


def load_registry(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value.get("sources"), dict):
        raise ValueError("source registry requires a sources mapping")
    return value


def request(
    url: str,
    *,
    options: NetworkOptions,
    params: dict[str, Any] | None = None,
    accept: str = "*/*",
) -> tuple[bytes, dict[str, str], dict[str, Any]]:
    headers = {
        "User-Agent": options.user_agent,
        "Api-User-Agent": options.user_agent,
        "Accept": accept,
        "Accept-Encoding": "identity",
    }
    last_error: Exception | None = None
    attempts = max(options.retries, 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=options.timeout,
                proxies=options.proxies,
                allow_redirects=True,
            )
            if response.status_code in RETRYABLE_STATUS and attempt < attempts:
                delay = response.headers.get("Retry-After")
                time.sleep(float(delay) if delay and delay.isdigit() else options.backoff * 2 ** (attempt - 1))
                continue
            response.raise_for_status()
            return response.content, dict(response.headers), {
                "requested_url": response.request.url,
                "final_url": response.url,
                "status": response.status_code,
                "attempts": attempt,
                "redirected": bool(response.history),
            }
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(options.backoff * 2 ** (attempt - 1))
    hint = (
        f" configured proxy={options.proxy!r}."
        if options.proxy
        else " Set HTTPS_PROXY/HTTP_PROXY or pass --proxy if a local proxy is required."
    )
    raise RuntimeError(f"download failed for {url} after {attempts} attempt(s): {last_error}.{hint}") from last_error


def request_json(
    url: str,
    *,
    options: NetworkOptions,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, _, metadata = request(url, options=options, params=params, accept="application/json")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {metadata['final_url']}")
    return value, metadata


def html_text(payload: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install mechet[knowledge]") from exc
    soup = BeautifulSoup(payload, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    return "\n".join(item.strip() for item in soup.get_text("\n").splitlines() if item.strip())


def gate(source_id: str, row: dict, noncommercial: bool, restricted: bool) -> None:
    if row.get("requires_noncommercial_acceptance") and not noncommercial:
        raise PermissionError(f"{source_id} requires --accept-noncommercial")
    if row.get("requires_restricted_acceptance") and not restricted:
        raise PermissionError(f"{source_id} requires --accept-restricted")


def write(
    root: Path,
    relative: str,
    payload: bytes,
    source_id: str,
    row: dict,
    url: str,
    media_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
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


def goldbook(source_id: str, row: dict, root: Path, options: NetworkOptions, **_: Any) -> list[dict]:
    output = []
    aliases = {str(k): str(v) for k, v in (row.get("term_aliases") or {}).items()}
    for configured_id in row.get("terms") or []:
        configured_id = str(configured_id)
        request_id = aliases.get(configured_id, configured_id)
        url = str(row["base_url"]).format(term_id=request_id)
        response, network = request_json(url, options=options)
        canonical_id = str((response.get("term") or {}).get("code") or request_id)
        payload = json.dumps(response, indent=2, ensure_ascii=False).encode()
        output.append(write(
            root, f"terms/{canonical_id}.json", payload, source_id, row,
            str(network["final_url"]), "application/json",
            {
                "configured_term_id": configured_id,
                "requested_term_id": request_id,
                "canonical_term_id": canonical_id,
                "alias_resolved": configured_id != request_id,
                "redirected": bool(network.get("redirected")) or configured_id != request_id or canonical_id != request_id,
                "requested_url": network.get("requested_url"),
            },
        ))
    return output


def urls(source_id: str, row: dict, root: Path, options: NetworkOptions, **_: Any) -> list[dict]:
    output = []
    for index, url in enumerate(row.get("urls") or []):
        payload, headers, network = request(str(url), options=options)
        name = Path(urlparse(str(url)).path).name or f"download_{index}"
        output.append(write(
            root, name, payload, source_id, row, str(network["final_url"]),
            headers.get("content-type", "application/octet-stream"),
            {"requested_url": network.get("requested_url")},
        ))
    return output


def _mediawiki_action(row: dict, title: str, options: NetworkOptions) -> dict[str, Any]:
    response, network = request_json(str(row["api_url"]), options=options, params={
        "action": "query", "prop": "revisions|info", "rvprop": "ids|timestamp|content",
        "rvslots": "main", "inprop": "url", "format": "json", "formatversion": 2,
        "redirects": 1, "titles": title,
    })
    page = ((response.get("query") or {}).get("pages") or [{}])[0]
    if page.get("missing"):
        raise FileNotFoundError(title)
    revision = (page.get("revisions") or [{}])[0]
    return {
        "title": page.get("title") or title,
        "page_id": page.get("pageid"),
        "canonical_url": page.get("canonicalurl"),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "license": row.get("license"),
        "wikitext": ((revision.get("slots") or {}).get("main") or {}).get("content", ""),
        "retrieval_backend": "action_api",
        "retrieval_url": network.get("final_url"),
    }


def _mediawiki_rest(row: dict, title: str, options: NetworkOptions) -> dict[str, Any]:
    encoded = quote(title.replace(" ", "_"), safe="")
    template = str(row.get("rest_api_url") or "https://en.wikibooks.org/w/rest.php/v1/page/{title}")
    response, network = request_json(template.format(title=encoded), options=options)
    latest, license_info = response.get("latest") or {}, response.get("license") or {}
    return {
        "title": response.get("title") or title,
        "page_id": response.get("id"),
        "canonical_url": response.get("html_url"),
        "revision_id": latest.get("id"),
        "revision_timestamp": latest.get("timestamp"),
        "license": license_info.get("title") or row.get("license"),
        "license_url": license_info.get("url"),
        "wikitext": response.get("source") or "",
        "retrieval_backend": "rest",
        "retrieval_url": network.get("final_url"),
    }


def _find_xml(root: ET.Element, name: str) -> str | None:
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == name:
            return node.text
    return None


def _parse_export(payload: bytes, row: dict, title: str, retrieval_url: str, backend: str) -> dict[str, Any]:
    document = ET.fromstring(payload)
    page_title = _find_xml(document, "title") or title
    page_id = _find_xml(document, "id")
    revision = next((node for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "revision"), None)
    if revision is None:
        raise ValueError(f"MediaWiki export contains no revision: {title}")
    revision_id = _find_xml(revision, "id")
    wikitext = _find_xml(revision, "text") or ""
    if not wikitext:
        raise ValueError(f"MediaWiki export contains no wikitext: {title}")
    return {
        "title": page_title,
        "page_id": int(page_id) if page_id and page_id.isdigit() else page_id,
        "canonical_url": None,
        "revision_id": int(revision_id) if revision_id and revision_id.isdigit() else revision_id,
        "revision_timestamp": _find_xml(revision, "timestamp"),
        "license": row.get("license"),
        "wikitext": wikitext,
        "retrieval_backend": backend,
        "retrieval_url": retrieval_url,
    }


def _mediawiki_export(row: dict, title: str, options: NetworkOptions) -> dict[str, Any]:
    encoded = quote(title.replace(" ", "_"), safe="/:")
    template = str(row.get("export_url") or "https://en.wikibooks.org/wiki/Special:Export/{title}")
    payload, _, network = request(
        template.format(title=encoded), options=options,
        accept="application/xml,text/xml;q=0.9,*/*;q=0.1",
    )
    value = _parse_export(payload, row, title, str(network["final_url"]), "export")
    page_template = str(row.get("page_url") or "https://en.wikibooks.org/wiki/{title}")
    value["canonical_url"] = page_template.format(title=quote(value["title"].replace(" ", "_"), safe="/:"))
    return value


def _mediawiki_raw(row: dict, title: str, options: NetworkOptions) -> dict[str, Any]:
    encoded = quote(title.replace(" ", "_"), safe="")
    template = str(row.get("raw_url") or "https://en.wikibooks.org/w/index.php?title={title}&action=raw")
    payload, _, network = request(template.format(title=encoded), options=options, accept="text/plain,*/*;q=0.1")
    wikitext = payload.decode("utf-8", errors="replace")
    if not wikitext.strip():
        raise ValueError(f"action=raw returned an empty page: {title}")
    return {
        "title": title, "page_id": None, "canonical_url": None,
        "revision_id": None, "revision_timestamp": None, "license": row.get("license"),
        "wikitext": wikitext, "retrieval_backend": "raw",
        "retrieval_url": network.get("final_url"),
        "revision_note": "revision ID unavailable from raw fallback; content hash is retained",
    }


def _local_mediawiki(row: dict, title: str, directory: Path | None) -> dict[str, Any] | None:
    if directory is None:
        return None
    stem = slug(title)
    for suffix in (".json", ".xml", ".txt"):
        path = directory / f"{stem}{suffix}"
        if not path.exists():
            continue
        if suffix == ".xml":
            return _parse_export(path.read_bytes(), row, title, str(path), "local_import")
        if suffix == ".txt":
            return {
                "title": title, "page_id": None, "canonical_url": None,
                "revision_id": None, "revision_timestamp": None, "license": row.get("license"),
                "wikitext": path.read_text(encoding="utf-8"),
                "retrieval_backend": "local_import", "retrieval_url": str(path),
            }
        value = json.loads(path.read_text(encoding="utf-8"))
        latest = value.get("latest") or {}
        return {
            "title": value.get("title") or title,
            "page_id": value.get("page_id", value.get("id")),
            "canonical_url": value.get("canonical_url", value.get("html_url")),
            "revision_id": value.get("revision_id", latest.get("id")),
            "revision_timestamp": value.get("revision_timestamp", latest.get("timestamp")),
            "license": value.get("license") or row.get("license"),
            "wikitext": value.get("wikitext", value.get("source", "")),
            "retrieval_backend": "local_import", "retrieval_url": str(path),
        }
    return None


def _download_mediawiki_page(
    row: dict, title: str, options: NetworkOptions,
    backends: list[str], import_dir: Path | None,
) -> dict[str, Any]:
    local = _local_mediawiki(row, title, import_dir)
    if local is not None:
        return local
    functions: dict[str, Callable[[dict, str, NetworkOptions], dict[str, Any]]] = {
        "rest": _mediawiki_rest, "action_api": _mediawiki_action,
        "export": _mediawiki_export, "raw": _mediawiki_raw,
    }
    errors = []
    for backend in backends:
        try:
            result = functions[backend](row, title, options)
            result["backend_errors"] = errors
            return result
        except Exception as exc:
            errors.append({"backend": backend, "error": str(exc)})
    raise RuntimeError(
        f"all MediaWiki backends failed for {title!r}: {errors}. "
        "Try --proxy http://127.0.0.1:PORT or set HTTPS_PROXY. "
        "If Wikimedia remains unreachable, download Special:Export XML elsewhere, "
        f"save it as {slug(title)}.xml, and pass --mediawiki-import-dir DIR."
    )


def mediawiki(
    source_id: str, row: dict, root: Path, options: NetworkOptions, *,
    mediawiki_backends: list[str] | None = None,
    mediawiki_import_dir: Path | None = None, **_: Any,
) -> list[dict]:
    backends = mediawiki_backends or list(row.get("backend_order") or MEDIAWIKI_BACKENDS)
    invalid = sorted(set(backends) - set(MEDIAWIKI_BACKENDS))
    if invalid:
        raise ValueError(f"unknown MediaWiki backends: {invalid}")
    output = []
    for title in row.get("pages") or []:
        value = _download_mediawiki_page(row, str(title), options, backends, mediawiki_import_dir)
        payload = json.dumps(value, indent=2, ensure_ascii=False).encode()
        output.append(write(
            root, f"pages/{slug(str(title))}.json", payload, source_id, row,
            str(value.get("canonical_url") or value.get("retrieval_url") or ""),
            "application/json",
            {"revision_id": value.get("revision_id"), "retrieval_backend": value.get("retrieval_backend")},
        ))
    return output


def html_pages(source_id: str, row: dict, root: Path, options: NetworkOptions, **_: Any) -> list[dict]:
    expected = [str(x).lower() for x in row.get("expected_license_markers") or []]
    excluded = [str(x).lower() for x in row.get("exclude_markers") or []]
    output = []
    for index, url in enumerate(row.get("pages") or []):
        raw, headers, network = request(str(url), options=options)
        text = html_text(raw)
        lowered = text.lower()
        verified = not expected or any(x in lowered for x in expected)
        hits = [x for x in excluded if x in lowered]
        status = "downloaded" if verified and not hits else ("excluded_marker_detected" if hits else "license_marker_missing")
        value = {
            "url": network.get("final_url"), "requested_url": network.get("requested_url"),
            "license": row.get("license"), "status": status,
            "license_marker_verified": verified, "excluded_marker_hits": hits,
            "raw_html_sha256": sha(raw), "text": text if status == "downloaded" else "",
        }
        payload = json.dumps(value, indent=2, ensure_ascii=False).encode()
        output.append(write(
            root, f"pages/{index:03d}_{slug(urlparse(str(url)).path)}.json", payload,
            source_id, row, str(network.get("final_url") or url), "application/json",
            {"status": status, "content_type": headers.get("content-type")},
        ))
    return output


def manual(source_id: str, row: dict, root: Path, options: NetworkOptions, **_: Any) -> list[dict]:
    del options
    text = (
        f"# Manual download required: {row.get('title', source_id)}\n\n"
        f"License: {row.get('license', '')}\n\nRequest URL: {row.get('request_url', '')}\n\n"
        "MechET does not automate acceptance of this license or redistribute modified derivatives. "
        "Download the unmodified archive through the upstream request flow and record its local hash.\n"
    )
    return [write(root, "MANUAL_DOWNLOAD_REQUIRED.md", text.encode(), source_id, row,
                  str(row.get("request_url") or ""), "text/markdown", {"status": "manual_required"})]


def download_one(
    source_id: str, row: dict, output: Path, options: NetworkOptions, *,
    mediawiki_backends: list[str] | None = None,
    mediawiki_import_dir: Path | None = None,
) -> list[dict]:
    mode, root = str(row.get("downloader") or ""), output / source_id
    functions = {
        "iupac_goldbook_terms": goldbook, "url": urls,
        "mediawiki_pages": mediawiki, "html_pages": html_pages,
        "manual_gate": manual,
    }
    if mode not in functions:
        raise ValueError(f"unknown downloader {mode!r} for {source_id}")
    return functions[mode](
        source_id, row, root, options,
        mediawiki_backends=mediawiki_backends,
        mediawiki_import_dir=mediawiki_import_dir,
    )


def verify(root: Path) -> dict[str, Any]:
    rows = (json.loads((root / "manifest.json").read_text()) or {}).get("artifacts") or []
    failures, checked = [], 0
    for row in rows:
        if not row.get("path") or not row.get("sha256"):
            continue
        path = root / str(row.get("source_id") or "") / str(row["path"])
        checked += 1
        if not path.exists() or sha(path.read_bytes()) != row["sha256"]:
            failures.append(str(path))
    return {"checked": checked, "failed": len(failures), "failures": failures}


def _backend_list(values: list[str]) -> list[str] | None:
    if not values or values == ["auto"]:
        return None
    output = [item.strip() for value in values for item in value.split(",") if item.strip()]
    invalid = sorted(set(output) - set(MEDIAWIKI_BACKENDS))
    if invalid:
        raise ValueError(f"unknown MediaWiki backends: {invalid}")
    return output


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
    download.add_argument("--proxy", default=os.environ.get("MECHET_HTTPS_PROXY", ""))
    download.add_argument("--timeout", type=float, default=60.0)
    download.add_argument("--retries", type=int, default=3)
    download.add_argument("--backoff", type=float, default=1.0)
    download.add_argument("--user-agent", default=os.environ.get("MECHET_USER_AGENT", DEFAULT_USER_AGENT))
    download.add_argument(
        "--mediawiki-backend", action="append", default=[],
        help="auto or repeat/comma-separate rest,action_api,export,raw",
    )
    download.add_argument(
        "--mediawiki-import-dir", type=Path,
        help="directory containing slug(title).json/.xml/.txt for offline import",
    )
    sub.add_parser("verify").add_argument("--output", type=Path, default=Path("knowledge/raw"))
    args = parser.parse_args()
    sources = dict(load_registry(args.registry)["sources"])
    if args.command == "list":
        rows = {k: {x: v.get(x) for x in ("title", "source_type", "license", "downloader", "redistribution")} for k, v in sources.items()}
        print(json.dumps(rows, indent=2, ensure_ascii=False) if args.json else "\n".join(
            f"{k}\t{v['license']}\t{v['downloader']}\t{v['title']}" for k, v in rows.items()
        ))
        return 0
    if args.command == "verify":
        result = verify(args.output)
        print(json.dumps(result, indent=2))
        return int(result["failed"] > 0)
    selected = list(args.source or sources)
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    plan = []
    for source_id in selected:
        row = sources[source_id]
        gate(source_id, row, args.accept_noncommercial, args.accept_restricted)
        plan.append({
            "source_id": source_id, "title": row.get("title"), "license": row.get("license"),
            "downloader": row.get("downloader"), "output": str(args.output / source_id),
        })
    if args.dry_run:
        print(json.dumps({"plan": plan}, indent=2, ensure_ascii=False))
        return 0
    options = NetworkOptions(args.timeout, args.retries, args.backoff, args.proxy, args.user_agent)
    backends, artifacts = _backend_list(args.mediawiki_backend), []
    for item in plan:
        artifacts.extend(download_one(
            item["source_id"], sources[item["source_id"]], args.output, options,
            mediawiki_backends=backends, mediawiki_import_dir=args.mediawiki_import_dir,
        ))
    manifest = {
        "schema_version": 1, "registry": str(args.registry),
        "registry_sha256": sha(args.registry.read_bytes()), "created_at": now(),
        "sources": selected,
        "network": {
            "proxy_configured": bool(args.proxy), "timeout": args.timeout,
            "retries": args.retries, "mediawiki_backends": backends or "registry_default",
        },
        "artifacts": artifacts, "candidate_evidence_only": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"written": len(artifacts), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
