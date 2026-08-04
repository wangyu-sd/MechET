#!/usr/bin/env python3
"""Check registered external source availability, revision, and content health."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.source_health import source_quality_metadata, validate_mediawiki_result

DOWNLOAD_SCRIPT = REPO / "scripts" / "download_mechanistic_sources.py"
spec = importlib.util.spec_from_file_location("mechet_source_health_download", DOWNLOAD_SCRIPT)
download = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = download
assert spec.loader is not None
spec.loader.exec_module(download)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(
    source_id: str,
    item: str,
    *,
    status: str,
    details: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "item": item,
        "status": status,
        "checked_at": _now(),
        "details": details or {},
        "error": error,
    }


def _check_mediawiki(
    source_id: str,
    row: dict[str, Any],
    options,
    backends: list[str] | None,
) -> list[dict[str, Any]]:
    records = []
    selected = backends or list(row.get("backend_order") or download.MEDIAWIKI_BACKENDS)
    for title in row.get("pages") or []:
        title = str(title)
        try:
            result = download._download_mediawiki_page(
                row, title, options, selected, None
            )
            backend = str(result.get("retrieval_backend") or "")
            validation = validate_mediawiki_result(
                result,
                configured_title=title,
                backend=backend,
            )
            quality = source_quality_metadata(row, title=title)
            status = (
                "warning"
                if quality["review_warning"]
                or quality["quality_status"] == "low_priority"
                else "healthy"
            )
            records.append(
                _record(
                    source_id,
                    title,
                    status=status,
                    details={
                        **validation,
                        **quality,
                        "retrieval_url": result.get("retrieval_url"),
                        "canonical_url": result.get("canonical_url"),
                        "backend_errors": result.get("backend_errors") or [],
                    },
                )
            )
        except Exception as exc:
            records.append(_record(source_id, title, status="error", error=str(exc)))
    return records


def _check_goldbook(source_id: str, row: dict[str, Any], options) -> list[dict[str, Any]]:
    records = []
    aliases = {str(k): str(v) for k, v in (row.get("term_aliases") or {}).items()}
    quality = source_quality_metadata(row)
    for configured in row.get("terms") or []:
        configured = str(configured)
        requested = aliases.get(configured, configured)
        url = str(row["base_url"]).format(term_id=requested)
        try:
            value, network = download.request_json(url, options=options)
            canonical = str((value.get("term") or {}).get("code") or "")
            if not canonical:
                raise ValueError("GOLDBOOK_CANONICAL_CODE_MISSING")
            records.append(
                _record(
                    source_id,
                    configured,
                    status="healthy",
                    details={
                        "configured_term_id": configured,
                        "requested_term_id": requested,
                        "canonical_term_id": canonical,
                        "final_url": network.get("final_url"),
                        **quality,
                    },
                )
            )
        except Exception as exc:
            records.append(
                _record(source_id, configured, status="error", error=str(exc))
            )
    return records


def _check_urls(source_id: str, row: dict[str, Any], options) -> list[dict[str, Any]]:
    records = []
    quality = source_quality_metadata(row)
    values = row.get("urls") or row.get("pages") or []
    for url in values:
        url = str(url)
        try:
            payload, headers, network = download.request(url, options=options)
            if not payload:
                raise ValueError("SOURCE_EMPTY_RESPONSE")
            status = (
                "warning"
                if quality["review_warning"]
                or quality["quality_status"] == "low_priority"
                else "healthy"
            )
            records.append(
                _record(
                    source_id,
                    url,
                    status=status,
                    details={
                        "final_url": network.get("final_url"),
                        "http_status": network.get("status"),
                        "bytes": len(payload),
                        "content_type": headers.get("content-type"),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        **quality,
                    },
                )
            )
        except Exception as exc:
            records.append(_record(source_id, url, status="error", error=str(exc)))
    return records


def check_source(
    source_id: str,
    row: dict[str, Any],
    options,
    backends: list[str] | None,
) -> list[dict[str, Any]]:
    mode = str(row.get("downloader") or "")
    if mode == "mediawiki_pages":
        return _check_mediawiki(source_id, row, options, backends)
    if mode == "iupac_goldbook_terms":
        return _check_goldbook(source_id, row, options)
    if mode in {"url", "html_pages"}:
        return _check_urls(source_id, row, options)
    if mode == "manual_gate":
        return [
            _record(
                source_id,
                str(row.get("request_url") or source_id),
                status="skipped_manual",
                details={**source_quality_metadata(row), "reason": "manual license gate"},
            )
        ]
    return [_record(source_id, source_id, status="error", error=f"unknown downloader {mode!r}")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("knowledge/source_registry.yaml"))
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("outputs/source_health.json"))
    parser.add_argument("--proxy", default=os.environ.get("MECHET_HTTPS_PROXY", ""))
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--user-agent", default=download.DEFAULT_USER_AGENT)
    parser.add_argument("--mediawiki-backend", action="append", default=[])
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "never"),
        default="error",
    )
    args = parser.parse_args()

    registry = download.load_registry(args.registry)
    sources = dict(registry["sources"])
    selected = list(args.source or sources)
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    backends = download._backend_list(args.mediawiki_backend)
    options = download.NetworkOptions(
        args.timeout,
        args.retries,
        args.backoff,
        args.proxy,
        args.user_agent,
    )
    records = [
        record
        for source_id in selected
        for record in check_source(source_id, sources[source_id], options, backends)
    ]
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("healthy", "warning", "error", "skipped_manual")
    }
    result = {
        "schema_version": 1,
        "artifact_type": "external_source_health_report",
        "created_at": _now(),
        "registry": str(args.registry),
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "selected_sources": selected,
        "network": {
            "proxy_configured": bool(args.proxy),
            "timeout": args.timeout,
            "retries": args.retries,
            "mediawiki_backends": backends or "registry_default",
        },
        "counts": counts,
        "records": records,
        "healthy": counts["error"] == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.fail_on == "never":
        return 0
    if counts["error"]:
        return 1
    if args.fail_on == "warning" and counts["warning"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
