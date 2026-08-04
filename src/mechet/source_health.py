"""Validation and quality metadata for external mechanistic sources."""
from __future__ import annotations

from typing import Any, Mapping


_ALLOWED_QUALITY = {"reviewed", "usable_with_caution", "low_priority"}
_SOFT_MISSING_MARKERS = (
    "there is currently no text in this page",
    "the page you requested does not exist",
    "badtitle",
    "missingtitle",
)


def normalize_mediawiki_title(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).casefold()


def source_quality_metadata(
    source: Mapping[str, Any],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """Merge source-level and optional page-level evidence quality metadata."""

    quality = dict(source.get("quality") or {})
    page_quality = dict(source.get("page_quality") or {})
    if title is not None and isinstance(page_quality.get(title), Mapping):
        quality.update(dict(page_quality[title]))
    status = str(quality.get("quality_status") or "usable_with_caution")
    if status not in _ALLOWED_QUALITY:
        raise ValueError(f"unknown quality_status: {status}")
    weight = float(quality.get("retrieval_weight", 1.0))
    if not 0.0 <= weight <= 1.0:
        raise ValueError("retrieval_weight must be in [0, 1]")
    return {
        "quality_status": status,
        "retrieval_weight": weight,
        "review_warning": bool(quality.get("review_warning", False)),
        "last_human_reviewed_at": quality.get("last_human_reviewed_at"),
        "scientific_scope": [str(item) for item in quality.get("scientific_scope") or []],
        "allowed_uses": [str(item) for item in quality.get("allowed_uses") or []],
        "disallowed_uses": [
            str(item) for item in quality.get("disallowed_uses") or []
        ],
        "quality_notes": str(quality.get("quality_notes") or ""),
    }


def validate_mediawiki_result(
    result: Mapping[str, Any],
    *,
    configured_title: str,
    backend: str,
    minimum_characters: int = 80,
) -> dict[str, Any]:
    """Validate one backend result before it can enter the source manifest."""

    resolved_title = str(result.get("title") or "").strip()
    wikitext = str(result.get("wikitext") or "")
    if not resolved_title:
        raise ValueError(f"MEDIAWIKI_TITLE_MISSING:{configured_title}:{backend}")

    lowered = wikitext.casefold()
    marker = next((item for item in _SOFT_MISSING_MARKERS if item in lowered), None)
    if marker:
        raise ValueError(
            f"MEDIAWIKI_SOFT_MISSING_PAGE:{configured_title}:{backend}:{marker}"
        )
    if wikitext.lstrip().casefold().startswith("#redirect"):
        raise ValueError(
            f"MEDIAWIKI_UNRESOLVED_REDIRECT:{configured_title}:{backend}:"
            f"{resolved_title}"
        )
    if len(wikitext.strip()) < int(minimum_characters):
        raise ValueError(
            f"MEDIAWIKI_CONTENT_TOO_SHORT:{configured_title}:{backend}:"
            f"{len(wikitext.strip())}"
        )

    revision_id = result.get("revision_id")
    if backend not in {"raw", "local_import"} and revision_id in (None, ""):
        raise ValueError(
            f"MEDIAWIKI_REVISION_MISSING:{configured_title}:{backend}"
        )
    return {
        "configured_title": configured_title,
        "resolved_title": resolved_title,
        "title_changed": normalize_mediawiki_title(configured_title)
        != normalize_mediawiki_title(resolved_title),
        "revision_id": revision_id,
        "revision_timestamp": result.get("revision_timestamp"),
        "content_characters": len(wikitext),
        "content_nonempty": True,
        "backend": backend,
    }
