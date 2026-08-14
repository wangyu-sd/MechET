"""Source-aware cleaning for provenance-preserving textbook extraction.

The downloader deliberately keeps source payloads close to the upstream
representation.  This module is the boundary that turns those payloads into
model-visible prose.  It removes navigation, wiki syntax, page furniture and
reference markup while retaining headings and chemically meaningful text.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Iterable, Mapping


_HEADING_RE = re.compile(r"(?m)^\s*(={2,6})\s*(.*?)\s*\1\s*$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_DROP_BLOCK_RE = re.compile(
    r"<(noinclude|ref|gallery|imagemap|timeline|references)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DROP_SINGLE_RE = re.compile(
    r"<\s*(references|ref|br|hr)\b[^>]*/?\s*>", re.IGNORECASE
)
_FILE_LINK_RE = re.compile(
    r"\[\[(?:File|Image|Category):.*?\]\]", re.IGNORECASE | re.DOTALL
)
_EXTERNAL_LINK_RE = re.compile(r"\[(?:https?://|//)[^\s\]]+(?:\s+([^\]]+))?\]")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
_TAG_RE = re.compile(r"</?[^>]+>")
_MAGIC_RE = re.compile(r"__\w+__")
_TABLE_RE = re.compile(r"(?ms)^\s*\{\|.*?^\s*\|\}\s*$")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")

_UI_LINES = {
    "skip to main content",
    "table of contents",
    "login",
    "sign in",
    "search",
    "toolbar",
    "exit reader mode",
    "back to top",
    "was this article helpful?",
    "yes",
    "no",
    "download as pdf",
    "download page (pdf)",
    "request instructor account",
}
_NOISE_PATTERNS = (
    "mindtouch.deki.logic",
    "extensionprocessorqueryprovider",
    "property get [map",
    "powered by nice",
    "selected template will load here",
    "this page has no tags",
    "enable dyslexic font",
    "privacy policy",
    "terms & conditions",
)


@dataclass(frozen=True)
class CleanSection:
    """One source section before bounded passage chunking."""

    heading: str
    text: str
    locator_suffix: str = ""
    section_kind: str = "content"


def normalize_prose(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_templates(text: str) -> str:
    """Remove nested templates without allowing them to swallow whole pages."""

    previous = None
    while previous != text:
        previous = text
        text = _TEMPLATE_RE.sub(" ", text)
    return text.replace("{{", " ").replace("}}", " ")


def _fallback_strip_wikicode(text: str) -> str:
    text = _COMMENT_RE.sub(" ", text)
    text = _DROP_BLOCK_RE.sub(" ", text)
    text = _DROP_SINGLE_RE.sub("\n", text)
    text = _TABLE_RE.sub(" ", text)
    text = _FILE_LINK_RE.sub(" ", text)
    text = _remove_templates(text)
    text = _WIKI_LINK_RE.sub(lambda match: match.group(2) or match.group(1), text)
    text = _EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or " ", text)
    text = text.replace("'''", "").replace("''", "")
    text = _MAGIC_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return text


def strip_wikicode(text: str) -> str:
    """Render MediaWiki source as conservative plain text.

    ``mwparserfromhell`` handles nesting when the knowledge extra is installed.
    The fallback intentionally prefers dropping an unfamiliar template over
    exposing template instructions or navigation to the model.
    """

    prepared = _COMMENT_RE.sub(" ", str(text or ""))
    prepared = _DROP_BLOCK_RE.sub(" ", prepared)
    prepared = _FILE_LINK_RE.sub(" ", prepared)
    try:
        import mwparserfromhell  # type: ignore

        rendered = mwparserfromhell.parse(prepared).strip_code(
            normalize=True, collapse=True, keep_template_params=False
        )
    except ImportError:
        rendered = _fallback_strip_wikicode(prepared)
    return clean_page_furniture(rendered)


def clean_page_furniture(text: str) -> str:
    kept: list[str] = []
    for raw_line in normalize_prose(text).splitlines():
        line = raw_line.strip(" \t*#:;|-")
        lowered = line.lower()
        if not line or lowered in _UI_LINES:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if any(pattern in lowered for pattern in _NOISE_PATTERNS):
            continue
        if lowered.startswith(("category:", "file:", "image:")):
            continue
        if re.fullmatch(r"[+x\s]+", line, re.IGNORECASE):
            continue
        kept.append(line)
    return normalize_prose("\n".join(kept))


def wikitext_sections(text: str, *, default_heading: str) -> list[CleanSection]:
    matches = list(_HEADING_RE.finditer(str(text or "")))
    output: list[CleanSection] = []
    start = 0
    heading = default_heading
    heading_path: list[tuple[int, str]] = []
    for match in [*matches, None]:
        end = match.start() if match is not None else len(text)
        rendered = strip_wikicode(text[start:end])
        if rendered:
            path = " / ".join(item[1] for item in heading_path)
            output.append(
                CleanSection(
                    heading=heading,
                    text=rendered,
                    locator_suffix=path,
                )
            )
        if match is None:
            break
        level = len(match.group(1))
        label = strip_wikicode(match.group(2)) or default_heading
        heading_path = [item for item in heading_path if item[0] < level]
        heading_path.append((level, label))
        heading = label
        start = match.end()
    return output


def structured_sections(
    rows: Iterable[Mapping[str, Any]], *, default_heading: str
) -> list[CleanSection]:
    output: list[CleanSection] = []
    for index, row in enumerate(rows):
        text = clean_page_furniture(str(row.get("text") or ""))
        if not text:
            continue
        heading = clean_page_furniture(str(row.get("heading") or default_heading))
        path = str(row.get("locator_suffix") or row.get("heading_path") or "")
        if isinstance(row.get("heading_path"), list):
            path = " / ".join(map(str, row["heading_path"]))
        output.append(
            CleanSection(
                heading=heading or default_heading,
                text=text,
                locator_suffix=path or f"section-{index}",
                section_kind=str(row.get("section_kind") or "content"),
            )
        )
    return output


def text_sections(text: str, *, default_heading: str) -> list[CleanSection]:
    cleaned = clean_page_furniture(text)
    return [CleanSection(default_heading, cleaned)] if cleaned else []


def quality_flags(text: str) -> tuple[str, ...]:
    """Return deterministic extraction warnings used by the corpus audit."""

    lowered = str(text or "").lower()
    flags: list[str] = []
    if any(marker in lowered for marker in _NOISE_PATTERNS):
        flags.append("page_furniture")
    if re.search(r"\[\[|\]\]|\{\{|\}\}|<ref\b|</?noinclude", text, re.I):
        flags.append("wiki_markup")
    alpha = sum(character.isalpha() for character in text)
    if text and alpha / len(text) < 0.45:
        flags.append("low_alpha_fraction")
    if len(set(re.findall(r"[a-z]+", lowered))) < 12:
        flags.append("low_lexical_diversity")
    urls = re.findall(r"(?:https?://|www\.)\S+", text, re.I)
    if len(urls) >= 2 or (urls and sum(map(len, urls)) / max(len(text), 1) > 0.08):
        flags.append("url_heavy")
    return tuple(flags)


__all__ = [
    "CleanSection",
    "clean_page_furniture",
    "normalize_prose",
    "quality_flags",
    "strip_wikicode",
    "structured_sections",
    "text_sections",
    "wikitext_sections",
]
