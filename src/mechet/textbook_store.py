"""Provenance-aware natural-language textbook passage storage."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TextbookPassage:
    passage_id: str
    title: str
    text: str
    source_id: str
    locator: str
    revision: str
    license: str
    source_url: str
    evidence_sha256: str
    topics: tuple[str, ...] = ()
    reaction_families: tuple[str, ...] = ()
    functional_groups: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    @classmethod
    def parse(cls, row: Mapping[str, Any]) -> "TextbookPassage":
        text = str(row.get("text") or "").strip()
        digest = str(row.get("evidence_sha256") or "") or hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        return cls(
            passage_id=str(row.get("passage_id") or ""),
            title=str(row.get("title") or ""),
            text=text,
            source_id=str(row.get("source_id") or ""),
            locator=str(row.get("locator") or ""),
            revision=str(row.get("revision") or ""),
            license=str(row.get("license") or ""),
            source_url=str(row.get("source_url") or ""),
            evidence_sha256=digest,
            topics=tuple(map(str, row.get("topics") or ())),
            reaction_families=tuple(map(str, row.get("reaction_families") or ())),
            functional_groups=tuple(map(str, row.get("functional_groups") or ())),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["topics"] = list(self.topics)
        value["reaction_families"] = list(self.reaction_families)
        value["functional_groups"] = list(self.functional_groups)
        return value


class TextbookStore:
    def __init__(self, passages: Iterable[TextbookPassage]) -> None:
        self.passages = tuple(passages)
        self.by_id = {item.passage_id: item for item in self.passages}
        self.validate()

    @classmethod
    def load(cls, path: str | Path) -> "TextbookStore":
        rows = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(TextbookPassage.parse(json.loads(line)))
        return cls(rows)

    def validate(self) -> None:
        if not self.passages:
            raise ValueError("textbook passage store is empty")
        if len(self.by_id) != len(self.passages):
            raise ValueError("duplicate textbook passage IDs")
        for item in self.passages:
            if not item.passage_id or not item.text or not item.source_id:
                raise ValueError(f"invalid passage: {item.passage_id}")
            if not item.license or not item.evidence_sha256:
                raise ValueError(f"passage lacks license/hash: {item.passage_id}")
            actual = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
            if actual != item.evidence_sha256:
                raise ValueError(f"passage hash mismatch: {item.passage_id}")

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for item in self.passages:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def manifest(self) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        licenses: dict[str, int] = {}
        for item in self.passages:
            source_counts[item.source_id] = source_counts.get(item.source_id, 0) + 1
            licenses[item.license] = licenses.get(item.license, 0) + 1
        payload = "\n".join(sorted(item.evidence_sha256 for item in self.passages))
        return {
            "n_passages": len(self.passages),
            "source_counts": source_counts,
            "license_counts": licenses,
            "corpus_digest": hashlib.sha256(payload.encode()).hexdigest(),
        }
