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
from mechet.textbook_cleaning import (
    CleanSection,
    quality_flags,
    structured_sections,
    text_sections,
    wikitext_sections,
)


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
    "bonding_and_structure": ("orbital", "hybridization", "chemical bond", "resonance"),
    "acid_base": ("acid", "base", "pka", "proton donor", "proton acceptor"),
    "substitution": ("substitution", "nucleophilic displacement", "leaving group"),
    "elimination": ("elimination", "beta hydrogen", "e2", "e1"),
    "carbonyl": ("carbonyl", "aldehyde", "ketone", "acyl"),
    "addition": ("addition", "nucleophilic attack", "electrophilic attack"),
    "aromatic": ("aromatic", "rearomatization", "arenium"),
    "proton_transfer": ("proton transfer", "protonation", "deprotonation"),
    "oxidation_reduction": ("oxidation", "reduction", "hydride"),
    "radical": ("radical", "homolytic", "radical chain", "single electron"),
    "photochemistry": ("photochemical", "photochemistry", "excited state", "photosensit"),
    "electrochemistry": ("electrochemical", "electrolysis", "electrode", "redox potential"),
    "pericyclic": ("pericyclic", "diels-alder", "cycloaddition", "sigmatropic", "electrocyclic"),
    "rearrangement": ("rearrangement", "migration", "migratory aptitude"),
    "stereochemistry": ("stereochemistry", "inversion", "retention", "enantio"),
    "organometallic": ("organometallic", "grignard", "organolithium", "cross-coupling"),
    "heterocycle": ("heterocycle", "heterocyclic", "pyridine", "furan", "indole"),
    "polymer": ("polymer", "polymerization", "monomer"),
    "biomolecule": ("amino acid", "peptide", "carbohydrate", "lipid", "nucleic acid"),
    "kinetics": ("rate law", "activation energy", "transition state", "kinetic"),
    "thermodynamics": ("free energy", "enthalpy", "entropy", "thermodynamic"),
    "solvent_effects": (
        "solvent effect", "polar solvent", "protic solvent", "aprotic solvent",
        "solvation", "solvated",
    ),
    "mass_spectrometry": ("mass spectrom", "mass spectrum", "mass-to-charge", "m/z"),
    "ionization": ("ionization", "electron ionization", "electrospray", "chemical ionization"),
    "fragmentation": ("fragmentation", "fragment ion", "neutral loss", "molecular ion"),
    "spectroscopy": ("spectroscopy", "infrared", "nmr", "ultraviolet", "uv-vis"),
    "chromatography": ("chromatography", "retention time", "stationary phase", "mobile phase"),
}

_FUNCTIONAL_GROUP_KEYWORDS = {
    "alkane": ("alkane",), "alkene": ("alkene", "olefin"),
    "alkyne": ("alkyne",), "aromatic": ("arene", "aromatic", "benzene"),
    "alkyl_halide": ("alkyl halide", "haloalkane"),
    "alcohol": ("alcohol", "hydroxyl"), "ether": ("ether",),
    "epoxide": ("epoxide", "oxirane"), "amine": ("amine",),
    "aldehyde": ("aldehyde",), "ketone": ("ketone",),
    "carboxylic_acid": ("carboxylic acid",), "ester": ("ester",),
    "amide": ("amide",), "nitrile": ("nitrile", "cyano"),
    "nitro": ("nitro",), "thiol": ("thiol",),
    "organometallic": ("organometallic", "grignard", "organolithium"),
}

_PHASE_KEYWORDS = {
    "solution_phase": (
        "in solution", "aqueous", "solvent", "dissolved", "protic", "aprotic",
    ),
    "gas_phase": (
        "gas phase", "gas-phase", "molecular ion", "electron ionization",
        "chemical ionization", "mass spectrom", "ion source", "vacuum",
    ),
    "condensed_phase": ("solid state", "liquid phase", "crystal", "melt"),
    "surface_phase": ("surface reaction", "adsorbed", "heterogeneous catalyst"),
}

_MODALITY_KEYWORDS = {
    "mechanism": ("mechanism", "electron pair", "curved arrow", "transition state"),
    "synthesis": ("synthesis", "preparation", "yield"),
    "spectroscopy": ("spectrum", "spectroscopy", "chemical shift", "absorption"),
    "mass_spectrometry": ("mass spectrom", "m/z", "fragment ion", "molecular ion"),
    "kinetics": ("rate constant", "rate law", "activation energy"),
    "thermodynamics": ("free energy", "enthalpy", "entropy", "equilibrium constant"),
    "physical_properties": ("boiling point", "melting point", "solubility"),
}
_EXCLUDED_SECTION_RE = re.compile(
    r"^(?:references?|external links?|further reading|bibliography|licensing|"
    r"articles?(?: and web pages.*)?|citations?(?:\b| for))",
    re.IGNORECASE,
)


def _collect_strings(value: Any, *, key: str = "") -> Iterable[str]:
    if isinstance(value, str):
        # Downloaded JSON contains long provenance fields (for example Gold Book
        # citations, licence text and disclaimers) alongside the evidence text.
        # Only explicit content-bearing keys belong in the retrieval corpus.
        if key.lower() in _TEXT_KEYS:
            yield value
        return
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _collect_strings(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _collect_strings(child, key=key)


def _artifact_sections(path: Path) -> tuple[str, list[CleanSection]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        term = payload.get("term") if isinstance(payload, dict) else None
        term = term if isinstance(term, dict) else {}
        title = (
            str(payload.get("title") or term.get("title") or term.get("name") or path.stem)
            if isinstance(payload, dict)
            else path.stem
        )
        if isinstance(payload, dict) and isinstance(payload.get("sections"), list):
            return title, structured_sections(payload["sections"], default_heading=title)
        if isinstance(payload, dict) and payload.get("wikitext"):
            return title, wikitext_sections(str(payload["wikitext"]), default_heading=title)
        parts = [item.strip() for item in _collect_strings(payload) if item.strip()]
        return title, text_sections("\n\n".join(dict.fromkeys(parts)), default_heading=title)
    return path.stem, text_sections(raw, default_heading=path.stem)


def _artifact_text(path: Path) -> tuple[str, str]:
    """Compatibility helper retained for extraction tests and small utilities."""

    title, sections = _artifact_sections(path)
    return title, "\n\n".join(section.text for section in sections)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allowed_licenses(spec: dict[str, Any] | None) -> set[str] | None:
    if spec is None:
        return None
    output: set[str] = set()
    for layer in dict(spec.get("layers") or {}).values():
        if not isinstance(layer, dict) or layer.get("include_in_corpus") is False:
            continue
        output.update(map(str, layer.get("allowed_licenses") or ()))
    if not output:
        raise ValueError("corpus spec declares no allowed licenses")
    return output


def _artifact_path(download_root: Path, source_id: str, relative: str) -> Path:
    root = download_root.resolve()
    path = (root / source_id / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes download root: {source_id}/{relative}")
    return path


def _chunks(text: str, *, minimum: int, maximum: int, overlap: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    output: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > maximum:
            if current and len(current) >= minimum:
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
        if any(_contains_phrase(lowered, phrase) for phrase in phrases)
    )


def _contains_phrase(lowered: str, phrase: str) -> bool:
    return bool(
        re.search(
            r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])",
            lowered,
        )
    )


def _labels(text: str, vocabulary: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        label
        for label, phrases in vocabulary.items()
        if any(_contains_phrase(lowered, phrase) for phrase in phrases)
    )


def _quality_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_status": artifact.get("quality_status"),
        "retrieval_weight": artifact.get("retrieval_weight"),
        "review_warning": artifact.get("review_warning"),
        "last_human_reviewed_at": artifact.get("last_human_reviewed_at"),
        "scientific_scope": list(artifact.get("scientific_scope") or []),
        "allowed_uses": list(artifact.get("allowed_uses") or []),
        "disallowed_uses": list(artifact.get("disallowed_uses") or []),
        "quality_notes": artifact.get("quality_notes"),
    }


def build_with_report(
    download_root: Path,
    *,
    minimum: int,
    maximum: int,
    overlap: int,
    allowed_licenses: set[str] | None = None,
    strict_artifacts: bool = False,
) -> tuple[TextbookStore, dict[str, Any]]:
    if minimum <= 0 or maximum < minimum or overlap < 0 or overlap >= maximum:
        raise ValueError(
            "chunking requires 0 < minimum <= maximum and 0 <= overlap < maximum"
        )
    manifest_path = download_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    passages: list[TextbookPassage] = []
    exact_texts: set[str] = set()
    report: dict[str, Any] = {
        "artifacts_seen": 0,
        "artifacts_used": 0,
        "sections_seen": 0,
        "sections_used": 0,
        "chunks_rejected_quality": 0,
        "chunks_rejected_duplicate": 0,
        "quality_flags": {},
        "artifact_errors": [],
    }
    artifacts = sorted(
        payload.get("artifacts") or [],
        key=lambda row: (str(row.get("source_id") or ""), str(row.get("path") or "")),
    )
    for artifact in artifacts:
        report["artifacts_seen"] += 1
        if artifact.get("status") not in (None, "downloaded", "offline_import"):
            report.setdefault("artifacts_rejected_status", 0)
            report["artifacts_rejected_status"] += 1
            continue
        source_id = str(artifact.get("source_id") or "")
        relative = str(artifact.get("path") or "")
        license_name = str(artifact.get("license") or "")
        if allowed_licenses is not None and license_name not in allowed_licenses:
            raise ValueError(
                f"artifact license outside corpus protocol: {source_id}/{relative}: "
                f"{license_name!r}"
            )
        path = _artifact_path(download_root, source_id, relative)
        if not source_id or not relative or not path.exists():
            message = f"missing artifact: {source_id}/{relative}"
            report["artifact_errors"].append(message)
            if strict_artifacts:
                raise FileNotFoundError(message)
            continue
        if path.suffix.lower() not in {".json", ".txt", ".xml", ".md"}:
            report.setdefault("artifacts_rejected_media_type", 0)
            report["artifacts_rejected_media_type"] += 1
            continue
        expected_artifact_sha = str(artifact.get("sha256") or "")
        if not expected_artifact_sha:
            raise ValueError(f"artifact lacks SHA-256: {source_id}/{relative}")
        actual_artifact_sha = _sha256_file(path)
        if actual_artifact_sha != expected_artifact_sha:
            raise ValueError(
                f"artifact hash mismatch: {source_id}/{relative}: "
                f"declared={expected_artifact_sha}, observed={actual_artifact_sha}"
            )
        try:
            title, sections = _artifact_sections(path)
        except Exception as exc:
            message = f"artifact extraction failed: {source_id}/{relative}: {exc}"
            report["artifact_errors"].append(message)
            if strict_artifacts:
                raise ValueError(message) from exc
            continue
        report["artifacts_used"] += 1
        report["sections_seen"] += len(sections)
        for section_index, section in enumerate(sections):
            if _EXCLUDED_SECTION_RE.search(section.heading.strip()):
                report.setdefault("sections_rejected_heading", 0)
                report["sections_rejected_heading"] += 1
                continue
            section_text = _normalize(section.text)
            if not section_text:
                continue
            report["sections_used"] += 1
            for chunk_index, chunk in enumerate(
                _chunks(section_text, minimum=minimum, maximum=maximum, overlap=overlap)
            ):
                flags = quality_flags(chunk)
                for flag in flags:
                    report["quality_flags"][flag] = report["quality_flags"].get(flag, 0) + 1
                if set(flags) & {
                    "page_furniture", "wiki_markup", "low_alpha_fraction", "url_heavy"
                }:
                    report["chunks_rejected_quality"] += 1
                    continue
                normalized_key = re.sub(r"\W+", " ", chunk.lower()).strip()
                normalized_digest = hashlib.sha256(normalized_key.encode()).hexdigest()
                if normalized_digest in exact_texts:
                    report["chunks_rejected_duplicate"] += 1
                    continue
                exact_texts.add(normalized_digest)
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                annotation_text = f"{title} {section.heading} {chunk}"
                base_locator = str(
                    artifact.get("canonical_url")
                    or artifact.get("url")
                    or relative
                )
                locator_suffix = section.locator_suffix or section.heading
                locator = base_locator + (f"#{locator_suffix}" if locator_suffix else "")
                passages.append(
                    TextbookPassage(
                        passage_id=f"{source_id}:{digest[:16]}:{section_index}:{chunk_index}",
                        title=(
                            title
                            if not section.heading or section.heading == title
                            else f"{title} — {section.heading}"
                        ),
                        text=chunk,
                        source_id=source_id,
                        locator=locator,
                        revision=str(
                            artifact.get("revision_id")
                            or artifact.get("revision")
                            or artifact.get("canonical_term_id")
                            or ""
                        ),
                        license=str(artifact.get("license") or "unknown"),
                        source_url=str(artifact.get("url") or ""),
                        evidence_sha256=digest,
                        topics=_topics(annotation_text),
                        reaction_families=_topics(annotation_text),
                        functional_groups=_labels(annotation_text, _FUNCTIONAL_GROUP_KEYWORDS),
                        phases=_labels(annotation_text, _PHASE_KEYWORDS) or ("unspecified",),
                        modalities=_labels(annotation_text, _MODALITY_KEYWORDS) or ("textbook_explanation",),
                        metadata={
                            "artifact_path": relative,
                            "artifact_sha256": artifact.get("sha256"),
                            "retrieval_backend": artifact.get("retrieval_backend"),
                            "document_title": title,
                            "section_heading": section.heading,
                            "section_kind": section.section_kind,
                            "extraction_quality_flags": list(flags),
                            **_quality_metadata(artifact),
                        },
                    )
                )
    report["passages_written"] = len(passages)
    return TextbookStore(passages), report


def build(
    download_root: Path,
    *,
    minimum: int,
    maximum: int,
    overlap: int,
) -> TextbookStore:
    store, _ = build_with_report(
        download_root, minimum=minimum, maximum=maximum, overlap=overlap
    )
    return store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, default=Path("knowledge/raw"))
    parser.add_argument("--output", type=Path, default=Path("knowledge/corpus/passages.jsonl"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--spec",
        type=Path,
        help="optional license-layer allowlist (for example knowledge/corpus_v2_spec.yaml)",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="fail on missing or unparseable eligible artifacts; hash mismatches always fail",
    )
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=160)
    args = parser.parse_args()

    spec = None
    if args.spec:
        import yaml

        spec = yaml.safe_load(args.spec.read_text(encoding="utf-8")) or {}
    store, build_report = build_with_report(
        args.download_root,
        minimum=args.min_chars,
        maximum=args.max_chars,
        overlap=args.overlap_chars,
        allowed_licenses=_allowed_licenses(spec),
        strict_artifacts=args.strict_artifacts,
    )
    store.save(args.output)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                **store.manifest(),
                "corpus_file_sha256": _sha256_file(args.output),
                "download_root": str(args.download_root),
                "source_manifest_sha256": hashlib.sha256(
                    (args.download_root / "manifest.json").read_bytes()
                ).hexdigest(),
                "source_registry_sha256": json.loads(
                    (args.download_root / "manifest.json").read_text(encoding="utf-8")
                ).get("registry_sha256"),
                "corpus_spec": str(args.spec) if args.spec else None,
                "corpus_spec_sha256": _sha256_file(args.spec) if args.spec else None,
                "builder_sha256": _sha256_file(Path(__file__)),
                "chunking": {
                    "min_chars": args.min_chars,
                    "max_chars": args.max_chars,
                    "overlap_chars": args.overlap_chars,
                },
                "build_report": build_report,
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
