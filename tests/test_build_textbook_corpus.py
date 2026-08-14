from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_textbook_corpus.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_textbook_corpus", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_extraction_excludes_long_provenance_fields(tmp_path: Path) -> None:
    builder = _load_builder()
    artifact = tmp_path / "term.json"
    artifact.write_text(
        json.dumps(
            {
                "term": {
                    "title": "electrophile",
                    "definitions": [{"text": "Evidence definition " * 20}],
                    "citation": "Citation metadata " * 20,
                    "license": "Licence metadata " * 20,
                    "disclaimer": "Disclaimer metadata " * 20,
                }
            }
        ),
        encoding="utf-8",
    )

    title, text = builder._artifact_text(artifact)

    assert title == "electrophile"
    assert "Evidence definition" in text
    assert "Citation metadata" not in text
    assert "Licence metadata" not in text
    assert "Disclaimer metadata" not in text


def test_topic_matching_uses_term_boundaries() -> None:
    builder = _load_builder()
    assert "addition" not in builder._topics("One additional hydrogen is present.")
    assert "addition" in builder._topics("Nucleophilic addition occurs.")


def _raw_download(tmp_path: Path, *, license_name: str = "CC-BY-4.0") -> Path:
    root = tmp_path / "raw"
    artifact = root / "source" / "page.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "title": "Nucleophilic substitution",
                "text": (
                    "A nucleophile donates an electron pair during nucleophilic "
                    "substitution while a leaving group departs from carbon. " * 4
                ),
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "artifacts": [
            {
                "source_id": "source",
                "path": "page.json",
                "status": "downloaded",
                "license": license_name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_builder_rejects_raw_artifact_hash_mismatch(tmp_path: Path) -> None:
    builder = _load_builder()
    root = _raw_download(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["artifacts"][0]["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        builder.build_with_report(root, minimum=80, maximum=1200, overlap=120)


def test_builder_enforces_protocol_license_allowlist(tmp_path: Path) -> None:
    builder = _load_builder()
    root = _raw_download(tmp_path, license_name="UNKNOWN")

    with pytest.raises(ValueError, match="license outside corpus protocol"):
        builder.build_with_report(
            root,
            minimum=80,
            maximum=1200,
            overlap=120,
            allowed_licenses={"CC-BY-4.0"},
        )
