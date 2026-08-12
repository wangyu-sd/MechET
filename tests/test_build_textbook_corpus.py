from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
