from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from mechet.textbook_store import TextbookPassage, TextbookStore


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "extract_textbook_knowledge.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("extract_textbook_knowledge", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passage(
    passage_id: str,
    *,
    license_name: str,
    topic: str,
    phase: str = "solution_phase",
) -> TextbookPassage:
    text = f"A bounded explanation of {topic} with enough source context."
    import hashlib

    return TextbookPassage(
        passage_id=passage_id,
        title=topic,
        text=text,
        source_id="source-a",
        locator=f"https://example.test/{passage_id}",
        revision="7",
        license=license_name,
        source_url="https://example.test",
        evidence_sha256=hashlib.sha256(text.encode()).hexdigest(),
        topics=(topic,),
        phases=(phase,),
        modalities=("mechanism",) if topic == "substitution" else ("spectroscopy",),
        metadata={"section_kind": "content", "artifact_sha256": "abc"},
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus = tmp_path / "passages.jsonl"
    store = TextbookStore(
        [
            _passage(
                "redistributable",
                license_name="CC-BY-4.0",
                topic="substitution",
            ),
            _passage(
                "noncommercial",
                license_name="CC-BY-NC-SA-4.0",
                topic="mass_spectrometry",
                phase="gas_phase",
            ),
        ]
    )
    store.save(corpus)
    corpus_manifest = tmp_path / "passages.manifest.json"
    corpus_manifest.write_text(json.dumps(store.manifest()), encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        """layers:
  redistributable:
    allowed_licenses: [CC-BY-4.0]
  noncommercial_research:
    allowed_licenses: [CC-BY-NC-SA-4.0]
    release_separately: true
  external_reference_only:
    include_in_corpus: false
""",
        encoding="utf-8",
    )
    return corpus, corpus_manifest, spec


def test_prepare_requires_explicit_noncommercial_acceptance(tmp_path: Path) -> None:
    module = _load_script()
    corpus, corpus_manifest, spec = _write_inputs(tmp_path)
    output = tmp_path / "tasks.jsonl"
    args = argparse.Namespace(
        corpus=corpus,
        corpus_manifest=corpus_manifest,
        spec=spec,
        output=output,
        manifest=None,
        accept_noncommercial=False,
        limit=0,
    )

    assert module.prepare(args) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["passage_id"] for row in rows] == ["redistributable"]
    assert rows[0]["released_knowledge_anchor"] is False
    assert rows[0]["passage_sha256"] in rows[0]["messages"][1]["content"] or (
        rows[0]["evidence_span"] in rows[0]["messages"][1]["content"]
    )
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["skipped"] == {
        "noncommercial_requires_explicit_acceptance": 1
    }


def test_prepare_includes_all_layers_only_after_acceptance(tmp_path: Path) -> None:
    module = _load_script()
    corpus, corpus_manifest, spec = _write_inputs(tmp_path)
    output = tmp_path / "tasks.jsonl"
    args = argparse.Namespace(
        corpus=corpus,
        corpus_manifest=corpus_manifest,
        spec=spec,
        output=output,
        manifest=None,
        accept_noncommercial=True,
        limit=0,
    )

    module.prepare(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["passage_id"] for row in rows} == {
        "redistributable",
        "noncommercial",
    }
    by_id = {row["passage_id"]: row for row in rows}
    assert by_id["noncommercial"]["source"]["license_layer"] == (
        "noncommercial_research"
    )
    assert by_id["noncommercial"]["extraction_profile"] == (
        "gas_phase_and_mass_spectrometry"
    )


def test_prepare_rejects_license_outside_protocol(tmp_path: Path) -> None:
    module = _load_script()
    corpus, corpus_manifest, spec = _write_inputs(tmp_path)
    rows = corpus.read_text(encoding="utf-8").splitlines()
    value = json.loads(rows[0])
    value["license"] = "UNKNOWN"
    corpus.write_text(json.dumps(value) + "\n", encoding="utf-8")
    bad_store = TextbookStore.load(corpus)
    corpus_manifest.write_text(json.dumps(bad_store.manifest()), encoding="utf-8")
    args = argparse.Namespace(
        corpus=corpus,
        corpus_manifest=corpus_manifest,
        spec=spec,
        output=tmp_path / "tasks.jsonl",
        manifest=None,
        accept_noncommercial=False,
        limit=0,
    )

    with pytest.raises(ValueError, match="outside protocol allowlist"):
        module.prepare(args)


def test_balanced_limit_covers_profiles_before_repeating() -> None:
    module = _load_script()
    tasks = [
        {"candidate_id": "a1", "extraction_profile": "a"},
        {"candidate_id": "a2", "extraction_profile": "a"},
        {"candidate_id": "b1", "extraction_profile": "b"},
        {"candidate_id": "c1", "extraction_profile": "c"},
    ]
    selected = module._balanced_prefix(tasks, 3)
    assert [row["candidate_id"] for row in selected] == ["a1", "b1", "c1"]


def test_response_schema_requires_explicit_unknowns() -> None:
    module = _load_script()
    value = {
        "candidate_name": "UNKNOWN",
        "knowledge_type": "UNKNOWN",
        "explicit_claims": [],
        "participants": [],
        "electron_moves": [],
        "preconditions": [],
        "warnings_or_exceptions": [],
        "competing_pathways": [],
        "stereochemical_effects": [],
        "phase_and_medium": {
            "phase": "UNKNOWN",
            "solvent_or_medium": "UNKNOWN",
            "temperature_or_pressure": "UNKNOWN",
            "support": "absent",
        },
        "analytical_context": {
            "ionization_method": "UNKNOWN",
            "ion_or_radical_state": "UNKNOWN",
            "instrument_or_collision_context": "UNKNOWN",
            "support": "absent",
        },
        "uncertain_fields": [],
        "review_notes": [],
    }
    assert module._validate_extraction_schema(value) == []
    del value["phase_and_medium"]["support"]
    assert "phase_and_medium keys mismatch" in " ".join(
        module._validate_extraction_schema(value)
    )
