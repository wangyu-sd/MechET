import importlib.util
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "download_mechanistic_sources.py"
spec = importlib.util.spec_from_file_location("mechet_source_download_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_registry_uses_current_goldbook_code():
    registry = module.load_registry(REPO / "knowledge" / "source_registry.yaml")
    source = registry["sources"]["iupac_goldbook_terms"]
    assert "I03096" in source["terms"]
    assert "R05171" not in source["terms"]
    assert source["term_aliases"]["R05171"] == "I03096"


def test_registry_uses_existing_wikibooks_carbonyl_chapter():
    registry = module.load_registry(REPO / "knowledge" / "source_registry.yaml")
    pages = registry["sources"]["wikibooks_organic_chemistry"]["pages"]
    assert "Organic Chemistry/Ketones and aldehydes" in pages
    assert "Organic Chemistry/Carbonyls" not in pages


def test_goldbook_uses_canonical_code(monkeypatch, tmp_path):
    def fake_json(url, *, options, params=None):
        assert url.endswith("/I03096/json")
        return ({"term": {"code": "I03096", "title": "intermediate"}}, {
            "requested_url": url,
            "final_url": "https://goldbook.iupac.org/terms/view/I03096/json",
            "redirected": False,
        })

    monkeypatch.setattr(module, "request_json", fake_json)
    row = {
        "base_url": "https://goldbook.iupac.org/terms/view/{term_id}/json",
        "terms": ["R05171"],
        "term_aliases": {"R05171": "I03096"},
        "license": "CC-BY-SA-4.0-individual-terms",
        "redistribution": "individual_terms_only",
    }
    records = module.goldbook("iupac", row, tmp_path, module.NetworkOptions())
    assert (tmp_path / "terms" / "I03096.json").exists()
    assert records[0]["configured_term_id"] == "R05171"
    assert records[0]["requested_term_id"] == "I03096"
    assert records[0]["canonical_term_id"] == "I03096"
    assert records[0]["alias_resolved"]


def test_mediawiki_falls_back_to_export(monkeypatch):
    monkeypatch.setattr(
        module,
        "_mediawiki_rest",
        lambda *args: (_ for _ in ()).throw(RuntimeError("rest blocked")),
    )
    monkeypatch.setattr(
        module,
        "_mediawiki_action",
        lambda *args: (_ for _ in ()).throw(RuntimeError("api blocked")),
    )
    monkeypatch.setattr(
        module,
        "_mediawiki_export",
        lambda row, title, options: {
            "title": title,
            "wikitext": "A revisioned textbook passage describing a mechanistic pathway and its electron-flow context. " * 2,
            "revision_id": 7,
            "retrieval_backend": "export",
            "retrieval_url": "https://example/export",
        },
    )
    result = module._download_mediawiki_page(
        {},
        "Organic Chemistry/Test",
        module.NetworkOptions(),
        ["rest", "action_api", "export", "raw"],
        None,
    )
    assert result["retrieval_backend"] == "export"
    assert [item["backend"] for item in result["backend_errors"]] == [
        "rest",
        "action_api",
    ]


def test_mediawiki_local_import_works_without_network(tmp_path):
    title = "Organic Chemistry/Ketones and aldehydes"
    (tmp_path / f"{module.slug(title)}.txt").write_text(
        "A locally imported textbook passage with sufficient mechanistic content for validation. " * 2, encoding="utf-8"
    )
    result = module._download_mediawiki_page(
        {"license": "CC-BY-SA"},
        title,
        module.NetworkOptions(),
        ["rest"],
        tmp_path,
    )
    assert result["retrieval_backend"] == "local_import"
    assert "locally imported textbook passage" in result["wikitext"]


def test_mediawiki_failure_is_actionable(monkeypatch):
    for name in (
        "_mediawiki_rest",
        "_mediawiki_action",
        "_mediawiki_export",
        "_mediawiki_raw",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args: (_ for _ in ()).throw(RuntimeError("blocked")),
        )
    with pytest.raises(RuntimeError, match="--proxy") as error:
        module._download_mediawiki_page(
            {},
            "Organic Chemistry/Test",
            module.NetworkOptions(),
            ["rest", "action_api", "export", "raw"],
            None,
        )
    assert "--mediawiki-import-dir" in str(error.value)


def test_backend_list_accepts_comma_and_repeat():
    assert module._backend_list(["rest,export", "raw"]) == [
        "rest",
        "export",
        "raw",
    ]
    with pytest.raises(ValueError):
        module._backend_list(["unknown"])

def test_mediawiki_invalid_success_falls_through_to_next_backend(monkeypatch):
    monkeypatch.setattr(
        module,
        "_mediawiki_rest",
        lambda row, title, options: {
            "title": title,
            "revision_id": None,
            "wikitext": "",
            "retrieval_backend": "rest",
        },
    )
    monkeypatch.setattr(
        module,
        "_mediawiki_export",
        lambda row, title, options: {
            "title": title,
            "revision_id": 9,
            "wikitext": "A valid exported mechanistic textbook passage with enough content for the common validator. " * 2,
            "retrieval_backend": "export",
            "retrieval_url": "https://example/export",
        },
    )
    result = module._download_mediawiki_page(
        {},
        "Organic Chemistry/Ketones and aldehydes",
        module.NetworkOptions(),
        ["rest", "export"],
        None,
    )
    assert result["retrieval_backend"] == "export"
    assert result["validation"]["content_nonempty"]
    assert result["backend_errors"][0]["backend"] == "rest"

