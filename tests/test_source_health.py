import pytest

from mechet.source_health import (
    source_quality_metadata,
    validate_mediawiki_result,
)


def test_mediawiki_validator_accepts_revisioned_content():
    result = validate_mediawiki_result(
        {
            "title": "Organic Chemistry/Ketones and aldehydes",
            "revision_id": 123,
            "revision_timestamp": "2026-01-01T00:00:00Z",
            "wikitext": "Carbonyl compounds include aldehydes and ketones. " * 3,
        },
        configured_title="Organic Chemistry/Ketones and aldehydes",
        backend="export",
    )
    assert result["content_nonempty"]
    assert result["revision_id"] == 123


def test_mediawiki_validator_rejects_empty_soft_missing_and_missing_revision():
    with pytest.raises(ValueError, match="CONTENT_TOO_SHORT"):
        validate_mediawiki_result(
            {"title": "Test", "revision_id": 1, "wikitext": "short"},
            configured_title="Test",
            backend="rest",
        )
    with pytest.raises(ValueError, match="SOFT_MISSING_PAGE"):
        validate_mediawiki_result(
            {
                "title": "Test",
                "revision_id": 1,
                "wikitext": "There is currently no text in this page. " * 4,
            },
            configured_title="Test",
            backend="rest",
        )
    with pytest.raises(ValueError, match="REVISION_MISSING"):
        validate_mediawiki_result(
            {"title": "Test", "wikitext": "valid textbook content " * 8},
            configured_title="Test",
            backend="action_api",
        )


def test_mediawiki_validator_rejects_unresolved_redirect():
    with pytest.raises(ValueError, match="UNRESOLVED_REDIRECT"):
        validate_mediawiki_result(
            {
                "title": "Old title",
                "revision_id": 4,
                "wikitext": "#REDIRECT [[New title]]" + " " * 100,
            },
            configured_title="Old title",
            backend="export",
        )


def test_page_quality_overrides_source_quality():
    source = {
        "quality": {
            "quality_status": "usable_with_caution",
            "retrieval_weight": 0.7,
            "review_warning": True,
            "allowed_uses": ["broad_context"],
        },
        "page_quality": {
            "Page A": {
                "quality_status": "low_priority",
                "retrieval_weight": 0.3,
            }
        },
    }
    value = source_quality_metadata(source, title="Page A")
    assert value["quality_status"] == "low_priority"
    assert value["retrieval_weight"] == 0.3
    assert value["review_warning"] is True
