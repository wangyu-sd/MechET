from mechet.textbook_cleaning import quality_flags, strip_wikicode, wikitext_sections


def test_wikicode_cleaner_removes_navigation_templates_and_references() -> None:
    source = """<noinclude>[[Organic Chemistry|<< Previous]]</noinclude>
== Nucleophilic substitution ==
An [[nucleophile]] donates an electron pair to carbon.<ref>citation</ref>
[[File:Mechanism.svg|thumb|mechanism]] {{cleanup}}
=== Solvent effects ===
Polar aprotic solvents often favor the bimolecular pathway.
"""
    sections = wikitext_sections(source, default_heading="Test")
    combined = "\n".join(section.text for section in sections)
    assert "Previous" not in combined
    assert "citation" not in combined
    assert "Mechanism.svg" not in combined
    assert "cleanup" not in combined
    assert "nucleophile" in combined
    assert {section.heading for section in sections} >= {
        "Nucleophilic substitution",
        "Solvent effects",
    }
    assert not quality_flags(combined)


def test_fallback_wiki_link_rendering_is_human_readable() -> None:
    assert strip_wikicode("[[Target|visible label]] and [[Nucleophile]]") == (
        "visible label and Nucleophile"
    )


def test_quality_flags_url_heavy_reference_lists() -> None:
    text = (
        "This passage lists source pages https://example.org/one and "
        "https://example.org/two instead of explaining the chemistry."
    )
    assert "url_heavy" in quality_flags(text)
