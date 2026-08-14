from __future__ import annotations

import importlib.util
from pathlib import Path

from bs4 import BeautifulSoup


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_openstax_book.py"
    spec = importlib.util.spec_from_file_location("download_openstax_book", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_question_keeps_problem_id_and_marks_figure() -> None:
    module = _module()
    main = BeautifulSoup(
        '<main><div data-type="exercise-question" id="q1">'
        '<div class="os-prefix">Problem <span class="os-number">13-24</span></div>'
        '<p>Propose a structure.</p><img alt="skeletal formula"/></div></main>',
        "html.parser",
    ).main
    items = module.extract_assessment_items(main, role="evaluation_questions")
    assert items[0]["problem_id"] == "13-24"
    assert items[0]["has_figure"] is True
    assert "Propose a structure" in items[0]["text"]
    assert items[0]["image_alts"] == ["skeletal formula"]


def test_corpus_page_exposes_question_for_eval_but_excludes_it_from_prose() -> None:
    module = _module()
    main = BeautifulSoup(
        '<main><h2>Substitution</h2><p>SN2 reactions are concerted.</p>'
        '<div data-type="exercise-question"><div class="os-prefix">Problem '
        '<span class="os-number">11-4</span></div><p>Predict the product.</p></div></main>',
        "html.parser",
    ).main
    items = module.extract_assessment_items(main, role="corpus")
    sections = module.extract_sections(main, page_title="Substitution", section_kind="numbered")
    assert items[0]["problem_id"] == "11-4"
    assert "SN2 reactions are concerted" in sections[0]["text"]
    assert "Predict the product" not in sections[0]["text"]


def test_answer_extraction() -> None:
    module = _module()
    main = BeautifulSoup(
        '<main><div data-type="question-solution"><a class="os-prefix" '
        'href="/problems#q1">Problem <span class="os-number">3-7</span></a>'
        '<div class="os-solution-container">SN2 substitution</div></div></main>',
        "html.parser",
    ).main
    items = module.extract_assessment_items(main, role="evaluation_answer_key")
    assert items[0]["problem_id"] == "3-7"
    assert items[0]["text"] == "SN2 substitution"
    assert items[0]["source_anchor"] == "/problems#q1"
