from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_status_matrix_separates_infrastructure_from_scientific_results():
    text = read("docs/STATUS_MATRIX.md")
    for term in (
        "Tool-SFT checkpoints",
        "Pilot pending",
        "H1 causal faithfulness result",
        "Not established",
        "H2 compositional generalization result",
        "H3 evidence benefit result",
        "A green infrastructure row does not imply a positive scientific result",
    ):
        assert term in text


def test_documentation_map_exposes_integrity_tools():
    text = read("docs/README.md")
    for term in (
        "STATUS_MATRIX.md",
        "paired bootstrap confidence intervals",
        "exact McNemar tests",
        "Holm family-wise error correction",
        "scripts/check_source_health.py",
        "scripts/check_documentation_integrity.py",
        "aggregate_evaluation_seeds.py",
        "near-duplicate overlap",
    ):
        assert term in text


def test_source_registry_declares_quality_boundaries():
    text = read("knowledge/source_registry.yaml")
    for term in (
        "quality_status",
        "retrieval_weight",
        "allowed_uses",
        "disallowed_uses",
        "mechanism_ground_truth",
        "page_quality",
    ):
        assert term in text
