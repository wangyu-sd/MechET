"""Regression tests for the public documentation hierarchy."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_readme_describes_current_proof_hypothesis_method():
    text = read("README.md")
    assert "K proof hypotheses" in text
    assert "local executable primitives" in text
    assert "Generate–Falsify–Repair" in text
    assert "ExecutePass@K" in text
    assert "docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md" in text
    assert "docs/README.md" in text
    assert "docs/BENCHMARK_RESULTS.md" not in text
    assert "docs/EVAL.md" not in text


def test_authoritative_experiment_plan_has_complete_contract():
    text = read("docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md")
    required_sections = [
        "Pipeline A — source data, audit, and proof curriculum",
        "Pipeline B — matched baselines and proof models",
        "Pipeline C — inference modes",
        "Pipeline D — validation experiments",
        "Required paper result package",
        "Result interpretation and stopping rules",
        "Reproducibility and artifact contract",
    ]
    for heading in required_sections:
        assert heading in text
    required_terms = [
        "L_SFT",
        "L_DPO",
        "L_RLVR",
        "L_repair",
        "ExecutePass@K",
        "EndpointPass@K",
        "false acceptance rate",
        "fully verified route rate",
    ]
    for term in required_terms:
        assert term in text


def test_documentation_map_declares_authority_and_archives():
    text = read("docs/README.md")
    for path in [
        "PROOF_CARRYING.md",
        "PROOF_CENTRIC_EXPERIMENT_PLAN.md",
        "PROOF_EQUIVALENCE.md",
        "DATA_LEAKAGE_AND_ICLR_PLAN.md",
    ]:
        assert path in text
    for path in [
        "EXPERIMENT_PLAN_ICLR_TO_NMI.md",
        "EVAL.md",
        "BENCHMARK_RESULTS.md",
        "README_DESIGN_NOTES.md",
    ]:
        assert path in text


def test_deprecated_documents_are_explicitly_marked():
    deprecated = {
        "docs/EXPERIMENT_PLAN_ICLR_TO_NMI.md": "deprecated",
        "docs/EVAL.md": "deprecated",
        "docs/BENCHMARK_RESULTS.md": "not a result table",
        "docs/README_DESIGN_NOTES.md": "Archived",
    }
    for path, marker in deprecated.items():
        assert marker.lower() in read(path).lower(), path


def test_method_document_states_representation_boundary():
    text = read("docs/PROOF_CARRYING.md")
    assert "local operations rather than a library of complete reaction templates" in text
    assert "same autoregressive actor is sampled repeatedly" in text
    assert "does not yet uniquely pair every electron source" in text
    assert "deterministic and is not trained" in text
