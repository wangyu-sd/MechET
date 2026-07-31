from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_readme_contract():
    text = read("README.md")
    for term in [
        "K proof hypotheses",
        "local executable primitives",
        "Generate–Falsify–Repair",
        "ExecutePass@K",
        "docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md",
        "docs/README.md",
    ]:
        assert term in text
    assert "docs/BENCHMARK_RESULTS.md" not in text


def test_experiment_plan_contract():
    text = read("docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md")
    for term in [
        "Pipeline A — source data, audit, and proof curriculum",
        "Pipeline B — matched baselines and proof models",
        "Pipeline C — inference modes",
        "Pipeline D — validation experiments",
        "Required paper result package",
        "Result interpretation and stopping rules",
        "Reproducibility and artifact contract",
        "L_SFT",
        "L_DPO",
        "L_RLVR",
        "L_repair",
        "ExecutePass@K",
        "EndpointPass@K",
        "false acceptance rate",
        "fully verified route rate",
    ]:
        assert term in text


def test_documentation_map_and_deprecations():
    index = read("docs/README.md")
    for path in [
        "PROOF_CARRYING.md",
        "PROOF_CENTRIC_EXPERIMENT_PLAN.md",
        "PROOF_EQUIVALENCE.md",
        "DATA_LEAKAGE_AND_ICLR_PLAN.md",
        "EXPERIMENT_PLAN_ICLR_TO_NMI.md",
        "EVAL.md",
        "BENCHMARK_RESULTS.md",
        "README_DESIGN_NOTES.md",
    ]:
        assert path in index
    markers = {
        "docs/EXPERIMENT_PLAN_ICLR_TO_NMI.md": "deprecated",
        "docs/EVAL.md": "deprecated",
        "docs/BENCHMARK_RESULTS.md": "not a result table",
        "docs/README_DESIGN_NOTES.md": "archived",
    }
    for path, marker in markers.items():
        assert marker in read(path).lower()


def test_method_boundary_contract():
    text = read("docs/PROOF_CARRYING.md")
    for term in [
        "local operations rather than a library of complete reaction templates",
        "same autoregressive actor is sampled repeatedly",
        "uniquely pair every electron source",
        "electron sink",
        "deterministic and is not trained",
    ]:
        assert term in text
