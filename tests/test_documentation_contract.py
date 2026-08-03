from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_readme_scientific_contract():
    text = read("README.md")
    required = [
        "Causal and compositional electron-flow reasoning",
        "causal program induction over executable electron-flow actions",
        "environment-owned action trace",
        "finish_trace",
        "Electron-flow execution primitive",
        "Mechanistic knowledge anchor",
        "trace_no_knowledge",
        "trace_length_matched_irrelevant",
        "direct_textbook_rag",
        "scripts/train_inverse_agent_trace.py",
        "scripts/train_inverse_agent_knowledge.py",
        "scripts/validate_experiment_contract.py",
        "docs/SCIENTIFIC_THESIS.md",
        "docs/EXECUTION_PLAN.md",
    ]
    for term in required:
        assert term in text
    assert "Bidirectional electron-flow reasoning for reliable retrosynthesis" not in text
    assert "scripts/train_inverse_agent_trl.py" in text
    assert "Legacy complete-proof/loose-trace baseline" in text
    assert text.index("scripts/train_inverse_agent_trace.py") < text.index(
        "scripts/train_inverse_agent_trl.py"
    )


def test_scientific_thesis_contract():
    text = read("docs/SCIENTIFIC_THESIS.md")
    for term in [
        "Can mechanistic reasoning in retrosynthesis be made causal and compositional",
        "H1 — Causal faithfulness",
        "H2 — Compositional basis",
        "H3 — Separation of formal and empirical evidence",
        "The model does not submit an independent proof or answer in the main method",
        "Electron-flow execution primitive",
        "Mechanistic knowledge anchor",
        "TraceOwnedAgentEnv",
        "finish_trace",
        "Prohibited claims",
    ]:
        assert term in text


def test_trace_contract_is_main_method():
    text = read("docs/TRACE_FAITHFULNESS.md")
    for term in [
        "authoritative for the MechET main inference contract",
        "sole computational source of the proof and precursor",
        "FREE_FORM_PROOF_DISABLED",
        "finish_trace",
        "environment-compiled proof",
        "Causal interventions",
        "remove tool observations",
        "replace observations with stale states",
    ]:
        assert term in text
    assert "submit one complete executable proof" not in text


def test_experiment_plan_contract():
    text = read("docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md")
    for term in [
        "H1 — causal faithfulness",
        "H2 — compositional basis",
        "H3 — formal and empirical evidence separation",
        "submit_proof is disabled",
        "execution-primitive compositions",
        "Mechanistic knowledge anchors",
        "Real training smoke test",
        "Causal interventions",
        "Required paper result package",
        "Global stopping rules",
    ]:
        assert term in text
    for old_term in [
        "Pipeline A — source data, audit, and proof curriculum",
        "Pipeline B — matched baselines and proof models",
        "Pipeline C — inference modes",
        "Pipeline D — validation experiments",
    ]:
        assert old_term not in text


def test_execution_plan_orders_scientific_gates():
    text = read("docs/EXECUTION_PLAN.md")
    ordered = [
        "Phase 0 — freeze the scientific contract",
        "Phase 1 — data feasibility and conversion coverage",
        "Phase 2 — construct matched scientific conditions",
        "Phase 3 — real Tool-SFT smoke tests",
        "Phase 4 — test H1: causal faithfulness",
        "Phase 5 — test H2: compositional generalization",
        "Phase 6 — test H3: empirical evidence separation",
        "Phase 7 — scale and optimization",
        "Phase 8 — test-time hypotheses and planning extensions",
    ]
    positions = [text.index(term) for term in ordered]
    assert positions == sorted(positions)
    for term in [
        "build_knowledge_ablation_suite.py",
        "validate_experiment_contract.py",
        "train_inverse_agent_trace.py",
        "train_inverse_agent_knowledge.py",
        "replace observations with stale states",
        "Planning is an extension",
    ]:
        assert term in text


def test_evidence_documents_use_anchor_terminology():
    anchor_doc = read("docs/MECHANISTIC_PRIMITIVE_LIBRARY.md")
    for term in [
        "Mechanistic knowledge anchors",
        "Critical distinction",
        "Electron-flow execution primitive",
        "field name `primitive_id` remains for API compatibility",
        "No knowledge-anchor IDs are used to define the split",
    ]:
        assert term in anchor_doc

    knowledge = read("docs/KNOWLEDGE_ABLATIONS.md")
    for term in [
        "Matched evidence-layer experiments",
        "direct open-book",
        "derived automatically",
        "direct_answer_from_textbook",
        "trace_textbook_rag > trace_no_knowledge",
        "trace_textbook_rag > trace_length_matched_irrelevant",
    ]:
        assert term in knowledge

    agent = read("docs/KNOWLEDGE_AUGMENTED_AGENT.md")
    for term in [
        "Evidence-augmented trace-owned agent",
        "not a separate endpoint-generation architecture",
        "Free-form `submit_proof` remains disabled",
        "Tool-SFT adapter path and hash",
    ]:
        assert term in agent


def test_tool_sft_contract():
    text = read("docs/TOOL_SFT.md")
    for term in [
        "Replay-verified Tool-SFT",
        "Stable quarantine families",
        "Build all matched conditions",
        "anchors-only and direct open-book rows are derived automatically",
        "Required conversion report",
        "Real training smoke test",
        "Tool-SFT to RL lineage",
    ]:
        assert term in text


def test_documentation_map_and_deprecations():
    index = read("docs/README.md")
    for path in [
        "SCIENTIFIC_THESIS.md",
        "TRACE_FAITHFULNESS.md",
        "PROOF_CENTRIC_EXPERIMENT_PLAN.md",
        "EXECUTION_PLAN.md",
        "TOOL_SFT.md",
        "KNOWLEDGE_ABLATIONS.md",
        "MECHANISTIC_PRIMITIVE_LIBRARY.md",
        "PROOF_CARRYING.md",
        "FORWARD_ELECTRON_EXPERT.md",
        "FRAMEWORK_MIGRATION.md",
        "PROOF_EQUIVALENCE.md",
        "DATA_LEAKAGE_AND_ICLR_PLAN.md",
        "../knowledge/README.md",
    ]:
        assert path in index
    for term in [
        "single source of truth",
        "Execution primitive",
        "Mechanistic knowledge anchor",
        "submit_proof is disabled",
        "Planning is a downstream extension",
    ]:
        assert term.lower() in index.lower()

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
