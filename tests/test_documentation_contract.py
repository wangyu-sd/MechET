from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_readme_defines_the_causal_runtime():
    text = read("README.md")
    required = [
        "Causal and compositional electron-flow reasoning",
        "environment-owned trace",
        "replay declared moves",
        "full_precursor_state",
        "structural_precursor",
        "auxiliary_fragments",
        "scripts/infer_mechet.py",
        "scripts/evaluate_faithfulness.py",
        "scripts/evaluate_knowledge_ablation.py",
        "source_to_sink_execution_moves_v1",
        "artifact_type=prediction",
        "missing predictions remain in the denominator",
        "tool_sft_trace_no_knowledge.yaml",
        "tool_sft_direct_textbook.yaml",
    ]
    for term in required:
        assert term.lower() in text.lower()
    assert "train_inverse_agent_trace.py" in text
    assert "train_inverse_agent_trl.py" in text
    assert "label_oracle" in text
    assert "upper bound" in text.lower()


def test_readme_does_not_restore_the_old_main_path():
    text = read("README.md")
    tool_block = re.search(
        r"main\s+TRL-facing\s+environment\s+exposes\s+only.*?```text\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert tool_block is not None
    declared_tools = tool_block.group(1).lower()
    for internal in ("state_dict", "_snapshot", "submit_proof"):
        assert internal not in declared_tools
    for required in (
        "inspect_state",
        "import_fragment",
        "apply_electron_move",
        "apply_coupled_electron_moves",
        "finish_trace",
        "abstain",
    ):
        assert required in declared_tools

    lower = text.lower()
    main_section = lower.split("## main method", 1)[1].split("## terminology", 1)[0]
    assert "state_dict" in main_section and "private" in main_section
    assert "does not expose `submit_proof`" in main_section
    assert "forward expert and multistep planning" not in main_section


def test_scientific_and_execution_documents_are_aligned():
    scientific = read("docs/SCIENTIFIC_THESIS.md")
    experiment = read("docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md")
    execution = read("docs/EXECUTION_PLAN.md")
    for text in (scientific, experiment, execution):
        for term in ["H1", "H2", "H3"]:
            assert term in text
    for term in [
        "scripts/infer_mechet.py",
        "scripts/evaluate_faithfulness.py",
        "scripts/evaluate_knowledge_ablation.py",
        "scripts/validate_experiment_contract.py",
        "scripts/build_mechcomp_ood.py",
        "source_to_sink_execution_moves_v1",
        "label_oracle",
        "missing predictions",
    ]:
        assert term.lower() in execution.lower()
    assert "Prediction artifact contract" in experiment


def test_trace_document_contract():
    text = read("docs/TRACE_FAITHFULNESS.md")
    for term in [
        "TraceOwnedTRLEnvironment",
        "state_dict",
        "private",
        "root-level proof imports",
        "move_sequence_digest",
        "declared_moves_replayed",
        "remove_tool_observations",
        "stale_tool_observations",
        "shuffle_tool_observations",
        "same model, adapter",
        "evaluate_faithfulness.py",
    ]:
        assert term.lower() in text.lower()


def test_tool_sft_document_contract():
    text = read("docs/TOOL_SFT.md")
    for term in [
        "--query-mode state",
        "label_oracle",
        "messages",
        "tools",
        "JSON-object arguments",
        "assistant masks",
        "zero truncation",
        "adapter_manifest.json",
        "adapter SHA-256",
        "tool_sft_trace_no_knowledge.yaml",
        "tool_sft_direct_textbook.yaml",
    ]:
        assert term.lower() in text.lower()


def test_h2_document_uses_execution_moves_not_anchors():
    text = read("docs/PROOF_EQUIVALENCE.md")
    for term in [
        "source_to_sink_execution_moves_v1",
        "LP -> BOND",
        "BOND -> ATOM",
        "BOND -> BOND",
        "knowledge-anchor IDs",
        "non-empty held-out test",
        "zero train/test",
    ]:
        assert term.lower() in text.lower()


def test_h3_document_uses_frozen_predictions():
    text = read("docs/KNOWLEDGE_ABLATIONS.md")
    for term in [
        "frozen textbook evidence",
        "same bounded textbook evidence",
        "artifact_type=prediction",
        "missing predictions",
        "same base-model family and revision",
        "passage_shuffle",
        "same_topic_wrong",
        "remove_warnings",
        "remove_competing_pathways",
        "evaluate_knowledge_ablation.py",
    ]:
        assert term.lower() in text.lower()


def test_documentation_map_has_single_authority_order():
    index = read("docs/README.md")
    for path in [
        "SCIENTIFIC_THESIS.md",
        "TRACE_FAITHFULNESS.md",
        "PROOF_CENTRIC_EXPERIMENT_PLAN.md",
        "EXECUTION_PLAN.md",
        "TOOL_SFT.md",
        "PROOF_EQUIVALENCE.md",
        "KNOWLEDGE_ABLATIONS.md",
        "TEXTBOOK_RAG.md",
        "MECHANISTIC_PRIMITIVE_LIBRARY.md",
        "PROOF_CARRYING.md",
        "FORWARD_ELECTRON_EXPERT.md",
        "FRAMEWORK_MIGRATION.md",
    ]:
        assert path in index
    for term in [
        "explicit TRL facade",
        "Root imports",
        "Prediction artifacts",
        "source-to-sink execution primitives",
        "Planning is a downstream extension",
    ]:
        assert term.lower() in index.lower()


def test_archived_documents_remain_marked():
    markers = {
        "docs/EXPERIMENT_PLAN_ICLR_TO_NMI.md": "deprecated",
        "docs/EVAL.md": "deprecated",
        "docs/BENCHMARK_RESULTS.md": "not a result table",
        "docs/README_DESIGN_NOTES.md": "archived",
    }
    for path, marker in markers.items():
        assert marker in read(path).lower()
