from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def assert_terms(text: str, terms: list[str]) -> None:
    lower = text.lower()
    for term in terms:
        assert term.lower() in lower, term


def test_readme_presents_one_scientific_story():
    text = read("README.md")
    assert_terms(
        text,
        [
            "Causal and compositional electron-flow reasoning",
            "independent, unverifiable channel",
            "environment-owned trace",
            "H1 — causal faithfulness",
            "H2 — compositional basis",
            "H3 — evidence separation",
            "claim",
            "falsifying outcome",
            "replay declared moves",
            "full_precursor_state",
            "structural_precursor",
            "auxiliary_fragments",
            "source_to_sink_execution_moves_v1",
            "artifact_type=prediction",
            "missing predictions remain in the denominator",
            "scripts/infer_mechet.py",
            "scripts/evaluate_faithfulness.py",
            "scripts/evaluate_knowledge_ablation.py",
            "tool_sft_trace_no_knowledge.yaml",
            "tool_sft_direct_textbook.yaml",
            "train_inverse_agent_trace.py",
            "train_inverse_agent_trl.py",
            "label_oracle",
            "upper bound",
        ],
    )


def test_readme_declares_only_the_main_tool_surface():
    text = read("README.md")
    tool_block = re.search(
        r"main\s+trl-facing\s+environment\s+exposes\s+only\s*:\s*```text\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert tool_block is not None
    declared = tool_block.group(1).lower()
    for required in (
        "inspect_state",
        "import_fragment",
        "apply_electron_move",
        "apply_coupled_electron_moves",
        "finish_trace",
        "abstain",
    ):
        assert required in declared
    for forbidden in ("state_dict", "_snapshot", "submit_proof"):
        assert forbidden not in declared

    main = text.lower().split("## main method", 1)[1].split("## terminology", 1)[0]
    assert "state_dict" in main and "private" in main
    assert "does not expose `submit_proof`" in main
    assert "never completes an unfinished trace" in text.lower()
    assert "never falls back" in text.lower()


def test_readme_uses_pass_at_k_semantics():
    text = read("README.md")
    assert_terms(
        text,
        [
            "Pass@K",
            "StructuralEndpointPass@1/5/10",
            "MappedEndpointPass@1/5/10",
            "Without a frozen ranking score",
        ],
    )
    assert "structural precursor top-1/5/10" not in text.lower()


def test_scientific_thesis_has_claim_ladder_and_falsifiers():
    text = read("docs/SCIENTIFIC_THESIS.md")
    assert_terms(
        text,
        [
            "Retrosynthesis can be formulated as causal program induction",
            "Computational contract",
            "Required properties",
            "H1 — Causal faithfulness",
            "H2 — Compositional basis",
            "H3 — Separation of formal and empirical evidence",
            "Falsifier",
            "Formal assumptions",
            "Claim ladder",
            "Permitted headline claims",
            "Prohibited claims",
        ],
    )


def test_experiment_contract_maps_claims_to_controls():
    text = read("docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md")
    assert_terms(
        text,
        [
            "Claim matrix",
            "Primary comparison",
            "Mandatory controls",
            "Integrity gate",
            "Falsifier",
            "Prediction artifact contract",
            "source_to_sink_execution_moves_v1",
            "same bounded textbook evidence",
            "StructuralEndpointPass@1/5/10",
            "missing predictions remain in the denominator",
        ],
    )


def test_execution_plan_is_gate_driven():
    text = read("docs/EXECUTION_PLAN.md")
    assert_terms(
        text,
        [
            "Phase map",
            "Required artifacts",
            "Gate",
            "Stop",
            "scripts/infer_mechet.py",
            "scripts/evaluate_faithfulness.py",
            "scripts/evaluate_knowledge_ablation.py",
            "scripts/validate_experiment_contract.py",
            "scripts/build_mechcomp_ood.py",
            "source_to_sink_execution_moves_v1",
            "label_oracle",
            "missing predictions retained as failures",
            "explicit `finish_trace`",
            "StructuralEndpointPass@1/5/10",
        ],
    )


def test_trace_document_defines_runtime_invariants():
    text = read("docs/TRACE_FAITHFULNESS.md")
    assert_terms(
        text,
        [
            "Runtime invariants",
            "Single endpoint path",
            "Explicit completion",
            "Move–state binding",
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
        ],
    )


def test_tool_sft_document_defines_acceptance_and_lineage():
    text = read("docs/TOOL_SFT.md")
    assert_terms(
        text,
        [
            "Acceptance contract",
            "--query-mode state",
            "label_oracle",
            "messages",
            "tools",
            "JSON-object arguments",
            "assistant masks",
            "zero truncation",
            "adapter_manifest.json",
            "non-self-referential SHA-256",
            "base-model revision",
            "seed and data seed",
            "tool_sft_trace_no_knowledge.yaml",
            "tool_sft_direct_textbook.yaml",
        ],
    )


def test_h2_document_separates_composition_from_vocabulary_novelty():
    text = read("docs/PROOF_EQUIVALENCE.md")
    assert_terms(
        text,
        [
            "Identification target",
            "source_to_sink_execution_moves_v1",
            "LP -> BOND",
            "BOND -> ATOM",
            "BOND -> BOND",
            "knowledge-anchor IDs",
            "non-empty held-out test set",
            "zero train/test complete-composition overlap",
            "Structural overlap audit",
            "Proof equivalence is a separate object",
        ],
    )


def test_h3_document_defines_matched_identification():
    text = read("docs/KNOWLEDGE_ABLATIONS.md")
    assert_terms(
        text,
        [
            "Identification strategy",
            "frozen textbook evidence",
            "same bounded textbook evidence",
            "same base-model family and immutable revision",
            "artifact_type=prediction",
            "missing predictions remain in the denominator",
            "passage_shuffle",
            "same_topic_wrong",
            "remove_warnings",
            "remove_competing_pathways",
            "evaluate_knowledge_ablation.py",
            "StructuralEndpointPass@1/5/10",
        ],
    )


def test_documentation_map_has_authority_and_reader_paths():
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
    assert_terms(
        index,
        [
            "Start here",
            "Authority order",
            "Editor or reviewer",
            "Experimental lead",
            "explicit TRL facade",
            "Root imports",
            "Prediction artifacts",
            "source-to-sink execution primitives",
            "Planning as a downstream extension",
            "Pass@K",
        ],
    )


def test_archived_documents_remain_marked():
    markers = {
        "docs/EXPERIMENT_PLAN_ICLR_TO_NMI.md": "deprecated",
        "docs/EVAL.md": "deprecated",
        "docs/README_DESIGN_NOTES.md": "archived",
    }
    for path, marker in markers.items():
        assert marker in read(path).lower()
