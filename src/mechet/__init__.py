"""MechET: verifiable and proof-carrying retrosynthesis."""

from .collator import (
    AssistantOnlyCollator,
    encode_assistant_only,
    find_assistant_start,
)
from .mech_et import format_mech_et_cot, verify_mech_et
from .mech_graph import (
    format_mech_graph_cot,
    load_flower_graphs,
    verify_mech_graph,
)
from .proof_diagnostics import (
    FailureCertificate,
    ProofRepairResult,
    diagnose_proof,
    format_repair_feedback,
    repair_proof_once,
)
from .proof_equivalence import (
    ProofEquivalenceSignature,
    canonical_partial_order_signature,
    composition_signature,
    edge_primitive_signature,
    primitive_signatures,
    proofs_equivalent,
)
from .proof_program import (
    ProofEdge,
    ProofExecutionResult,
    ProofProgram,
    compile_mech_et_body,
    execute_proof,
    format_proof_output,
    parse_proof_program,
    verify_proof,
)
from .proof_sft import convert_mech_et_row_to_proof_sft
from .proof_splits import (
    ProofSplitFeatures,
    build_compositional_ood_split,
    extract_split_features,
)
from .rlvr import (
    compute_advantages,
    compute_mechvr_reward,
    compute_proofvr_reward,
    compute_rollout_reward,
    mechvr_gate,
)
from .sft import (
    convert_record_to_qwen_sft,
    format_mech_et_assistant,
    parse_mech_et_output,
)
from .verifier import compute_mech_et_reward, compute_reward

__all__ = [
    "AssistantOnlyCollator",
    "FailureCertificate",
    "ProofEdge",
    "ProofEquivalenceSignature",
    "ProofExecutionResult",
    "ProofProgram",
    "ProofRepairResult",
    "ProofSplitFeatures",
    "build_compositional_ood_split",
    "canonical_partial_order_signature",
    "compile_mech_et_body",
    "composition_signature",
    "compute_advantages",
    "compute_mech_et_reward",
    "compute_mechvr_reward",
    "compute_proofvr_reward",
    "compute_reward",
    "compute_rollout_reward",
    "convert_mech_et_row_to_proof_sft",
    "convert_record_to_qwen_sft",
    "diagnose_proof",
    "edge_primitive_signature",
    "encode_assistant_only",
    "execute_proof",
    "extract_split_features",
    "find_assistant_start",
    "format_mech_et_assistant",
    "format_mech_et_cot",
    "format_mech_graph_cot",
    "format_proof_output",
    "format_repair_feedback",
    "load_flower_graphs",
    "mechvr_gate",
    "parse_mech_et_output",
    "parse_proof_program",
    "primitive_signatures",
    "proofs_equivalent",
    "repair_proof_once",
    "verify_mech_et",
    "verify_mech_graph",
    "verify_proof",
]

__version__ = "0.5.0"
