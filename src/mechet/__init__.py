"""MechET: verifiable and proof-carrying retrosynthesis."""

from .collator import (
    AssistantOnlyCollator,
    encode_assistant_only,
    find_assistant_start,
)
from .data_audit import (
    KEY_LEVELS,
    NormalizationConfig,
    ReactionKeys,
    ReactionRecord,
    RoleSplit,
    build_key_index,
    canonical_multiset,
    quarantine_reason,
    reaction_keys,
    split_structural_and_environment,
)
from .iclr_rewards import CoreProofRewardConfig, compute_core_proof_reward, core_gold
from .iclr_tasks import (
    build_net_edit_row,
    build_outcome_only_row,
    build_proof_row,
    build_state_cot_row,
    core_precursor,
)
from .map_invariance import (
    record_map_permutation,
    remap_proof_text,
    remap_smiles,
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
    ChargeAction,
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
    "ChargeAction",
    "CoreProofRewardConfig",
    "FailureCertificate",
    "KEY_LEVELS",
    "NormalizationConfig",
    "ProofEdge",
    "ProofEquivalenceSignature",
    "ProofExecutionResult",
    "ProofProgram",
    "ProofRepairResult",
    "ProofSplitFeatures",
    "ReactionKeys",
    "ReactionRecord",
    "RoleSplit",
    "build_compositional_ood_split",
    "build_key_index",
    "build_net_edit_row",
    "build_outcome_only_row",
    "build_proof_row",
    "build_state_cot_row",
    "canonical_multiset",
    "canonical_partial_order_signature",
    "compile_mech_et_body",
    "composition_signature",
    "compute_advantages",
    "compute_core_proof_reward",
    "compute_mech_et_reward",
    "compute_mechvr_reward",
    "compute_proofvr_reward",
    "compute_reward",
    "compute_rollout_reward",
    "convert_mech_et_row_to_proof_sft",
    "convert_record_to_qwen_sft",
    "core_gold",
    "core_precursor",
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
    "quarantine_reason",
    "reaction_keys",
    "record_map_permutation",
    "remap_proof_text",
    "remap_smiles",
    "repair_proof_once",
    "split_structural_and_environment",
    "verify_mech_et",
    "verify_mech_graph",
    "verify_proof",
]

__version__ = "0.6.0"
