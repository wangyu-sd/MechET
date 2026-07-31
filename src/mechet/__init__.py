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
    "ProofEdge",
    "ProofExecutionResult",
    "ProofProgram",
    "compile_mech_et_body",
    "compute_advantages",
    "compute_mech_et_reward",
    "compute_mechvr_reward",
    "compute_proofvr_reward",
    "compute_reward",
    "compute_rollout_reward",
    "convert_mech_et_row_to_proof_sft",
    "convert_record_to_qwen_sft",
    "encode_assistant_only",
    "execute_proof",
    "find_assistant_start",
    "format_mech_et_assistant",
    "format_mech_et_cot",
    "format_mech_graph_cot",
    "format_proof_output",
    "load_flower_graphs",
    "mechvr_gate",
    "parse_mech_et_output",
    "parse_proof_program",
    "verify_mech_et",
    "verify_mech_graph",
    "verify_proof",
]

__version__ = "0.4.0"
