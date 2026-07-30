"""MechET: mechanism electron-transfer CoT for retrosynthesis."""

from .collator import AssistantOnlyCollator, encode_assistant_only, find_assistant_start
from .mech_et import format_mech_et_cot, verify_mech_et
from .mech_graph import format_mech_graph_cot, load_flower_graphs, verify_mech_graph
from .sft import convert_record_to_qwen_sft, format_mech_et_assistant, parse_mech_et_output
from .rlvr import compute_advantages, compute_mechvr_reward, mechvr_gate
from .verifier import compute_mech_et_reward, compute_reward

__all__ = [
    "AssistantOnlyCollator",
    "compute_advantages",
    "compute_mech_et_reward",
    "compute_mechvr_reward",
    "compute_reward",
    "convert_record_to_qwen_sft",
    "encode_assistant_only",
    "find_assistant_start",
    "format_mech_et_assistant",
    "format_mech_et_cot",
    "format_mech_graph_cot",
    "load_flower_graphs",
    "mechvr_gate",
    "parse_mech_et_output",
    "verify_mech_et",
    "verify_mech_graph",
]

__version__ = "0.3.0"
