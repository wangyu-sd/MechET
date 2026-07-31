"""Build matched ICLR task variants from one frozen, decontaminated corpus."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from mechet.data_audit import split_structural_and_environment

_PROOF_BLOCK = re.compile(r"<proof>\s*(.*?)\s*</proof>", re.I | re.S)
_MECHANISM_BLOCK = re.compile(r"<mechanism>\s*(.*?)\s*</mechanism>", re.I | re.S)


def _product(row: Mapping[str, Any]) -> str:
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and content.startswith("TARGET:"):
            return content.split("\n", 1)[0].replace("TARGET:", "", 1).strip()
    return ""


def _assistant(row: Mapping[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _gold_precursor(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("derived_precursor") or metadata.get("initial_reactants") or "")


def core_precursor(row: Mapping[str, Any]) -> str:
    roles = split_structural_and_environment(_gold_precursor(row), _product(row))
    return roles.structural_smiles


def _replace_assistant(row: Mapping[str, Any], content: str, task_type: str) -> dict[str, Any]:
    output = deepcopy(dict(row))
    messages = [dict(message) for message in output.get("messages") or [] if message.get("role") != "assistant"]
    messages.append({"role": "assistant", "content": content})
    output["messages"] = messages
    output["task_type"] = task_type
    metadata = dict(output.get("metadata") or {})
    metadata["task_type"] = task_type
    metadata["core_precursor"] = core_precursor(row)
    output["metadata"] = metadata
    return output


def build_outcome_only_row(row: Mapping[str, Any]) -> dict[str, Any]:
    answer = core_precursor(row)
    return _replace_assistant(row, f"<answer>\n{answer}\n</answer>", "outcome_only_retro")


def _net_actions(proof: str) -> list[str]:
    body_match = _PROOF_BLOCK.search(proof)
    body = body_match.group(1) if body_match else proof
    bond_totals: dict[tuple[int, int], int] = {}
    charge_last: dict[int, tuple[int, int]] = {}
    imports: set[str] = set()
    for raw in body.splitlines():
        line = raw.strip()
        parts = line.split()
        if line.startswith("IMPORT "):
            imports.add(line)
        elif len(parts) == 4 and parts[0] == "BOND":
            key = tuple(sorted((int(parts[1]), int(parts[2]))))
            bond_totals[key] = bond_totals.get(key, 0) + int(parts[3])
        elif len(parts) == 4 and parts[0] == "CHARGE":
            atom = int(parts[1])
            if atom in charge_last:
                charge_last[atom] = (charge_last[atom][0], int(parts[3]))
            else:
                charge_last[atom] = (int(parts[2]), int(parts[3]))
    lines = ["NET_EDIT v1", *sorted(imports)]
    for (i, j), delta in sorted(bond_totals.items()):
        if delta:
            lines.append(f"BOND {i} {j} {delta:+d}")
    for atom, (q0, q1) in sorted(charge_last.items()):
        if q0 != q1:
            lines.append(f"CHARGE {atom} {q0} {q1}")
    return lines


def build_net_edit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    actions = "\n".join(_net_actions(_assistant(row)))
    answer = core_precursor(row)
    content = f"<edit>\n{actions}\n</edit>\n<answer>\n{answer}\n</answer>"
    return _replace_assistant(row, content, "net_edit_retro")


def build_state_cot_row(row: Mapping[str, Any]) -> dict[str, Any]:
    assistant = _assistant(row)
    mechanism = _MECHANISM_BLOCK.search(assistant)
    if not mechanism:
        raise ValueError("state-CoT baseline requires a <mechanism> block")
    answer = core_precursor(row)
    content = f"<mechanism>\n{mechanism.group(1).strip()}\n</mechanism>\n<answer>\n{answer}\n</answer>"
    return _replace_assistant(row, content, "state_cot_core_retro")


def build_proof_row(row: Mapping[str, Any]) -> dict[str, Any]:
    proof = _PROOF_BLOCK.search(_assistant(row))
    if not proof:
        raise ValueError("proof baseline requires a <proof> block")
    return _replace_assistant(row, f"<proof>\n{proof.group(1).strip()}\n</proof>", "mech_proof_core_retro")
