"""Auditable execution feedback and examples for continual electron-program learning."""
from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

from rdkit import Chem
from .forward_expert import verify_electron_step
from .python_program import PythonElectronProgram, execute_python_program
from .python_template_slots import parse_template_slots, format_template_slots
from .python_repair_rollout import execution_feedback
from .python_template_rlvr import score_template_rlvr_candidate
from .endpoints import split_precursor_endpoints


def mol(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    value = Chem.MolFromSmiles(smiles, params)
    if value is None:
        raise ValueError("INVALID_SMILES")
    return value


def atom_maps(smiles):
    return [a.GetAtomMapNum() for a in mol(smiles).GetAtoms()]


def referenced(move):
    if "source" in move:
        return set(move["source"]["atoms"]) | set(move["sink"]["atoms"])
    return {a for b in move.get("bond_deltas", []) for a in b["atoms"]} | {
        b["atom_map"] for b in move.get("charge_actions", [])}


def normalize(smiles):
    value = mol(smiles)
    for atom in value.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(value), canonical=True, isomericSmiles=True)


def restore_declared_imports(target, text, previous):
    """Recover only unambiguous fragments already proposed by this same actor.

    Never invent atoms, use gold fragments, or guess a map renaming. Original
    sampled tokens stay unchanged; the tool's transformed program is audited.
    """
    changes = []
    try:
        program = parse_template_slots(text, target_smiles=target)
        old = parse_template_slots(previous, target_smiles=target)
        candidates = {s: set(atom_maps(s)) for step in old.steps for s in step.imports}
        available = set(atom_maps(target))
        steps = []
        for i, step in enumerate(program.steps):
            additions = list(step.imports)
            for s in additions:
                available.update(atom_maps(s))
            missing = set().union(*(referenced(m) for m in step.moves)) - available
            for a in sorted(missing):
                if a in available:
                    continue
                matches = [s for s, maps in candidates.items() if a in maps and not maps & available
                           and 0 not in maps and len(maps) == len(atom_maps(s))]
                if len(matches) == 1:
                    s = matches[0]
                    additions.append(s)
                    available.update(candidates[s])
                    changes.append({"step": i, "restored_import": s, "source": "previous_actor_draft"})
            steps.append(replace(step, imports=tuple(additions)))
        return format_template_slots(PythonElectronProgram(target, tuple(steps))), changes
    except (ValueError, KeyError, TypeError):
        return text, []


def detailed_feedback(target, text, *, terminated=True):
    """Public, label-free diagnostics with exact references and local actual state."""
    basic = execution_feedback(target, text, terminated=terminated)
    if not terminated:
        return basic
    try:
        program = parse_template_slots(text, target_smiles=target)
        state = target
        for i, step in enumerate(program.steps):
            augmented = ".".join((state, *step.imports))
            value = mol(augmented)
            maps = atom_maps(augmented)
            available = set(maps)
            moves = []
            for j, move in enumerate(step.moves):
                absent = sorted(referenced(move) - available)
                if absent:
                    moves.append({"action": j, "missing_maps": absent, "move": move})
            replay = verify_electron_step(augmented, step.moves)
            if not replay.get("ok"):
                refs = set().union(*(referenced(m) for m in step.moves))
                atoms = [{"map": a.GetAtomMapNum(), "element": a.GetSymbol(),
                          "charge": a.GetFormalCharge(), "hydrogens": a.GetTotalNumHs()}
                         for a in value.GetAtoms() if a.GetAtomMapNum() in refs]
                bonds = [{"atoms": [b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()],
                          "order": b.GetBondTypeAsDouble()} for b in value.GetBonds()
                         if {b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()} & refs]
                return {"ok": False, "diagnostics": [{"code": replay.get("code"),
                    "message": replay.get("message"), "step": i, "missing_references": moves,
                    "duplicate_maps": sorted(a for a in available if maps.count(a) > 1),
                    "available_maps": sorted(available), "local_atoms": atoms, "local_bonds": bonds}]}
            state = replay["state_smiles"]
        executed = execute_python_program(program, target_smiles=target)
        if executed.ok:
            structural = split_precursor_endpoints(executed.precursor_smiles, target).structural
            if normalize(structural) == normalize(target):
                return {"ok": False, "diagnostics": [{"code": "NO_NET_TRANSFORMATION",
                    "message": "The structural endpoint equals TARGET. Undoing and redoing a bond is not a retrosynthesis solution."}]}
    except (ValueError, KeyError, TypeError):
        pass
    return basic


def feedback_message(feedback):
    if feedback["ok"]:
        raise ValueError("No oracle correction of executable candidates")
    # Only data produced by detailed_feedback; caller must never pass a reward record.
    return ("Execution failed. Zero-based step/action indices. Actual-state diagnostics:\n"
            + json.dumps(feedback["diagnostics"], separators=(",", ":"))
            + "\nCorrect the COMPLETE STEPS list from original TARGET. Keep necessary imports. "
            "Reference only atoms present in TARGET or explicitly imported fragments. "
            "Previously proposed fragments may be restored by the runtime only when unambiguous. "
            "Output only STEPS, no prose or final precursor SMILES.")


def evaluate(target, expected, text, terminated, previous=""):
    effective, changes = restore_declared_imports(target, text, previous) if previous else (text, [])
    public = detailed_feedback(target, effective, terminated=terminated)
    score = score_template_rlvr_candidate({"target_smiles": target, "expected_precursor": expected}, effective) if terminated else {
        "formal_execute": False, "structural_precursor_exact": False}
    noop = any(d["code"] == "NO_NET_TRANSFORMATION" for d in public["diagnostics"])
    # No positive task reward for validity alone. Gold appears only in scoring.
    correct = bool(score["formal_execute"] and score["structural_precursor_exact"] and not noop)
    return {"feedback": public, "formal_execute": bool(score["formal_execute"]),
            "correct": correct, "noop": noop, "reward": float(correct),
            "effective_program": effective, "tool_repairs": changes}


def supervised_record(tokenizer, prompt_ids, answer, source_id, kind):
    suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    return {"id": source_id, "kind": kind, "input_ids": prompt_ids + suffix,
            "loss_mask": [0] * len(prompt_ids) + [1] * len(suffix),
            "old_logps": [0.] * (len(prompt_ids) + len(suffix)), "advantage": 0.}


def curriculum_example(row, index):
    """Teacher-prefix auxiliary task. Never used as the product-only evaluation input."""
    program = parse_template_slots(row["messages"][-1]["content"], target_smiles=row["target_smiles"])
    n = index % len(program.steps)
    state = row["target_smiles"]
    for step in program.steps[:n]:
        result = verify_electron_step(".".join((state, *step.imports)), step.moves)
        if not result["ok"]:
            raise ValueError("Gold curriculum prefix failed replay")
        state = result["state_smiles"]
    text = format_template_slots(PythonElectronProgram(state, program.steps[n:]))
    return state, text
