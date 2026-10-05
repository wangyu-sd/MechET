#!/usr/bin/env python3
"""Audit teacher-forced runtime observations without reading future actions.

Each next prompt is rebuilt from the product, the previous accepted executor
result, and the compact history accumulated so far. This is a prerequisite for
closed-loop System-One evaluation; it is not an autonomous rollout result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.trajectory_history import TrajectoryHistory
from mechet.structural_overlap import canonical_unmapped_smiles
from mechet.system_one_replay import _alias_graph_key
from scripts.build_natural_language_event_sft import _prompt
from scripts.build_natural_language_history_sft import PROMPT_SUFFIX
from scripts.train_system_one_electron_flow import file_sha256, verify_source


def mapped_from_visible(smiles: str) -> str:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError("accepted executor state is not a valid SMILES")
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    return Chem.MolToSmiles(molecule, canonical=False)


def runtime_prompt(target: str, current: str, history: TrajectoryHistory) -> str:
    prompt = _prompt(target, mapped_from_visible(current), include_inventory=True)
    if not prompt.endswith(PROMPT_SUFFIX):
        raise ValueError("base prompt contract changed")
    return prompt[: -len(PROMPT_SUFFIX)] + "\n\n" + history.render() + PROMPT_SUFFIX


def chemically_equivalent_prompt(predicted: str, expected: str) -> bool:
    """Allow only stereo-text normalization with unchanged atom addresses.

    RDKit can print opposite ``@``/``@@`` tags for the same stereochemical
    graph after private atom maps are reassigned. The rest of the observation
    must remain byte-identical, including the compact history.
    """
    fields = ("CURRENT STATE SMILES: ", "ANNOTATED CURRENT STATE: ")
    dynamic: list[tuple[str, str]] = []
    stripped: list[str] = []
    for prompt in (predicted, expected):
        lines = prompt.splitlines(keepends=True)
        values = []
        for field in fields:
            matches = [i for i, line in enumerate(lines) if line.startswith(field)]
            if len(matches) != 1:
                return False
            index = matches[0]
            values.append(lines[index][len(field):].rstrip("\r\n"))
            lines[index] = field + "<STATE>\n"
        dynamic.append(tuple(values))
        stripped.append("".join(lines))
    if stripped[0] != stripped[1]:
        return False
    for left, right in zip(*dynamic, strict=True):
        visible_left = re.sub(r"<A\d+>", "", left)
        visible_right = re.sub(r"<A\d+>", "", right)
        if _alias_graph_key(visible_left) != _alias_graph_key(visible_right):
            return False
        if canonical_unmapped_smiles(visible_left) != canonical_unmapped_smiles(visible_right):
            return False
        if re.findall(r"<A\d+>", left) != re.findall(r"<A\d+>", right):
            return False
    return True


def audit_rows(rows: Iterable[dict[str, Any]], *, limit_reactions: int = 0) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    completed: set[str] = set()
    reaction_id = ""
    target = ""
    current = ""
    history = TrajectoryHistory()
    expected_index = 0
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id:
            raise ValueError("source reaction ID is missing")
        if source_id != reaction_id:
            if source_id in completed:
                raise ValueError(f"noncontiguous reaction ID: {source_id}")
            if limit_reactions and counts["reactions"] >= limit_reactions:
                break
            if reaction_id:
                completed.add(reaction_id)
            reaction_id = source_id
            target = str(row["target_smiles"])
            current = target
            history = TrajectoryHistory()
            expected_index = 0
            counts["reactions"] += 1

        metadata = row.get("metadata") or {}
        if int(metadata.get("decision_index", -1)) != expected_index:
            raise ValueError(f"{row.get('id')}: decision index discontinuity")
        if str(row["target_smiles"]) != target:
            raise ValueError(f"{row.get('id')}: target changed within reaction")
        messages = list(row.get("messages") or [])
        users = [message for message in messages if message.get("role") == "user"]
        assistants = [message for message in messages if message.get("role") == "assistant"]
        tools = [message for message in messages if message.get("role") == "tool"]
        if len(users) != 1 or len(assistants) != 1 or len(tools) != 1:
            raise ValueError(f"{row.get('id')}: expected one user/action/result exchange")
        predicted_prompt = runtime_prompt(target, current, history)
        expected_prompt = users[0].get("content")
        if predicted_prompt == expected_prompt:
            counts["prompt_byte_exact"] += 1
        elif isinstance(expected_prompt, str) and chemically_equivalent_prompt(
            predicted_prompt, expected_prompt
        ):
            counts["prompt_stereo_text_equivalent"] += 1
        else:
            raise ValueError(f"{row.get('id')}: runtime observation differs from SFT prompt")

        calls = list(assistants[0].get("tool_calls") or [])
        if len(calls) != 1:
            raise ValueError(f"{row.get('id')}: expected one accepted tool call")
        function = calls[0].get("function") or {}
        name = str(function.get("name") or "")
        if tools[0].get("name") != name:
            raise ValueError(f"{row.get('id')}: action/result tool mismatch")
        arguments = function.get("arguments") or {}
        result = json.loads(str(tools[0].get("content") or "{}"))
        history = history.accept(name, arguments, result)
        if name != "finish_trace":
            next_state = result.get("current_state")
            if not isinstance(next_state, str) or not next_state:
                raise ValueError(f"{row.get('id')}: accepted action lacks next state")
            current = next_state
        counts["decisions"] += 1
        counts[name] += 1
        expected_index += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit-reactions", type=int, default=0)
    args = parser.parse_args()
    if args.limit_reactions < 0:
        raise ValueError("--limit-reactions must be nonnegative")
    source = verify_source(args.data)
    with args.data.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        counts = audit_rows(rows, limit_reactions=args.limit_reactions)
    if not args.limit_reactions:
        if counts.get("reactions") != source["reaction_denominator"]:
            raise ValueError("reaction denominator mismatch")
        if counts.get("decisions") != source["decision_rows"]:
            raise ValueError("decision denominator mismatch")
    print(json.dumps({
        "artifact_type": "system_one_runtime_observation_parity",
        "scope": "teacher_forced_previous_accepted_results_not_product_start_rollout",
        "data": str(args.data.resolve()),
        "sha256": file_sha256(args.data),
        "limited": bool(args.limit_reactions),
        **counts,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
