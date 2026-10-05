#!/usr/bin/env python3
"""Check whether reference first electron events touch the input product.

This is a diagnostic on the strict executable trace view, not an assertion
about every possible route in the complete reaction-level benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.eval_system_one_product_start_pilot import principal_component_atom_indices
from scripts.train_system_one_electron_flow import verify_source

CURRENT_STATE = re.compile(r"^CURRENT STATE SMILES: (.+)$", re.MULTILINE)
ALIASES = re.compile(r"A(\d+)")
EXPECTED_FULL = {"train": 24959, "valid": 3120, "test": 3120}


def reference_touched_atoms(arguments: dict) -> set[int]:
    """Extract temporary public atom handles from one recorded event."""
    return {int(value) - 1 for value in ALIASES.findall(json.dumps(arguments))}


def audit(strict_dir: Path, full_dir: Path, split: str) -> dict:
    strict_path = strict_dir / f"{split}.jsonl"
    strict_source = verify_source(strict_path)
    full_path = full_dir / f"{split}.jsonl"
    manifest = json.loads((full_dir / "manifest.json").read_text())
    declared = manifest["splits"][split]
    if (manifest["product_source_field"] != "rxn_prod_equ"
            or manifest["executor_filtering"] is not False
            or declared["rows"] != EXPECTED_FULL[split]
            or declared["endpoint_sha256"] != sha256(full_path)):
        raise ValueError("full equ-proxy source contract mismatch")
    products = {}
    for line in full_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in products:
            raise ValueError("duplicate full endpoint reaction ID")
        products[reaction_id] = str(row["product_unmapped"])
    if len(products) != EXPECTED_FULL[split]:
        raise ValueError("full endpoint denominator mismatch")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for line in strict_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["metadata"]["reaction_id"])
        if row["metadata"]["decision_type"] != "event" or reaction_id in seen:
            continue
        seen.add(reaction_id)
        users = [message["content"] for message in row["messages"]
                 if message.get("role") == "user"]
        assistants = [message for message in row["messages"]
                      if message.get("role") == "assistant"]
        if len(users) != 1 or len(assistants) != 1 or reaction_id not in products:
            raise ValueError(f"{reaction_id}: invalid first-event source")
        state_match = CURRENT_STATE.search(users[0])
        if state_match is None:
            raise ValueError(f"{reaction_id}: missing visible current state")
        state = state_match.group(1)
        principal_atoms = principal_component_atom_indices(state, products[reaction_id])
        calls = assistants[0].get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "apply_electron_flow":
            raise ValueError(f"{reaction_id}: first event/tool mismatch")
        touched = reference_touched_atoms(calls[0]["function"]["arguments"])
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(state, params)
        if not touched or mol is None or max(touched) >= mol.GetNumAtoms():
            raise ValueError(f"{reaction_id}: reference event has no valid public atom location")
        counts["first_events"] += 1
        counts["touches_principal"] += int(bool(touched & principal_atoms))
        counts["context_only"] += int(not bool(touched & principal_atoms))
        counts["principal_only"] += int(touched <= principal_atoms)
    if len(seen) != strict_source["reaction_denominator"]:
        raise ValueError("strict reaction denominator differs from first-event coverage")
    return {
        "artifact_type": "system_one_strict_reference_first_event_product_locality_audit",
        "split": split,
        "interpretation": "Strict-trace reference first events only; not full-benchmark oracle actions",
        "strict_source_sha256": strict_source["sha256"],
        "full_endpoint_source_sha256": declared["endpoint_sha256"],
        "product_source_field": "rxn_prod_equ",
        "reaction_denominator": strict_source["reaction_denominator"],
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED_FULL), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.strict_dir, args.full_endpoint_dir, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
