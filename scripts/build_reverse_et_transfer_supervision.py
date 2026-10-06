#!/usr/bin/env python3
"""Build matched FlowER supervision for testing reverse-ET transfer.

Three training conditions share exactly the same reaction IDs and endpoint task:

1. endpoint_only: endpoint example duplicated once to match example count;
2. endpoint_plus_netedit: endpoint example + endpoint-derived net-edit auxiliary;
3. endpoint_plus_reverse_et: endpoint example + trace-derived reverse-ET auxiliary.

Auxiliary targets never contain the precursor answer. At evaluation time all
conditions use the same endpoint prompt and generate only the precursor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem

EXPECTED = {"train": 257_167, "valid": 2_890, "test": 28_967}

ENDPOINT_SYSTEM = (
    "Given only the mapped product SMILES, predict the atom-contributing "
    "structural precursor SMILES. Output exactly one <answer> block."
)
NETEDIT_SYSTEM = (
    "Given only the mapped product SMILES, predict the inverse net-edit "
    "description. Output exactly one <net_edit> block and do not output "
    "precursor SMILES."
)
REVERSE_ET_SYSTEM = (
    "Given only the mapped product SMILES, predict the complete reverse "
    "electron-transfer supervision program in one shot. Output exactly one "
    "<reverse_et> block and do not output precursor SMILES."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def product(row: dict[str, Any]) -> str:
    value = str(row.get("target_smiles") or "").strip()
    if not value:
        raise ValueError(f"{row.get('id')}: missing target_smiles")
    return value


def precursor(row: dict[str, Any]) -> str:
    value = str(row.get("structural_precursor") or "").strip()
    if not value:
        raise ValueError(f"{row.get('id')}: missing structural_precursor")
    return value


def trace_plan(row: dict[str, Any]) -> dict[str, Any]:
    plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
    if not plan.get("steps"):
        raise ValueError(f"{row.get('id')}: missing trace_plan.steps")
    return plan


def base_example(row: dict[str, Any], *, task: str, system: str, assistant: str) -> dict[str, Any]:
    return {
        "id": f"{row['id']}::{task}",
        "source_id": row.get("source_id") or row.get("id"),
        "reaction_id": row.get("id"),
        "artifact_type": "supervision",
        "task_type": task,
        "target_smiles": product(row),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"TARGET: {product(row)}"},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source_dataset": (row.get("metadata") or {}).get("source_dataset"),
            "source_split": (row.get("metadata") or {}).get("source_split"),
            "reaction_id": row.get("id"),
            "auxiliary_contains_endpoint_answer": False if task != "endpoint" else True,
            "mechanistic_trace_used": task == "reverse_et_aux",
            "net_edit_used": task == "net_edit_aux",
        },
    }


def endpoint_example(row: dict[str, Any], *, copy_index: int = 0) -> dict[str, Any]:
    answer = f"<answer>\n{precursor(row)}\n</answer>"
    out = base_example(
        row,
        task="endpoint",
        system=ENDPOINT_SYSTEM,
        assistant=answer,
    )
    out["id"] = f"{row['id']}::endpoint:{copy_index}"
    out["structural_precursor"] = precursor(row)
    return out


def all_imports(row: dict[str, Any]) -> list[str]:
    plan = trace_plan(row)
    values = [str(value) for value in plan.get("initial_imports") or []]
    for step in plan["steps"]:
        values.extend(str(value) for value in step.get("imports") or [])
    # Preserve chemical strings but remove duplicate bookkeeping imports.
    return sorted(set(values))


def mapped_mol(smiles: str, *, label: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{label} does not parse")
    maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if not maps or min(maps) <= 0 or len(set(maps)) != len(maps):
        raise ValueError(f"{label} must have unique positive atom maps")
    return mol


def atom_map_index(mol: Chem.Mol) -> dict[int, Chem.Atom]:
    return {int(atom.GetAtomMapNum()): atom for atom in mol.GetAtoms()}


def bond_table(mol: Chem.Mol) -> dict[tuple[int, int], str]:
    output: dict[tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        left = int(bond.GetBeginAtom().GetAtomMapNum())
        right = int(bond.GetEndAtom().GetAtomMapNum())
        output[tuple(sorted((left, right)))] = str(bond.GetBondType())
    return output


def net_edit_target(row: dict[str, Any]) -> str:
    p = mapped_mol(product(row), label="product")
    r = mapped_mol(precursor(row), label="precursor")
    p_atoms = atom_map_index(p)
    r_atoms = atom_map_index(r)
    p_bonds = bond_table(p)
    r_bonds = bond_table(r)

    lines = ["NET_EDIT_AUX v1"]
    lines.extend(f"IMPORT {value}" for value in all_imports(row))

    for pair in sorted(set(p_bonds) | set(r_bonds)):
        before = p_bonds.get(pair, "NONE")
        after = r_bonds.get(pair, "NONE")
        if before != after:
            lines.append(f"BOND {pair[0]} {pair[1]} {before} -> {after}")

    for atom_map in sorted(set(p_atoms) & set(r_atoms)):
        q0 = int(p_atoms[atom_map].GetFormalCharge())
        q1 = int(r_atoms[atom_map].GetFormalCharge())
        if q0 != q1:
            lines.append(f"CHARGE {atom_map} {q0} -> {q1}")
        h0 = int(p_atoms[atom_map].GetTotalNumHs())
        h1 = int(r_atoms[atom_map].GetTotalNumHs())
        if h0 != h1:
            lines.append(f"HYDROGEN {atom_map} {h0} -> {h1}")

    deleted = sorted(set(p_atoms) - set(r_atoms))
    if deleted:
        lines.append("DELETE_MAPS " + ",".join(str(value) for value in deleted))
    lines.append("END")
    return "<net_edit>\n" + "\n".join(lines) + "\n</net_edit>"


def canonical_move(move: dict[str, Any]) -> str:
    if move.get("mode") == "BE_DELTA":
        return "BE_DELTA " + compact(
            {
                "bond_deltas": move.get("bond_deltas") or [],
                "charge_actions": move.get("charge_actions") or [],
                "lone_pair_deltas": move.get("lone_pair_deltas") or [],
            }
        )
    source = dict(move.get("source") or {})
    sink = dict(move.get("sink") or {})
    source_atoms = ",".join(str(int(x)) for x in source.get("atoms") or [])
    sink_atoms = ",".join(str(int(x)) for x in sink.get("atoms") or [])
    return (
        f"{source.get('kind','UNKNOWN')}({source_atoms})"
        f" -> {sink.get('kind','UNKNOWN')}({sink_atoms})"
    )


def reverse_et_target(row: dict[str, Any]) -> str:
    plan = trace_plan(row)
    lines = ["REVERSE_ET_AUX v1"]
    lines.extend(f"IMPORT {value}" for value in all_imports(row))
    for step_index, step in enumerate(plan["steps"]):
        moves = sorted(canonical_move(dict(move)) for move in step.get("moves") or [])
        if not moves:
            raise ValueError(f"{row.get('id')}: empty moves at step {step_index}")
        lines.append(f"STEP {step_index}")
        lines.extend(f"  ET {move}" for move in moves)
    lines.append("END")
    return "<reverse_et>\n" + "\n".join(lines) + "\n</reverse_et>"


def netedit_example(row: dict[str, Any]) -> dict[str, Any]:
    return base_example(
        row,
        task="net_edit_aux",
        system=NETEDIT_SYSTEM,
        assistant=net_edit_target(row),
    )


def reverse_et_example(row: dict[str, Any]) -> dict[str, Any]:
    return base_example(
        row,
        task="reverse_et_aux",
        system=REVERSE_ET_SYSTEM,
        assistant=reverse_et_target(row),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_split(source: Path, output_dir: Path, split: str) -> dict[str, Any]:
    source_rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = EXPECTED[split]
    if len(source_rows) != expected:
        raise ValueError(f"{split}: {len(source_rows)} != {expected}")
    ids = [str(row.get("id") or "") for row in source_rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{split}: missing or duplicate reaction IDs")

    endpoint_rows = [endpoint_example(row, copy_index=0) for row in source_rows]
    netedit_rows = [netedit_example(row) for row in source_rows]
    reverse_rows = [reverse_et_example(row) for row in source_rows]

    # Training files are intentionally the same size: two examples per reaction.
    endpoint_x2 = []
    endpoint_netedit = []
    endpoint_reverse = []
    for row, endpoint, netedit, reverse in zip(
        source_rows, endpoint_rows, netedit_rows, reverse_rows, strict=True
    ):
        endpoint_x2.extend(
            [endpoint, endpoint_example(row, copy_index=1)]
        )
        endpoint_netedit.extend([endpoint, netedit])
        endpoint_reverse.extend([endpoint, reverse])

    write_jsonl(output_dir / split / "endpoint_eval.jsonl", endpoint_rows)
    write_jsonl(output_dir / split / "net_edit_aux.jsonl", netedit_rows)
    write_jsonl(output_dir / split / "reverse_et_aux.jsonl", reverse_rows)
    if split == "train":
        write_jsonl(output_dir / "train_endpoint_only_matched.jsonl", endpoint_x2)
        write_jsonl(output_dir / "train_endpoint_plus_netedit.jsonl", endpoint_netedit)
        write_jsonl(output_dir / "train_endpoint_plus_reverse_et.jsonl", endpoint_reverse)

    return {
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "reaction_rows": len(source_rows),
        "reaction_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "endpoint_rows": len(endpoint_rows),
        "net_edit_aux_rows": len(netedit_rows),
        "reverse_et_aux_rows": len(reverse_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/flower_inverse_tool_sft_action_delta_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reverse_et_transfer_v1"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=["train", "valid", "test"],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for split in args.splits:
        reports[split] = build_split(
            args.source_dir / f"{split}.jsonl",
            args.output_dir,
            split,
        )

    manifest = {
        "schema_version": 1,
        "artifact_type": "reverse_et_transfer_matched_supervision_v1",
        "source_contract": "FlowER strict executable common IDs",
        "conditions": {
            "endpoint_only": "two endpoint examples per reaction",
            "endpoint_plus_netedit": "one endpoint + one net-edit auxiliary",
            "endpoint_plus_reverse_et": "one endpoint + one reverse-ET auxiliary",
        },
        "evaluation_contract": (
            "all conditions use endpoint-only product->precursor inference; "
            "auxiliary tasks are never invoked at test time"
        ),
        "same_reaction_ids_across_conditions": True,
        "same_examples_per_reaction": 2,
        "auxiliary_contains_precursor_answer": False,
        "reports": reports,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
