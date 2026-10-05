#!/usr/bin/env python3
"""Build a principal-product-aware history view without changing actions/states.

The reference full-endpoint product chooses *which existing component* of the
strict final mixture is named in the prompt. The visible molecule and every
recorded executor transition remain byte-for-byte unchanged. This is a new
proxy-target representation, not recovery of original patent product labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.train_system_one_electron_flow import verify_source

ARTIFACT = "mech_uspto_31k_natural_language_history_principal_target_v1"
CONTRACT = "principal_product_target_line_with_unchanged_executor_mixture_v1"
EXPECTED_FULL = {"train": 24959, "valid": 3120, "test": 3120}


def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError(f"invalid molecular SMILES: {smiles[:120]}")
    return molecule


def principal_component_from_mixture(mixture: str, proxy_product: str) -> tuple[str, bool]:
    """Choose the proxy-matched component, preserving strict-view stereotags."""
    state = _mol(mixture)
    product = _mol(proxy_product)
    key = Chem.MolToSmiles(product, canonical=True, isomericSmiles=False)
    matches = [
        Chem.MolFragmentToSmiles(state, atomsToUse=list(indices),
                                 canonical=True, isomericSmiles=True)
        for indices in Chem.GetMolFrags(state)
        if Chem.MolFragmentToSmiles(state, atomsToUse=list(indices),
                                    canonical=True, isomericSmiles=False) == key
    ]
    if not matches:
        raise ValueError("full-endpoint proxy product is absent from the strict final mixture")
    if len(set(matches)) > 1:
        raise ValueError("multiple constitutional matches have distinct stereochemistry")
    return matches[0], len(matches) > 1


def convert_row(row: dict, principal: str) -> dict:
    """Replace only the target prompt line; leave action and executor fields intact."""
    output = dict(row)
    messages = [dict(message) for message in row["messages"]]
    user_indices = [index for index, message in enumerate(messages)
                    if message.get("role") == "user"]
    if len(user_indices) != 1:
        raise ValueError(f"{row['id']}: expected exactly one user observation")
    index = user_indices[0]
    content = str(messages[index]["content"])
    expected = f"TARGET PRODUCT SMILES: {row['target_smiles']}\n"
    if not content.startswith(expected):
        raise ValueError(f"{row['id']}: source target prompt differs from strict source")
    messages[index]["content"] = (
        f"TARGET PRODUCT SMILES: {principal}\n" + content[len(expected):]
    )
    output["messages"] = messages
    output["principal_product_smiles"] = principal
    output["metadata"] = dict(row["metadata"], target_prompt_contract=CONTRACT)
    if (output["id"] != row["id"]
            or output["target_smiles"] != row["target_smiles"]
            or output.get("expected_precursor") != row.get("expected_precursor")
            or [message for message in messages if message["role"] != "user"]
            != [message for message in row["messages"] if message["role"] != "user"]):
        raise ValueError(f"{row['id']}: action/executor state changed during prompt conversion")
    return output


def load_products(full_dir: Path, split: str) -> tuple[dict[str, str], str]:
    manifest = json.loads((full_dir / "manifest.json").read_text())
    path = full_dir / f"{split}.jsonl"
    declared = manifest["splits"][split]
    if (manifest["product_source_field"] != "rxn_prod_equ"
            or manifest["product_selection_is_proxy"] is not True
            or manifest["executor_filtering"] is not False
            or declared["rows"] != EXPECTED_FULL[split]
            or declared["endpoint_sha256"] != sha256(path)):
        raise ValueError(f"{split}: equ-proxy endpoint source contract mismatch")
    products = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in products:
            raise ValueError(f"{split}: duplicate endpoint reaction ID")
        products[reaction_id] = str(row["product_unmapped"])
    if len(products) != EXPECTED_FULL[split]:
        raise ValueError(f"{split}: endpoint denominator mismatch")
    return products, declared["endpoint_sha256"]


def convert_split(source_dir: Path, full_dir: Path, output_dir: Path,
                  split: str) -> dict:
    source_path = source_dir / f"{split}.jsonl"
    source = verify_source(source_path)
    products, endpoint_sha = load_products(full_dir, split)
    output_path = output_dir / f"{split}.jsonl"
    seen: set[str] = set()
    principal_by_id: dict[str, str] = {}
    previous_id = ""
    counts: Counter[str] = Counter()
    with source_path.open() as reader, output_path.open("w") as writer:
        for line in reader:
            if not line.strip():
                raise ValueError(f"{split}: blank source decision row")
            row = json.loads(line)
            reaction_id = str(row["metadata"]["reaction_id"])
            if reaction_id != str(row["source_id"]):
                raise ValueError(f"{split}: source and metadata reaction IDs differ")
            if reaction_id != previous_id:
                if reaction_id in seen:
                    raise ValueError(f"{split}: noncontiguous reaction ID")
                if reaction_id not in products:
                    raise ValueError(f"{split}: source reaction missing from endpoint handoff")
                seen.add(reaction_id)
                previous_id = reaction_id
                principal, symmetric = principal_component_from_mixture(
                    str(row["target_smiles"]), products[reaction_id]
                )
                principal_by_id[reaction_id] = principal
                counts["symmetry_equivalent_reactions"] += int(symmetric)
            elif principal_by_id[reaction_id] != principal_component_from_mixture(
                str(row["target_smiles"]), products[reaction_id]
            )[0]:
                raise ValueError(f"{split}: target mixture changed within reaction")
            converted = convert_row(row, principal_by_id[reaction_id])
            writer.write(json.dumps(converted, separators=(",", ":")) + "\n")
            counts["decision_rows"] += 1
            counts[f"{row['metadata']['decision_type']}_decisions"] += 1
    if (counts["decision_rows"] != source["decision_rows"]
            or len(seen) != source["reaction_denominator"]
            or counts["event_decisions"] != source["event_decisions"]):
        raise ValueError(f"{split}: strict source denominator/action counts changed")
    return {
        "source_sha256": source["sha256"],
        "full_endpoint_sha256": endpoint_sha,
        "output_sha256": sha256(output_path),
        "reactions": len(seen),
        **dict(counts),
    }


def build(source_dir: Path, full_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    full_manifest_path = full_dir / "manifest.json"
    if (source_manifest["artifact_type"]
            != "mech_uspto_31k_natural_language_electron_event_history_v2"
            or source_manifest["training_allowed"] is not True):
        raise ValueError("source must be validated current-compiler history-v2")
    output_dir.mkdir(parents=True)
    splits = {split: convert_split(source_dir, full_dir, output_dir, split)
              for split in ("train", "valid", "test")}
    manifest = {
        "artifact_type": ARTIFACT,
        "training_allowed": False,
        "target_prompt_contract": CONTRACT,
        "full_endpoint_product_field": "rxn_prod_equ",
        "full_endpoint_product_selection_is_proxy": True,
        "source_manifest_sha256": sha256(source_manifest_path),
        "full_endpoint_manifest_sha256": sha256(full_manifest_path),
        "reaction_denominator": {split: result["reactions"] for split, result in splits.items()},
        "decision_rows": {split: result["decision_rows"] for split, result in splits.items()},
        "splits": splits,
        "limitations": [
            "strict executable trace-view only; not all 31,199 endpoint reactions",
            "principal product chosen from rxn_prod_equ largest-organic proxy, not original reaction table",
            "assistant actions and executor states are unchanged; only model-visible target line differs",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    status = {
        "artifact_id": ARTIFACT,
        "status": "built_pending_independent_audit",
        "training_allowed": False,
        "rows": manifest["reaction_denominator"],
        "full_reaction_denominator": EXPECTED_FULL,
        "target_prompt_contract": CONTRACT,
    }
    (output_dir / "ARTIFACT_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_dir, args.full_endpoint_dir, args.output_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
