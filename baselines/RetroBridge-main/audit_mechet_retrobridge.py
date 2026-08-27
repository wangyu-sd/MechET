#!/usr/bin/env python3
"""Run the frozen-ID/identity audit on processed MechET RetroBridge graphs."""

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import networkx as nx
from rdkit import Chem

from src.data.retrobridge_dataset import RetroBridgeDataset


STAGES = ("train", "val", "test")
STAGE_CSV = {
    "train": "uspto50k_train.csv",
    "val": "uspto50k_val.csv",
    "test": "uspto50k_test.csv",
}


def canonical_mapped(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def canonical_unmapped(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True)


def source_native_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    graph = nx.Graph()
    bond_types = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    for atom in molecule.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in molecule.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond_types[bond.GetBondType()],
        )
    return graph


def processed_native_graph(graph_data, node_key, edge_index_key, edge_attr_key, atom_decoder):
    graph = nx.Graph()
    node_types = graph_data[node_key].argmax(dim=-1).tolist()
    real_nodes = {}
    for node_index, type_index in enumerate(node_types):
        element = atom_decoder[type_index]
        if element != "*":
            real_nodes[node_index] = element
            graph.add_node(node_index, element=element)
    edge_index = graph_data[edge_index_key]
    edge_types = graph_data[edge_attr_key].argmax(dim=-1).tolist()
    for edge_position, bond_type in enumerate(edge_types):
        start = int(edge_index[0, edge_position])
        end = int(edge_index[1, edge_position])
        if start in real_nodes and end in real_nodes and bond_type:
            graph.add_edge(start, end, bond_type=bond_type)
    return graph


def native_graphs_match(left, right):
    return nx.is_isomorphic(
        left,
        right,
        node_match=nx.algorithms.isomorphism.categorical_node_match("element", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("bond_type", None),
    )


def audit_split(data_root, stage):
    dataset = RetroBridgeDataset(stage=stage, root=str(data_root), extra_nodes=True)
    expected_count = data_root.joinpath("metadata.json")
    metadata = json.loads(expected_count.read_text())
    metadata_split = "valid" if stage == "val" else stage
    expected = metadata["splits"][metadata_split]
    with (data_root / "raw" / STAGE_CSV[stage]).open() as handle:
        source_rows = list(csv.DictReader(handle))

    id_mismatches = []
    mapped_identity_mismatches = []
    unmapped_identity_mismatches = []
    native_graph_identity_mismatches = []
    observed_ids = []
    for index in range(len(dataset)):
        graph = dataset[index]
        source_row = source_rows[index]
        stable_id = str(graph.stable_id)
        observed_ids.append(stable_id)
        if stable_id != source_row["id"]:
            id_mismatches.append({
                "index": index,
                "stable_id": stable_id,
                "expected": source_row["id"],
            })

        expected_precursor, _, expected_product = source_row[
            "reactants>reagents>production"
        ].split(">")

        for label, actual, expected_smiles in (
            ("precursor", graph.r_smiles, expected_precursor),
            ("product", graph.p_smiles, expected_product),
        ):
            if canonical_mapped(actual) != canonical_mapped(expected_smiles):
                mapped_identity_mismatches.append({"index": index, "stable_id": stable_id, "field": label})
            if canonical_unmapped(actual) != canonical_unmapped(expected_smiles):
                unmapped_identity_mismatches.append({"index": index, "stable_id": stable_id, "field": label})

        for label, expected_smiles, node_key, edge_index_key, edge_attr_key in (
            ("precursor", expected_precursor, "x", "edge_index", "edge_attr"),
            ("product", expected_product, "p_x", "p_edge_index", "p_edge_attr"),
        ):
            expected_graph = source_native_graph(expected_smiles)
            actual_graph = processed_native_graph(
                graph,
                node_key,
                edge_index_key,
                edge_attr_key,
                metadata["atom_decoder"],
            )
            if not native_graphs_match(actual_graph, expected_graph):
                native_graph_identity_mismatches.append({
                    "index": index,
                    "stable_id": stable_id,
                    "field": label,
                })

    return {
        "stage": stage,
        "expected_count": expected["count"],
        "processed_count": len(dataset),
        "count_matches": len(dataset) == expected["count"],
        "unique_stable_ids": len(set(observed_ids)),
        "stable_ids_unique": len(set(observed_ids)) == len(observed_ids),
        "id_mismatch_count": len(id_mismatches),
        "mapped_identity_mismatch_count": len(mapped_identity_mismatches),
        "unmapped_identity_mismatch_count": len(unmapped_identity_mismatches),
        "native_graph_identity_mismatch_count": len(native_graph_identity_mismatches),
        "first_stable_id": observed_ids[0] if observed_ids else None,
        "last_stable_id": observed_ids[-1] if observed_ids else None,
    }


def package_versions():
    names = ["torch", "torch-geometric", "pytorch-lightning", "rdkit", "pandas", "numpy"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def repository_revision(repo_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable_not_a_git_checkout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()

    metadata = json.loads((args.data_root / "metadata.json").read_text())
    split_audits = [audit_split(args.data_root, stage) for stage in args.stages]
    passed = all(
        item["count_matches"]
        and item["stable_ids_unique"]
        and item["id_mismatch_count"] == 0
        and item["mapped_identity_mismatch_count"] == 0
        and item["unmapped_identity_mismatch_count"] == 0
        and item["native_graph_identity_mismatch_count"] == 0
        for item in split_audits
    )
    report = {
        "passed": passed,
        "data_root": str(args.data_root.resolve()),
        "repository_revision": repository_revision(Path(__file__).resolve().parent),
        "python": platform.python_version(),
        "packages": package_versions(),
        "vocabulary_source": metadata.get("vocabulary_source", "selected_train_rows"),
        "train_only_vocabulary": metadata.get("vocabulary_source_splits") == ["train"],
        "published_fixed_vocabulary": metadata.get("vocabulary_source", "").startswith("published_"),
        "vocabulary_leakage_free": not (
            {"valid", "val", "test"} & set(metadata.get("vocabulary_source_splits", []))
        ),
        "train_only_dummy_capacity": metadata.get("capacity_source_splits") == ["train"],
        "atom_decoder": metadata["atom_decoder"],
        "max_n_dummy_nodes": metadata["max_n_dummy_nodes"],
        "splits": split_audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
