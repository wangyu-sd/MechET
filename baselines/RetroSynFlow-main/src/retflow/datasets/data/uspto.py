import os
import subprocess
from typing import Any, Dict, Sequence

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from retflow.datasets.info import DOWNLOAD_URL_TEMPLATE, RetrosynthesisInfo
from retflow.datasets.data.preprocess_utils import (
    row_value,
    stable_id_for_row,
    write_preprocessing_audit,
)
from retflow.utils.data import (
    build_graph_from_mol_with_mapping,
    compute_joint_nodes_order_mapping,
)


class USPTO(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        download_and_process=False,
        swap=False,
        max_dummy_nodes=RetrosynthesisInfo.max_n_dummy_nodes,
        allow_failed_rows=False,
    ):
        self.split = split
        self.max_dummy_nodes = max_dummy_nodes
        self.allow_failed_rows = allow_failed_rows

        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        elif self.split == "test":
            self.file_idx = 2
        else:
            raise NotImplementedError
        self.download_and_process = download_and_process
        super().__init__(root=root)

        self.slices: Dict
        self.data, self.slices = torch.load(
            self.processed_paths[self.file_idx], weights_only=False
        )

        if swap:
            self.data = Data(
                x=self.data.p_x,
                edge_index=self.data.p_edge_index,
                edge_attr=self.data.p_edge_attr,
                p_x=self.data.x,
                p_edge_index=self.data.edge_index,
                p_edge_attr=self.data.edge_attr,
                y=self.data.y,
                idx=self.data.idx,
                r_smiles=self.data.p_smiles,
                p_smiles=self.data.r_smiles,
            )
            self.slices = {
                "x": self.slices["p_x"],
                "edge_index": self.slices["p_edge_index"],
                "edge_attr": self.slices["p_edge_attr"],
                "y": self.slices["y"],
                "idx": self.slices["idx"],
                "p_x": self.slices["x"],
                "p_edge_index": self.slices["edge_index"],
                "p_edge_attr": self.slices["edge_attr"],
                "r_smiles": self.slices["p_smiles"],
                "p_smiles": self.slices["r_smiles"],
            }

    @property
    def raw_file_names(self):
        return ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]

    @property
    def split_file_name(self):
        return ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return [f"train.pt", f"val.pt", f"test.pt"]

    def download(self):
        if self.download_and_process:
            os.makedirs(self.raw_dir, exist_ok=True)
            for fname in self.raw_file_names:
                url = DOWNLOAD_URL_TEMPLATE.format(fname=fname)
                path = os.path.join(self.raw_dir, fname)
                subprocess.run(f"wget {url} -O {path}", shell=True)

    def process(self):
        if self.download_and_process:
            self._process_data()

    def _process_data(self):
        table = pd.read_csv(self.split_paths[self.file_idx])
        data_list = []
        failures = []
        input_stable_ids = [
            stable_id_for_row(row, i) for i, (_, row) in enumerate(table.iterrows())
        ]
        output_stable_ids = []
        max_product_dummy_nodes = 0
        max_reactant_dummy_nodes = 0
        for i, (_, row) in enumerate(table.iterrows()):
            stable_id = input_stable_ids[i]
            reaction_smiles = str(row["reactants>reagents>production"])
            try:
                parts = reaction_smiles.split(">")
                if len(parts) != 3:
                    raise ValueError("reaction SMILES must contain exactly two '>' separators")
                reactants_smi, _, product_smi = parts
                rmol = Chem.MolFromSmiles(reactants_smi)
                pmol = Chem.MolFromSmiles(product_smi)
                if rmol is None or pmol is None:
                    raise ValueError("RDKit could not parse the reactants or product")

                mapping = compute_joint_nodes_order_mapping(rmol, pmol)
                required_dummy_nodes = len(mapping) - pmol.GetNumAtoms()
                if required_dummy_nodes > self.max_dummy_nodes:
                    raise ValueError(
                        f"reaction requires {required_dummy_nodes} product dummy nodes, "
                        f"but capacity is {self.max_dummy_nodes}"
                    )
                num_nodes = pmol.GetNumAtoms() + self.max_dummy_nodes
                r_x, r_edge_index, r_edge_attr = build_graph_from_mol_with_mapping(
                    rmol,
                    mapping,
                    num_nodes,
                    types=RetrosynthesisInfo.atom_encoder,
                    bonds=RetrosynthesisInfo.bonds,
                )
                p_x, p_edge_index, p_edge_attr = build_graph_from_mol_with_mapping(
                    pmol,
                    mapping,
                    num_nodes,
                    types=RetrosynthesisInfo.atom_encoder,
                    bonds=RetrosynthesisInfo.bonds,
                )
                reactant_mask = ~(r_x[:, -1].bool()).squeeze()
                product_mask = ~(p_x[:, -1].bool()).squeeze()
                shared_mask = reactant_mask & product_mask
                if not torch.allclose(r_x[shared_mask], p_x[shared_mask]):
                    raise ValueError("shared atom maps disagree on atom identity")

                max_product_dummy_nodes = max(
                    max_product_dummy_nodes, int((~product_mask).sum().item())
                )
                max_reactant_dummy_nodes = max(
                    max_reactant_dummy_nodes, int((~reactant_mask).sum().item())
                )

                # Shuffle deterministically without changing stable-ID order.
                generator = torch.Generator().manual_seed(i)
                new2old_idx = torch.randperm(num_nodes, generator=generator).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(num_nodes)

                r_x = r_x[new2old_idx]
                r_edge_index = torch.stack(
                    [old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]], dim=0
                )
                r_edge_index, r_edge_attr = self.sort_edges(
                    r_edge_index, r_edge_attr, num_nodes
                )

                p_x = p_x[new2old_idx]
                p_edge_index = torch.stack(
                    [old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]], dim=0
                )
                p_edge_index, p_edge_attr = self.sort_edges(
                    p_edge_index, p_edge_attr, num_nodes
                )

                y = torch.zeros(size=(1, 0), dtype=torch.float)
                data = Data(
                    x=r_x,
                    edge_index=r_edge_index,
                    edge_attr=r_edge_attr,
                    y=y,
                    idx=i,
                    source_row_index=int(row_value(row, "source_row_index", i)),
                    source_reaction_id=str(row_value(row, "source_reaction_id")),
                    stable_id=stable_id,
                    p_x=p_x,
                    p_edge_index=p_edge_index,
                    p_edge_attr=p_edge_attr,
                    r_smiles=reactants_smi,
                    p_smiles=product_smi,
                )
                data_list.append(data)
                output_stable_ids.append(stable_id)
            except Exception as error:
                raw_parts = reaction_smiles.split(">")
                failures.append(
                    {
                        "row_index": i,
                        "stable_id": stable_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "reference_precursors": raw_parts[0] if len(raw_parts) == 3 else "",
                        "product": raw_parts[2] if len(raw_parts) == 3 else "",
                    }
                )

        report = write_preprocessing_audit(
            self.processed_dir,
            builder="direct",
            split=self.split,
            input_stable_ids=input_stable_ids,
            output_stable_ids=output_stable_ids,
            failures=failures,
            statistics={
                "configured_max_dummy_nodes": self.max_dummy_nodes,
                "max_product_dummy_nodes": max_product_dummy_nodes,
                "max_reactant_dummy_nodes": max_reactant_dummy_nodes,
            },
        )
        if not report["coverage_ok"] and not self.allow_failed_rows:
            raise RuntimeError(
                f"Strict direct preprocessing coverage failed for {self.split}: "
                f"{len(output_stable_ids)}/{len(input_stable_ids)} rows; "
                f"see {self.processed_dir}/direct_{self.split}.failures.jsonl"
            )
        if not data_list:
            raise RuntimeError(f"Direct preprocessing produced no rows for {self.split}")
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    @staticmethod
    def sort_edges(edge_index, edge_attr, max_num_nodes):
        if len(edge_attr) != 0:
            perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        return edge_index, edge_attr


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
