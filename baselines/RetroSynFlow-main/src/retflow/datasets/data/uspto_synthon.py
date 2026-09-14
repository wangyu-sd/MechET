import os
import subprocess
from typing import Dict

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torchdrug.data import Molecule
from tqdm import tqdm

from retflow.datasets.data.uspto import USPTO, to_list
from retflow.datasets.data.preprocess_utils import (
    row_value,
    stable_id_for_row,
    write_preprocessing_audit,
)
from retflow.datasets.info import DOWNLOAD_URL_TEMPLATE, RetrosynthesisInfo
from retflow.utils.data import (
    build_graph_from_mol_with_mapping,
    compute_joint_nodes_order_mapping,
    get_synthons,
    reactants_with_partial_atom_mapping,
)


def _generalized_synthon_mol(rmol, pmol):
    """Construct synthons for net reactions with one or more formed bonds.

    The published helper only emits synthons for a single formed bond (or one
    simple atom/bond modification). Complete reaction-level benchmarks also
    contain multi-center net reactions. For those rows, remove every product
    bond absent from the mapped precursor graph. If no bond was formed, the
    product itself is the synthon, matching the published modification cases.
    """
    reactant_bonds = set()
    for bond in rmol.GetBonds():
        begin = rmol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        end = rmol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        reactant_bonds.add(tuple(sorted((begin, end))))

    editable = Chem.RWMol(pmol)
    bonds_to_remove = []
    for bond in pmol.GetBonds():
        begin = pmol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        end = pmol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        if tuple(sorted((begin, end))) not in reactant_bonds:
            bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    for begin, end in bonds_to_remove:
        editable.RemoveBond(begin, end)
    return editable.GetMol(), len(bonds_to_remove)


class SynthonUSPTO(USPTO):
    def __init__(
        self,
        split,
        root,
        download_and_process=False,
        swap=False,
        max_dummy_nodes=RetrosynthesisInfo.max_n_dummy_nodes,
        allow_failed_rows=False,
        generalize_unsupported_synthons=False,
    ):
        self.generalize_unsupported_synthons = generalize_unsupported_synthons
        super().__init__(
            split,
            root,
            download_and_process,
            swap,
            max_dummy_nodes=max_dummy_nodes,
            allow_failed_rows=allow_failed_rows,
        )
        self.split = split

        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        elif self.split == "test":
            self.file_idx = 2
        else:
            raise NotImplementedError
        self.download_and_process = download_and_process
        

        self.slices: Dict
        self.data, self.slices = torch.load(
            self.processed_paths[self.file_idx], weights_only=False
        )

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
        # Keep synthon graphs separate from direct product->reactant graphs
        # when both builders are audited under the same dataset root.
        return ["synthon_train.pt", "synthon_val.pt", "synthon_test.pt"]

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
        max_synthon_dummy_nodes = 0
        max_product_dummy_nodes = 0
        generalized_synthon_rows = 0
        max_formed_bonds_removed = 0
        for i, (_, row) in tqdm(
            enumerate(table.iterrows()), total=len(table), desc=f"synthon-{self.split}"
        ):
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

                reactants = Molecule.from_molecule(
                    rmol, atom_feature="synthon_completion"
                )
                product = Molecule.from_molecule(
                    pmol, atom_feature="synthon_completion"
                )
                reactants.bond_stereo[:] = 0
                product.bond_stereo[:] = 0
                partial_reactants = reactants_with_partial_atom_mapping(
                    reactants, product, atom_feature="synthon_completion"
                )
                _, synthons = get_synthons(partial_reactants, product)
                if synthons:
                    full_synthons_smi = ".".join(s.to_smiles() for s in synthons)
                    synthons_mol = Chem.MolFromSmiles(
                        full_synthons_smi, sanitize=False
                    )
                elif self.generalize_unsupported_synthons:
                    synthons_mol, formed_bonds_removed = _generalized_synthon_mol(
                        rmol, pmol
                    )
                    full_synthons_smi = Chem.MolToSmiles(
                        synthons_mol, canonical=True
                    )
                    generalized_synthon_rows += 1
                    max_formed_bonds_removed = max(
                        max_formed_bonds_removed, formed_bonds_removed
                    )
                else:
                    raise ValueError(
                        "reaction is unsupported by the published single-center "
                        "synthon construction"
                    )
                if synthons_mol is None or synthons_mol.GetNumAtoms() == 0:
                    raise ValueError("RDKit could not construct a non-empty synthon graph")

                mapping = compute_joint_nodes_order_mapping(rmol, synthons_mol, pmol)
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
                s_x, s_edge_index, s_edge_attr = build_graph_from_mol_with_mapping(
                    synthons_mol,
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
                synthon_mask = ~(s_x[:, -1].bool()).squeeze()
                product_mask = ~(p_x[:, -1].bool()).squeeze()
                shared_reactant_synthon = reactant_mask & synthon_mask
                shared_product_synthon = product_mask & synthon_mask
                if not torch.allclose(
                    r_x[shared_reactant_synthon], s_x[shared_reactant_synthon]
                ):
                    raise ValueError("reactant and synthon atom identities disagree")
                if not torch.allclose(
                    p_x[shared_product_synthon], s_x[shared_product_synthon]
                ):
                    raise ValueError("product and synthon atom identities disagree")

                max_synthon_dummy_nodes = max(
                    max_synthon_dummy_nodes, int((~synthon_mask).sum().item())
                )
                max_product_dummy_nodes = max(
                    max_product_dummy_nodes, int((~product_mask).sum().item())
                )

                generator = torch.Generator().manual_seed(i)
                new2old_idx = torch.randperm(num_nodes, generator=generator).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(num_nodes)

                r_x = r_x[new2old_idx]
                r_edge_index = torch.stack(
                    [old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]],
                    dim=0,
                )
                r_edge_index, r_edge_attr = self.sort_edges(
                    r_edge_index, r_edge_attr, num_nodes
                )

                p_x = p_x[new2old_idx]
                p_edge_index = torch.stack(
                    [old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]],
                    dim=0,
                )
                p_edge_index, p_edge_attr = self.sort_edges(
                    p_edge_index, p_edge_attr, num_nodes
                )

                s_x = s_x[new2old_idx]
                s_edge_index = torch.stack(
                    [old2new_idx[s_edge_index[0]], old2new_idx[s_edge_index[1]]],
                    dim=0,
                )
                s_edge_index, s_edge_attr = self.sort_edges(
                    s_edge_index, s_edge_attr, num_nodes
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
                    s_x=s_x,
                    s_edge_index=s_edge_index,
                    s_edge_attr=s_edge_attr,
                    p_x=p_x,
                    p_edge_index=p_edge_index,
                    p_edge_attr=p_edge_attr,
                    r_smiles=reactants_smi,
                    s_smiles=full_synthons_smi,
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
            builder="synthon",
            split=self.split,
            input_stable_ids=input_stable_ids,
            output_stable_ids=output_stable_ids,
            failures=failures,
            statistics={
                "configured_max_dummy_nodes": self.max_dummy_nodes,
                "generalization_enabled": self.generalize_unsupported_synthons,
                "max_synthon_dummy_nodes": max_synthon_dummy_nodes,
                "max_product_dummy_nodes": max_product_dummy_nodes,
                "generalized_synthon_rows": generalized_synthon_rows,
                "max_formed_bonds_removed": max_formed_bonds_removed,
            },
        )
        if not report["coverage_ok"] and not self.allow_failed_rows:
            raise RuntimeError(
                f"Strict synthon preprocessing coverage failed for {self.split}: "
                f"{len(output_stable_ids)}/{len(input_stable_ids)} rows; "
                f"see {self.processed_dir}/synthon_{self.split}.failures.jsonl"
            )
        if not data_list:
            raise RuntimeError(f"Synthon preprocessing produced no rows for {self.split}")
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
