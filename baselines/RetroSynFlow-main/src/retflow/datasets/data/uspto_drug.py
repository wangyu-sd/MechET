import csv
import math
import os
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from torchdrug import utils
from torchdrug.data import Molecule
from torchdrug.datasets import USPTO50k
from tqdm import tqdm

from retflow.datasets.info import DOWNLOAD_URL_TEMPLATE, RetrosynthesisInfo
from retflow.datasets.data.preprocess_utils import write_preprocessing_audit
from retflow.utils import reactants_with_partial_atom_mapping


class _TorchDrugUSPTO(USPTO50k):
    def __init__(self, data_split, path, as_synthon=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.data_split = data_split
        self.as_synthon = as_synthon
        self.transform = None

        if self.data_split == "train":
            self.file_idx = 0
        elif self.data_split == "val":
            self.file_idx = 1
        elif self.data_split == "test":
            self.file_idx = 2
        else:
            raise NotImplementedError

        self.load_csv(
            str(Path(self.path) / self.raw_file_names[self.file_idx]),
            smiles_field="reactants>reagents>production",
            target_fields=self.target_fields,
            verbose=verbose,
            **kwargs
        )

        if as_synthon:
            prefix = "Computing synthons"
            process_fn = self._get_synthon
        else:
            prefix = "Computing reaction centers"
            process_fn = self._get_reaction_center

        data = self.data
        targets = self.targets
        ids = self.ids
        row_indices = self.source_row_indices
        row_metadata = self.row_metadata
        self.data = []
        self.targets = defaultdict(list)
        self.ids = []
        self.source_row_indices = []

        indexes = range(len(data))
        if verbose:
            indexes = tqdm(indexes, prefix)
        failures = list(self.parse_failures)
        for i in indexes:
            reactant, product = data[i]
            stable_id = ids[i]
            metadata = row_metadata[i]
            try:
                reactant.bond_stereo[:] = 0
                product.bond_stereo[:] = 0
                reactant_converted = reactants_with_partial_atom_mapping(
                    reactant, product, kwargs["atom_feature"]
                )
                reactants, products = process_fn(reactant_converted, product)
                if not reactants:
                    raise ValueError(
                        "reaction is unsupported by the published single-center "
                        "reaction-center construction"
                    )
            except Exception as error:
                failures.append(
                    {
                        "row_index": row_indices[i],
                        "stable_id": stable_id,
                        "stage": "reaction_center",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "reference_precursors": metadata["reference_precursors"],
                        "product": metadata["product"],
                    }
                )
                continue
            self.data += zip(reactants, products)
            for k in targets:
                new_k = self.target_alias.get(k, k)
                values = targets[k][i]
                if k == "class":
                    values = max(0, int(values) - 1)
                self.targets[new_k] += [values] * len(reactants)
            self.targets["sample id"] += [row_indices[i]] * len(reactants)
            self.ids += [stable_id] * len(reactants)
            self.source_row_indices += [row_indices[i]] * len(reactants)

        input_stable_ids = self.input_stable_ids
        report = write_preprocessing_audit(
            self.processed_dir,
            builder="center",
            split=self.data_split,
            input_stable_ids=input_stable_ids,
            output_stable_ids=self.ids,
            failures=failures,
            statistics={
                "published_single_center_only": True,
                "failed_rows_count_as_incorrect_at_eval": True,
            },
        )
        self.preprocessing_report = report
        self.preprocessing_failures = failures
        self.valid_rate = len(self.data) / len(input_stable_ids)
        if not self.data:
            raise RuntimeError(
                f"Reaction-center preprocessing produced no rows for {data_split}"
            )

    def load_csv(
        self, csv_file, smiles_field="smiles", target_fields=None, verbose=0, **kwargs
    ):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)
        self.input_stable_ids = []
        self.ids = []
        self.source_row_indices = []
        self.row_metadata = []
        self.parse_failures = []
        self.data = []
        self.targets = defaultdict(list)
        self.smiles_list = []
        with open(csv_file, "r") as fin:
            reader = csv.DictReader(fin)
            if verbose:
                reader = tqdm(
                    reader,
                    "Loading %s" % csv_file,
                    total=max(0, utils.get_line_count(csv_file) - 1),
                )
            for row_index, row in enumerate(reader):
                stable_id = str(row.get("stable_id") or row.get("id") or row_index)
                self.input_stable_ids.append(stable_id)
                reaction_smiles = str(row.get(smiles_field, ""))
                parts = reaction_smiles.split(">")
                metadata = {
                    "reference_precursors": parts[0] if len(parts) == 3 else "",
                    "product": parts[2] if len(parts) == 3 else "",
                }
                try:
                    if len(parts) != 3:
                        raise ValueError(
                            "reaction SMILES must contain exactly two '>' separators"
                        )
                    molecules = []
                    for smiles in (parts[0], parts[2]):
                        molecule = Chem.MolFromSmiles(smiles)
                        if molecule is None:
                            raise ValueError("RDKit could not parse reaction molecule")
                        molecules.append(Molecule.from_molecule(molecule, **kwargs))
                except Exception as error:
                    self.parse_failures.append(
                        {
                            "row_index": row_index,
                            "stable_id": stable_id,
                            "stage": "parse",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            **metadata,
                        }
                    )
                    continue

                self.data.append(molecules)
                self.smiles_list.append(reaction_smiles)
                self.ids.append(stable_id)
                self.source_row_indices.append(row_index)
                self.row_metadata.append(metadata)
                for field, value in row.items():
                    if field == smiles_field:
                        continue
                    if target_fields is None or field in target_fields:
                        parsed_value = utils.literal_eval(value)
                        if parsed_value == "":
                            parsed_value = math.nan
                        self.targets[field].append(parsed_value)
        self.transform = None

    @property
    def processed_dir(self):
        path = Path(self.path)
        dataset_root = path.parent if path.name == "raw" else path
        return dataset_root / "processed"

    @property
    def raw_file_names(self):
        return ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]

    @staticmethod
    def download(split, path):
        raw_file_names = ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]
        if split == "train":
            file_idx = 0
        elif split == "val":
            file_idx = 1
        elif split == "test":
            file_idx = 2
        url = DOWNLOAD_URL_TEMPLATE.format(fname=raw_file_names[file_idx])
        utils.download(url, path)
