#!/usr/bin/env python3
"""Validate cached MechET graphs and run one tiny CPU training step."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from retflow.datasets.data.uspto import USPTO
from retflow.datasets.data.uspto_synthon import SynthonUSPTO
from retflow.datasets.info import RetrosynthesisInfo
from retflow.datasets.retro import RetroDataset
from retflow.datasets.synthon import SynthonDataset
from retflow.methods.discrete_fm.basic import GraphDiscreteFM
from retflow.models.graph_transformer import GraphTransformer
from retflow.utils import ExtraFeatures, GraphModelLayerInfo, GraphModelWrapper
from retflow.utils.data import to_dense
from retflow.utils.wrappers import GraphWrapper


RAW_FILES = {
    "train": "uspto50k_train.csv",
    "val": "uspto50k_val.csv",
    "test": "uspto50k_test.csv",
}


def _raw_stable_ids(root: Path, split: str) -> list[str]:
    with (root / "raw" / RAW_FILES[split]).open(newline="", encoding="utf-8") as handle:
        return [str(row["stable_id"]) for row in csv.DictReader(handle)]


def _validate_edges(data, prefix: str) -> None:
    edge_index = getattr(data, f"{prefix}edge_index")
    edge_attr = getattr(data, f"{prefix}edge_attr")
    x = getattr(data, f"{prefix}x")
    if edge_index.shape[0] != 2 or edge_index.shape[1] != edge_attr.shape[0]:
        raise AssertionError(f"misaligned {prefix or 'reactant'} edges")
    if edge_index.numel() and int(edge_index.max()) >= x.shape[0]:
        raise AssertionError(f"out-of-range {prefix or 'reactant'} edge")
    if edge_attr.shape[1] != len(RetrosynthesisInfo.bonds) + 1:
        raise AssertionError(f"wrong {prefix or 'reactant'} edge feature width")
    if edge_attr.numel() and not torch.allclose(
        edge_attr.sum(dim=-1), torch.ones(edge_attr.shape[0])
    ):
        raise AssertionError(f"non-one-hot {prefix or 'reactant'} edge feature")
    edge_set = {tuple(edge) for edge in edge_index.t().tolist()}
    if any((target, source) not in edge_set for source, target in edge_set):
        raise AssertionError(f"asymmetric {prefix or 'reactant'} edges")


def _validate_dataset(
    dataset, root: Path, split: str, builder_name: str, synthon: bool
) -> dict:
    raw_ids = _raw_stable_ids(root, split)
    report_path = root / "processed" / f"{builder_name}_{split}.preprocessing.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failure_path = Path(report["failure_file"])
    failed_ids = {
        json.loads(line)["stable_id"]
        for line in failure_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected_ids = [stable_id for stable_id in raw_ids if stable_id not in failed_ids]
    actual_ids = [str(dataset[index].stable_id) for index in range(len(dataset))]
    if actual_ids != expected_ids:
        raise AssertionError(
            "cached stable IDs do not match the supported raw rows in original order"
        )
    if len(actual_ids) != len(set(actual_ids)):
        raise AssertionError("cached stable IDs are not unique")

    prefixes = ["", "p_", "s_"] if synthon else ["", "p_"]
    maximum_dummy = {prefix or "r_": 0 for prefix in prefixes}
    maximum_nodes = 0
    for data in dataset:
        node_counts = set()
        for prefix in prefixes:
            x = getattr(data, f"{prefix}x")
            if x.shape[1] != len(RetrosynthesisInfo.atom_encoder):
                raise AssertionError(f"wrong {prefix or 'reactant'} atom feature width")
            if not torch.allclose(x.sum(dim=-1), torch.ones(x.shape[0])):
                raise AssertionError(f"non-one-hot {prefix or 'reactant'} atom feature")
            node_counts.add(x.shape[0])
            maximum_dummy[prefix or "r_"] = max(
                maximum_dummy[prefix or "r_"], int(x[:, -1].sum().item())
            )
            _validate_edges(data, prefix)
        if len(node_counts) != 1:
            raise AssertionError("reaction graphs do not share a node coordinate system")
        maximum_nodes = max(maximum_nodes, node_counts.pop())

    return {
        "rows": len(dataset),
        "input_rows": len(raw_ids),
        "preprocessing_failures": len(failed_ids),
        "stable_ids_unique": len(set(actual_ids)),
        "maximum_nodes": maximum_nodes,
        "maximum_dummy_nodes": maximum_dummy,
    }


def _tiny_train_step(train_dataset, val_dataset, synthon: bool) -> dict:
    descriptor = (
        SynthonDataset(name="smoke", batch_size=2)
        if synthon
        else RetroDataset(name="smoke", batch_size=2)
    )
    info = descriptor._get_info(train_dataset, val_dataset)
    batch = next(iter(DataLoader(train_dataset, batch_size=2, shuffle=False)))

    reactants, reactant_mask = to_dense(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch
    )
    reactants = reactants.mask(reactant_mask)
    source_prefix = "s_" if synthon else "p_"
    source, source_mask = to_dense(
        getattr(batch, f"{source_prefix}x"),
        getattr(batch, f"{source_prefix}edge_index"),
        getattr(batch, f"{source_prefix}edge_attr"),
        batch.batch,
    )
    source = source.mask(source_mask)
    if not torch.equal(reactant_mask, source_mask):
        raise AssertionError("source and target dense masks differ")

    if synthon:
        product, product_mask = to_dense(
            batch.p_x, batch.p_edge_index, batch.p_edge_attr, batch.batch
        )
        product = product.mask(product_mask)
        if not torch.equal(source_mask, product_mask):
            raise AssertionError("synthon and product dense masks differ")
        context = [source.clone(), product.clone()]
    else:
        context = source.clone()

    layer_dims = GraphModelLayerInfo(dim_X=32, dim_E=16, dim_y=16)
    model = GraphTransformer(
        n_layers=1,
        n_head=2,
        ff_dims=layer_dims,
        hidden_mlp_dims=layer_dims,
        hidden_dims=layer_dims,
    )
    method = GraphDiscreteFM(steps=2)
    torch_model = method.setup(info, model, device="cpu")
    wrapper = GraphModelWrapper(torch_model, ExtraFeatures(info.max_n_nodes))
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-4)

    noisy_graph, node_mask, t_float = method.apply_noise(
        X=source.X,
        E=source.E,
        y=source.y,
        X_T=reactants.X,
        E_T=reactants.E,
        y_T=reactants.y,
        node_mask=source_mask,
    )
    predicted_x, predicted_e = wrapper(
        noisy_graph, node_mask, context, t_float
    )
    prediction = GraphWrapper(predicted_x, predicted_e, source.y).mask(node_mask)
    loss, node_loss, edge_loss = method.compute_loss(
        reactants, prediction, node_mask, noisy_graph, t_float
    )
    if not all(math.isfinite(value.item()) for value in (loss, node_loss, edge_loss)):
        raise AssertionError("non-finite smoke-test loss")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "batch_size": 2,
        "input_dim": vars(info.input_dim),
        "output_dim": vars(info.output_dim),
        "max_n_nodes": info.max_n_nodes,
        "max_n_dummy_nodes": info.max_n_dummy_nodes,
        "max_n_product_dummy_nodes": info.max_n_product_dummy_nodes,
        "loss": loss.item(),
        "node_loss": node_loss.item(),
        "edge_loss": edge_loss.item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-roots", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--builders",
        nargs="+",
        choices=("direct", "synthon"),
        default=("direct", "synthon"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)

    results = []
    for unresolved_root in args.dataset_roots:
        root = unresolved_root.resolve()
        for builder_name, dataset_class, synthon in (
            ("direct", USPTO, False),
            ("synthon", SynthonUSPTO, True),
        ):
            if builder_name not in args.builders:
                continue
            datasets = {
                split: dataset_class(split=split, root=str(root))
                for split in RAW_FILES
            }
            split_results = {
                split: _validate_dataset(
                    dataset, root, split, builder_name, synthon
                )
                for split, dataset in datasets.items()
            }
            results.append(
                {
                    "dataset_root": str(root),
                    "builder": builder_name,
                    "splits": split_results,
                    "tiny_train_step": _tiny_train_step(
                        datasets["train"], datasets["val"], synthon
                    ),
                    "ok": True,
                }
            )

    payload = {"results": results}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
