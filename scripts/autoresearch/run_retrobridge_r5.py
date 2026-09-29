#!/usr/bin/env python3
"""Run the pinned, official RetroBridge checkpoint on frozen R5 products.

This is a diagnostic external source: candidates are ranked by empirical
sample frequency (the official evaluation's confidence), not by a fabricated
model log-probability. Product-only queries and all failures retain their
original 200-product denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


SOURCE_COMMIT = "5442b4f45edc2e956f1d1c1763bb94cefd4a3a13"
CHECKPOINT_SHA256 = "a78b9e251770e72a53b280fe0f19924e70f52098b2e7b1178538bf8d29bc70cc"
SOURCE_URL = "https://github.com/igashov/retrobridge"
CHECKPOINT_URL = "https://zenodo.org/records/10688201"


def rank_samples(samples: list[str], *, top_k: int = 5) -> list[dict[str, Any]]:
    """Official RetroBridge confidence = sample count / group size.

    Equal-frequency candidates retain first occurrence; no reference labels
    or executor outcomes enter the ranking.
    """
    if not samples or top_k <= 0:
        raise ValueError("rank_samples requires samples and positive Top-K")
    first: dict[str, int] = {}
    for index, sample in enumerate(samples):
        first.setdefault(sample, index)
    counts = Counter(samples)
    ordered = sorted(counts, key=lambda item: (-counts[item], first[item]))
    return [{"rank": rank, "precursors": precursor,
             "sample_count": counts[precursor],
             "frequency_confidence": counts[precursor] / len(samples),
             "first_sample_index": first[precursor]}
            for rank, precursor in enumerate(ordered[:top_k], 1)]


def load_queries(path: Path) -> list[str]:
    products: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        product = row["product_smiles"]
        if row["model_input"] != {"product_smiles": product}:
            raise ValueError("R5 query is not product-only input")
        product_key(product)  # Chemistry validation; preserve the frozen spelling.
        products.append(product)
    if len(products) != 200 or len({product_key(item) for item in products}) != 200:
        raise ValueError("R5 requires the fixed 200 unique products")
    return products


def official_train_overlap(path: Path, products: list[str]) -> dict[str, bool]:
    train_products: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "reactants>reagents>production" not in (reader.fieldnames or []):
            raise ValueError("Unexpected official RetroBridge training CSV schema")
        for row in reader:
            reaction = row["reactants>reagents>production"]
            train_products.add(product_key(reaction.rsplit(">>", 1)[1]))
    return {product: product_key(product) in train_products for product in products}


def _load_official(source: Path):
    source_resolved = source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source_resolved), "rev-parse", "HEAD"], text=True,
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"RetroBridge source commit mismatch: {commit}")
    sys.path.insert(0, str(source_resolved))
    from src.analysis.rdkit_functions import build_molecule
    from src.data.retrobridge_dataset import RetroBridgeDataset, RetroBridgeDatasetInfos
    from src.frameworks.markov_bridge import MarkovBridge
    from src.utils import set_deterministic
    return build_molecule, RetroBridgeDataset, RetroBridgeDatasetInfos, MarkovBridge, set_deterministic


def _sample_product(product: str, *, seed: int, n_samples: int, model: Any,
                    official: tuple[Any, ...], device: str) -> list[str]:
    import torch
    from torch_geometric.data import Data
    from rdkit import Chem

    build_molecule, dataset_class, dataset_info, _, set_deterministic = official
    set_deterministic(seed)
    molecule = Chem.MolFromSmiles(product)
    if molecule is None:
        raise ValueError("invalid product SMILES")
    unsupported = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()}
                         - set(dataset_class.types))
    if unsupported:
        raise ValueError("unsupported_atom_type:" + ",".join(unsupported))
    mapping = {}
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        atom.SetAtomMapNum(index)
        mapping[index] = index
    node_count = molecule.GetNumAtoms() + dataset_info.max_n_dummy_nodes
    x, edge_index, edge_attr = dataset_class.compute_graph(
        molecule, mapping, node_count, dataset_class.types, dataset_class.bonds)
    x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)
    dataset, batches = [], []
    offset = 0
    for index in range(n_samples):
        data = Data(idx=index, p_x=x, p_edge_index=edge_index.clone(),
                    p_edge_attr=edge_attr, p_smiles=product)
        data.p_edge_index += offset
        dataset.append(data)
        batches.append(torch.ones_like(data.p_x[:, 0], dtype=torch.long) * index)
        offset += len(data.p_x)
    batch_data, _ = dataset_class.collate(dataset)
    batch_data.batch = torch.cat(batches)
    with torch.inference_mode():
        _, _, _, _, predicted, _, _, _ = model.sample_chain(
            batch_data, batch_size=n_samples, keep_chain=0,
            number_chain_steps_to_save=1, save_true_reactants=False)
    samples = []
    for graph in predicted:
        result, _ = build_molecule(graph[0], graph[1], model.dataset_info.atom_decoder,
                                   return_n_dummy_atoms=True)
        smiles = Chem.MolToSmiles(result)
        try:
            smiles = product_key(smiles)
        except ValueError:
            pass  # Retain the raw, invalid candidate for intake accounting.
        samples.append(smiles)
    return samples


def run(args: argparse.Namespace) -> dict[str, Any]:
    from rdkit import Chem

    query_manifest = json.loads(args.query_manifest.read_text())
    if digest(args.query) != query_manifest["cohort_sha256"]:
        raise ValueError("Frozen R5 query checksum mismatch")
    products = load_queries(args.query)
    if digest(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Official RetroBridge checkpoint checksum mismatch")
    official = _load_official(args.source)
    overlap = official_train_overlap(args.source / "datasets/uspto50k_train.csv", products)
    train_csv_hash = digest(args.source / "datasets/uspto50k_train.csv")
    count = len(products) if args.limit is None else min(args.limit, len(products))
    run_config = {"query_sha256": digest(args.query), "source_commit": SOURCE_COMMIT,
                  "checkpoint_sha256": CHECKPOINT_SHA256, "samples": args.samples,
                  "steps": args.steps, "seed": args.seed, "device": args.device,
                  "product_count": count}
    predictions = args.output / "raw_predictions.jsonl"
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {args.output}")
        if json.loads((args.output / "run_config.json").read_text()) != run_config:
            raise ValueError("R5 resume settings differ from the frozen run")
        existing = ([json.loads(line) for line in predictions.read_text().splitlines()]
                    if predictions.exists() else [])
        if [row["product_smiles"] for row in existing] != products[:len(existing)]:
            raise ValueError("R5 resume predictions are not an exact query prefix")
        if len(existing) > count:
            raise ValueError("R5 resume has excess rows")
        start = len(existing)
    else:
        args.output.mkdir(parents=True)
        (args.output / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n")
        start = 0
    model = official[3].load_from_checkpoint(str(args.checkpoint), map_location=args.device).to(args.device)
    model.T = args.steps
    model.eval()
    with predictions.open("a" if args.resume else "w", encoding="utf-8", buffering=1) as stream:
        for index in range(start, count):
            product = products[index]
            unsupported = sorted({atom.GetSymbol() for atom in Chem.MolFromSmiles(product).GetAtoms()}
                                 - set(official[1].types))
            if unsupported:
                row = {"product_smiles": product, "inference_status": "failed",
                       "failure_reason": "unsupported_atom_type:" + ",".join(unsupported),
                       "candidates": []}
            else:
                try:
                    samples = _sample_product(
                        product, seed=args.seed + index, n_samples=args.samples,
                        model=model, official=official, device=args.device)
                    row = {"product_smiles": product, "inference_status": "completed",
                           "candidates": rank_samples(samples, top_k=5),
                           "raw_sample_count": len(samples)}
                except (ValueError, RuntimeError, KeyError, IndexError) as error:
                    if "out of memory" in str(error).lower():
                        raise
                    row = {"product_smiles": product, "inference_status": "failed",
                           "failure_reason": f"{type(error).__name__}:{error}", "candidates": []}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"R5 RetroBridge {index + 1}/{count} {row['inference_status']}", flush=True)
    report = {"artifact_type": "r5_retrobridge_official_generation_v1",
              "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
              "checkpoint_url": CHECKPOINT_URL, "checkpoint_sha256": CHECKPOINT_SHA256,
              "official_train_csv_sha256": train_csv_hash,
              "frozen_query_sha256": digest(args.query), "products_expected": len(products),
              "products_generated": count, "training_exact_product_overlap_count": sum(overlap.values()),
              "training_exact_product_overlap": overlap, "sampling_steps": args.steps,
              "samples_per_product": args.samples, "seed": args.seed,
              "ranking_semantics": "official empirical sample frequency; first occurrence breaks ties",
              "model_input": "product SMILES only", "target_semantics": "retrosynthetic_precursor_set",
              "raw_predictions_sha256": digest(predictions),
              "claim_boundary": "Diagnostic external source; exact-product overlap and element support prohibit a leakage-clean 200-product headline."}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    provenance = {"model_name": "RetroBridge", "checkpoint_identifier": CHECKPOINT_SHA256,
                  "checkpoint_source": CHECKPOINT_URL, "training_corpus": "USPTO-50K",
                  "license_or_terms": "CC-BY-NC-4.0 (Zenodo record)",
                  "input_fields": ["product_smiles"],
                  "target_semantics": "retrosynthetic_precursor_set",
                  "inference_status": "completed" if count == len(products) else "incomplete",
                  "inference_config": {"samples_per_product": args.samples, "steps": args.steps,
                                       "seed": args.seed, "device": args.device,
                                       "ranking": report["ranking_semantics"]},
                  "training_overlap_audited": True,
                  "training_exact_product_overlap_count": sum(overlap.values()),
                  "training_exact_product_overlap": overlap,
                  "source_commit": SOURCE_COMMIT, "source_train_csv_sha256": train_csv_hash}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="Engineering smoke only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 5 or args.steps < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Requires >=5 samples, positive steps and positive limit")
    report = run(args)
    print(json.dumps({key: value for key, value in report.items()
                      if key != "training_exact_product_overlap"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
