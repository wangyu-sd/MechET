import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from rdkit import Chem
from torch import distributed as dist
from tqdm import tqdm

from retflow import config


def process_data_compute_metrics(
    output_save_path: Path, samples_per_product: int, compute_round_trip=False
):
    csv_path = output_save_path.parent / f"processed_{output_save_path.stem}.csv"

    if not csv_path.is_file():
        table = make_table_from_processed(output_save_path, samples_per_product)
        config.get_logger().info(f"Finished processing. Saving csv file to: {csv_path}")
        table.to_csv(csv_path, index=False)
    else:
        config.get_logger().info(
            f"Processed csv file already exists: {csv_path}. Computing metrics from that data."
        )
        table = pd.read_csv(csv_path)

    for key in ["product"]:
        table[key] = table[key].apply(canonicalize)

    if compute_round_trip and "pred_product" not in table.columns:
        table = compute_forward_predictions(table)
        table.to_csv(csv_path, index=False)

    config.get_logger().info("Computing metrics...")
    df = assign_groups(table, samples_per_product)
    df.loc[(df["product"] == "C") & (df["true"] == "C"), "true"] = "Placeholder"

    df_processed = compute_confidence(df)

    # if compute_round_trip:
    #     df_processed = compute_forward_predictions(df_processed)

    compute_accuracy(
        df_processed,
        top=[1, 3, 5, 10],
        scoring=lambda df: np.log(df["confidence"]),
        verbose=True,
    )


def make_table_from_processed(output_save_path: Path, samples_per_product: int):
    config.get_logger().info(f"Loading data from file: {output_save_path}.")
    with open(output_save_path, "rb") as file:
        output_data = pickle.load(file)

    config.get_logger().info("Finished loading data.")

    true_molecules_smiles = []
    pred_molecules_smiles = []
    product_molecules_smiles = []
    computed_scores = []
    synthons_smiles = []
    stable_ids_flat = []
    preprocess_errors = []

    reactants = output_data["reactants"]
    products = output_data["products"]
    predicted_reactants = output_data["predicted_reactants"]
    scores = output_data["scores"]
    synthons = None if "synthons" not in output_data else output_data["synthons"]
    stable_ids = output_data.get(
        "stable_ids", [f"processed:{index}" for index in range(len(products))]
    )
    preprocessing_failures = output_data.get("preprocessing_failures", [])

    config.get_logger().info(
        "Processing data to compute Top-K accuracy metric. This might take a while..."
    )

    for i in tqdm(range(len(products))):
        true_smi = reactants[i]
        product_smi = products[i]

        for pred_smi, pred_score, synthon_smi in zip(
            predicted_reactants[i],
            scores[i],
            (
                synthons[i]
                if synthons is not None
                else [None] * len(predicted_reactants[i])
            ),
        ):
            true_molecules_smiles.append(true_smi)
            product_molecules_smiles.append(product_smi)
            pred_molecules_smiles.append(pred_smi)
            computed_scores.append(pred_score)
            stable_ids_flat.append(stable_ids[i])
            preprocess_errors.append(False)
            if synthons is not None:
                synthons_smiles.append(synthon_smi)
    for failure in preprocessing_failures:
        for _ in range(samples_per_product):
            true_molecules_smiles.append(
                failure.get("reference_precursors", "__PREPROCESS_ERROR_REFERENCE__")
            )
            product_molecules_smiles.append(failure.get("product", ""))
            pred_molecules_smiles.append("__PREPROCESS_ERROR__")
            computed_scores.append(float("-inf"))
            stable_ids_flat.append(failure["stable_id"])
            preprocess_errors.append(True)
            if synthons is not None:
                synthons_smiles.append("")

    expected_rows = output_data.get(
        "preprocessing_input_rows", len(products) + len(preprocessing_failures)
    )
    if len(products) + len(preprocessing_failures) != expected_rows:
        raise RuntimeError(
            "processed predictions and preprocessing failures do not cover "
            f"the full evaluation denominator ({expected_rows})"
        )
    config.get_logger().info(
        "Evaluation denominator: %d rows (%d predictions, %d preprocessing errors).",
        expected_rows,
        len(products),
        len(preprocessing_failures),
    )
    if synthons is not None:
        return pd.DataFrame(
            {
                "product": product_molecules_smiles,
                "pred": pred_molecules_smiles,
                "true": true_molecules_smiles,
                "synthons": synthons_smiles,
                "score": computed_scores,
                "stable_id": stable_ids_flat,
                "preprocess_error": preprocess_errors,
            }
        )
    else:
        return pd.DataFrame(
            {
                "product": product_molecules_smiles,
                "pred": pred_molecules_smiles,
                "true": true_molecules_smiles,
                "score": computed_scores,
                "stable_id": stable_ids_flat,
                "preprocess_error": preprocess_errors,
            }
        )


def canonicalize(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return np.nan
    return Chem.MolToSmiles(m)


def assign_groups(df, samples_per_product):
    df["group"] = np.arange(len(df)) // samples_per_product
    return df


def compute_confidence(df):
    counts = df.groupby(["group", "pred"]).size().reset_index(name="count")
    group_size = df.groupby(["group"]).size().reset_index(name="group_size")

    #     Don't use .merge() as it can change the order of rows
    #     df = df.merge(counts, on=['group', 'pred'], how='left')
    #     df = df.merge(counts, on=['group', 'pred'], how='inner')
    #     df = df.merge(group_size, on=['group'], how='left')

    counts_dict = {
        (g, p): c for g, p, c in zip(counts["group"], counts["pred"], counts["count"])
    }
    df["count"] = df.apply(lambda x: counts_dict[(x["group"], x["pred"])], axis=1)

    size_dict = {g: s for g, s in zip(group_size["group"], group_size["group_size"])}
    df["group_size"] = df.apply(lambda x: size_dict[x["group"]], axis=1)

    df["confidence"] = df["count"] / df["group_size"]

    # sanity check
    assert (df.groupby(["group", "pred"])["confidence"].nunique() == 1).all()
    assert (df.groupby(["group"])["group_size"].nunique() == 1).all()

    return df


def get_top_k(df, k, scoring=None):
    if callable(scoring):
        df["_new_score"] = scoring(df)
        scoring = "_new_score"

    if scoring is not None:
        df = df.sort_values(by=scoring, ascending=False)
    df = df.drop_duplicates(subset="pred")

    return df.head(k)


def compute_accuracy(df, top=[1, 3, 5], scoring=None, verbose=False):
    round_trip = "pred_product" in df.columns

    results = {}
    results["Exact match"] = {}

    df["exact_match"] = df["true"] == df["pred"]

    if round_trip:
        results["Round-trip coverage"] = {}
        results["Round-trip accuracy"] = {}

        df["round_trip_match"] = df["product"] == df["pred_product"]
        df["match"] = df["exact_match"] | df["round_trip_match"]

    for k in top:
        topk_df = (
            df.groupby(["group"])
            .apply(partial(get_top_k, k=k, scoring=scoring))
            .reset_index(drop=True)
        )

        acc_exact_match = topk_df.groupby("group").exact_match.any().mean()
        results["Exact match"][f"top-{k}"] = acc_exact_match
        if verbose:
            config.get_logger().info(
                f"Top-{k} exact match accuracy is: {acc_exact_match * 100}"
            )

        if round_trip:
            cov_round_trip = topk_df.groupby("group").match.any().mean()
            acc_round_trip = topk_df.groupby("group").match.mean().mean()
            results["Round-trip coverage"][f"top-{k}"] = cov_round_trip
            results["Round-trip accuracy"][f"top-{k}"] = acc_round_trip

            if verbose:
                config.get_logger().info(
                    f"Top-{k} round trip accuracy is: {acc_round_trip * 100}"
                )
                config.get_logger().info(
                    f"Top-{k} round trip coverage is: {cov_round_trip * 100}"
                )

    return pd.DataFrame(results).T


def compute_forward_predictions(df):
    # OpenNMT is only needed for the optional round-trip metric. Keep the
    # dependency out of exact-match evaluation and preprocessing-error tests.
    from retflow.utils.forward_model import get_forward_model, smi_tokenizer

    config.get_logger().info("Computing forward predictions...")
    translator = get_forward_model(5)

    unique_smiles = list(set(df["pred"]))

    # Tokenize
    tokenized_smiles = [smi_tokenizer(s.strip()) for s in unique_smiles]

    _, pred_products = translator.translate(
        src_data_iter=tokenized_smiles,
        batch_size=128,
        attn_debug=False,
    )

    pred_products = [x[0].strip() for x in pred_products]
    # De-tokenize
    pred_products = ["".join(x.split()) for x in pred_products]

    # gather results
    pred_products = {r: p for r, p in zip(unique_smiles, pred_products)}

    # update dataframe
    df["pred_product"] = [pred_products[r] for r in df["pred"]]
    return df


def top_k_accuracy(
    grouped_samples, ground_truth, ddp: bool = False
):
    """
    Compute top-N accuracy. Inputs are matrices of atom types and bonds.
    """
    top_k_success = {k: 0 for k in [1, 3, 5, 10, 50]}

    total = 0

    for i, sampled_reactants in enumerate(grouped_samples):
        true_reactants = ground_truth[i]
        true_smi = Chem.MolToSmiles(true_reactants, canonical=True)
        if true_smi is None:
            continue

        sampled_smis = []
        for sample in sampled_reactants:
            sampled_smi = Chem.MolToSmiles(sample, canonical=True)
            if sampled_smi is None:
                continue
            sampled_smis.append(sampled_smi)

        total += 1
        for k in top_k_success.keys():
            top_k_success[k] += true_smi in set(sampled_smis[:k])


    if ddp:
        dist.barrier()
        total = torch.tensor(total).cuda()
        dist.all_reduce(total)
        total = total.item()

        for k in top_k_success.keys():
            tk = torch.tensor(top_k_success[k]).cuda()
            dist.all_reduce(tk)
            top_k_success[k] = tk.item()

    metrics = {}
    for k in top_k_success.keys():
        metrics[f"top_{k}_accuracy"] = top_k_success[k] / total

    return metrics
