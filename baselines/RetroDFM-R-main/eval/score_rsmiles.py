import argparse
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def extract_answer(text):
    left = text.find("<answer>")
    right = text.find("</answer>")
    if left == -1 or right <= left:
        return ""
    return text[left + len("<answer>") : right].strip()


def canonicalize(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "", ""
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        full = Chem.MolToSmiles(molecule, isomericSmiles=True)
    except Exception:
        return "", ""
    fragments = [(fragment, Chem.MolFromSmiles(fragment)) for fragment in full.split(".")]
    fragments = [(fragment, mol.GetNumAtoms()) for fragment, mol in fragments if mol is not None]
    largest = max(fragments, key=lambda item: item[1])[0] if fragments else ""
    return full, largest


def rank_predictions(predictions, alpha, beam_size):
    scores = {}
    invalid = [0] * beam_size
    for augmentation in predictions:
        for rank, prediction in enumerate(augmentation):
            if not prediction[0]:
                if rank < beam_size:
                    invalid[rank] += 1
                continue
            scores[prediction] = scores.get(prediction, 0.0) + 1.0 / (alpha * rank + 1.0)
    return sorted(scores, key=scores.get, reverse=True), invalid


def score(items, augmentation, beam_size, n_best, alpha):
    data_size = len(items) // augmentation
    items = items[: data_size * augmentation]
    if data_size == 0:
        raise ValueError("no complete augmentation group to score")
    grouped = []
    labels = []
    for index in range(data_size):
        predictions = []
        for aug_index in range(augmentation):
            row = items[index * augmentation + aug_index]
            predictions.append([canonicalize(extract_answer(text)) for text in row["output"]])
        grouped.append(predictions)
        labels.append(canonicalize(items[index * augmentation]["label"]["reactants"]))

    topk = [0] * n_best
    largest_fragment_topk = [0] * n_best
    invalid = [0] * beam_size
    ranked_rows = []
    unique_total = 0

    for index, predictions in enumerate(tqdm(grouped, desc="score")):
        ranked, row_invalid = rank_predictions(predictions, alpha, beam_size)
        invalid = [left + right for left, right in zip(invalid, row_invalid)]
        ranked = ranked[:n_best]
        unique_total += len(ranked)
        ranked_rows.append([prediction[0] for prediction in ranked])

        for rank, prediction in enumerate(ranked):
            if prediction[0] == labels[index][0]:
                for k in range(rank, n_best):
                    topk[k] += 1
                break
        for rank, prediction in enumerate(ranked):
            if prediction[1] == labels[index][1]:
                for k in range(rank, n_best):
                    largest_fragment_topk[k] += 1
                break

    for k in range(n_best):
        if k < 10 or k in (19, 49):
            print(
                f"Top-{k + 1} Acc: {topk[k] / data_size * 100:.3f}%, "
                f"MaxFrag: {largest_fragment_topk[k] / data_size * 100:.3f}%"
            )
    print(f"Unique rate: {unique_total / data_size / beam_size * 100:.3f}%")
    print(
        "Invalid rates: "
        + ",".join(f"{value / data_size / augmentation * 100:.3f}" for value in invalid)
    )
    return items, labels, ranked_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--augmentation", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--n_best", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    with (output_dir / f"{args.task}.jsonl").open() as source:
        items = [json.loads(line) for line in source if line.strip()]
    items, labels, ranked = score(
        items, args.augmentation, args.beam_size, args.n_best, args.alpha
    )

    with (output_dir / f"{args.task}_pred.jsonl").open("w") as output:
        for index, row in enumerate(ranked):
            source = items[index * args.augmentation]
            output.write(
                json.dumps(
                    {"input": source["input"], "label": labels[index][0], "ranked_result": row},
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
