#!/usr/bin/env python3
"""Train-only fragment-batch retrieval diagnostic for System-One IMPORT.

The index is made exclusively from training IMPORT states. Held-out reference
current states are used only as queries, so this is not a product-start result
or an open-vocabulary fragment generator. It measures whether a cheap,
gold-independent proposal source can support the next closed-loop pilot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from mechet.structural_overlap import canonical_unmapped_smiles
from scripts.audit_system_one_import_space import normalize_batch
from scripts.train_system_one_electron_flow import verify_source


CURRENT_RE = re.compile(r"^CURRENT STATE SMILES: (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class ImportExample:
    row_id: str
    reaction_id: str
    target: str
    current: str
    batch: tuple[tuple[str, int], ...]


def chemical_batch(arguments: dict) -> tuple[tuple[str, int], ...]:
    """Compare actual imported molecules/counts, not annotation-only purposes."""
    molecules: Counter[str] = Counter()
    for smiles, count, _purpose in normalize_batch(arguments):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid import fragment: {smiles}")
        molecules[Chem.MolToSmiles(molecule, isomericSmiles=True)] += count
    return tuple(sorted(molecules.items()))


def append_import_batch(current: str, batch: tuple[tuple[str, int], ...]) -> str:
    """Append proposed molecules as disjoint components; never alter old atoms."""
    if not batch:
        raise ValueError("empty import batch")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    combined = Chem.MolFromSmiles(current, params)
    if combined is None:
        raise ValueError("invalid current-state SMILES")
    for smiles, count in batch:
        fragment = Chem.MolFromSmiles(smiles, params)
        if fragment is None or count < 1:
            raise ValueError("invalid proposed import fragment/count")
        for _ in range(count):
            combined = Chem.CombineMols(combined, fragment)
    return Chem.MolToSmiles(combined, canonical=True, isomericSmiles=True)


def load_imports(path: Path) -> tuple[list[ImportExample], dict]:
    source = verify_source(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    target_is_principal = (
        manifest.get("artifact_type")
        == "mech_uspto_31k_natural_language_history_principal_target_v2"
    )
    declared_imports = int(manifest["splits"][path.stem]["import_decisions"])
    examples = []
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank source row in {path}")
            rows += 1
            row = json.loads(line)
            if row["metadata"]["decision_type"] != "import":
                continue
            row_id = str(row["id"])
            users = [message for message in row["messages"] if message.get("role") == "user"]
            calls = [call for message in row["messages"] if message.get("role") == "assistant"
                     for call in message.get("tool_calls", ())]
            if len(users) != 1 or len(calls) != 1 or calls[0]["function"]["name"] != "import_fragments":
                raise ValueError(f"{row_id}: invalid IMPORT decision record")
            match = CURRENT_RE.search(str(users[0]["content"]))
            if match is None:
                raise ValueError(f"{row_id}: missing current-state SMILES")
            tools = [message for message in row["messages"] if message.get("role") == "tool"]
            if len(tools) != 1:
                raise ValueError(f"{row_id}: missing authoritative IMPORT result")
            result = json.loads(str(tools[0]["content"]))
            if result.get("ok") is not True or tools[0].get("name") != "import_fragments":
                raise ValueError(f"{row_id}: IMPORT did not execute successfully")
            batch = chemical_batch(calls[0]["function"]["arguments"])
            if canonical_unmapped_smiles(append_import_batch(match.group(1), batch)) != (
                canonical_unmapped_smiles(str(result["current_state"]))
            ):
                raise ValueError(f"{row_id}: IMPORT composition differs from executor")
            if target_is_principal and not row.get("principal_product_smiles"):
                raise ValueError(f"{row_id}: missing principal-product target")
            examples.append(ImportExample(
                row_id=row_id,
                reaction_id=str(row["metadata"]["reaction_id"]),
                target=str(row["principal_product_smiles"] if target_is_principal
                           else row["target_smiles"]),
                current=match.group(1),
                batch=batch,
            ))
    if rows != source["decision_rows"] or len(examples) != declared_imports:
        raise ValueError(f"{path}: IMPORT decision denominator mismatch")
    return examples, {**source, "import_decisions": declared_imports,
                      "import_target_mode": "principal_product" if target_is_principal else "mixture",
                      "import_replay_ok": len(examples)}


def fingerprint(smiles: str, generator):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid state/product SMILES: {smiles}")
    return generator.GetFingerprint(molecule)


def rank_unique_batches(
    query_target_fp,
    query_current_fp,
    train: list[ImportExample],
    target_fps: list,
    current_fps: list,
    *,
    limit: int,
) -> list[tuple[tuple[str, int], ...]]:
    """Score all training states; return distinct chemistry batches, no GT input."""
    target_sim = DataStructs.BulkTanimotoSimilarity(query_target_fp, target_fps)
    current_sim = DataStructs.BulkTanimotoSimilarity(query_current_fp, current_fps)
    order = sorted(
        range(len(train)),
        key=lambda index: (-(target_sim[index] + current_sim[index]), index),
    )
    proposals = []
    seen = set()
    for index in order:
        batch = train[index].batch
        if batch not in seen:
            proposals.append(batch)
            seen.add(batch)
            if len(proposals) == limit:
                break
    return proposals


def evaluate(train: list[ImportExample], heldout: list[ImportExample], *, topk: int) -> tuple[dict, list[dict]]:
    if not train or not heldout or topk < 1:
        raise ValueError("retrieval requires nonempty splits and positive top-k")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target_fps = [fingerprint(row.target, generator) for row in train]
    current_fps = [fingerprint(row.current, generator) for row in train]
    target_keys = {Chem.MolToSmiles(Chem.MolFromSmiles(row.target), isomericSmiles=True)
                   for row in train}
    observed_batches = {row.batch for row in train}
    majority_batch = Counter(row.batch for row in train).most_common(1)[0][0]
    counts = Counter()
    cases = []
    for row in heldout:
        candidates = rank_unique_batches(
            fingerprint(row.target, generator), fingerprint(row.current, generator),
            train, target_fps, current_fps, limit=topk,
        )
        target_key = Chem.MolToSmiles(Chem.MolFromSmiles(row.target), isomericSmiles=True)
        overlap = target_key in target_keys
        seen = row.batch in observed_batches
        counts["n"] += 1
        counts["gold_batch_seen_in_train"] += int(seen)
        counts["exact_target_seen_in_train"] += int(overlap)
        counts["train_majority_hit"] += int(row.batch == majority_batch)
        for k in (1, 2, 4, topk):
            if k <= topk:
                counts[f"hit_at_{k}"] += int(row.batch in candidates[:k])
                if not overlap:
                    counts[f"no_exact_target_hit_at_{k}"] += int(row.batch in candidates[:k])
        counts["no_exact_target_n"] += int(not overlap)
        cases.append({
            "id": row.row_id,
            "reaction_id": row.reaction_id,
            "exact_target_seen_in_train": overlap,
            "gold_batch_seen_in_train": seen,
            "gold_batch": row.batch,
            "proposed_batches": candidates,
        })
    summary = dict(counts)
    summary["train_majority_rate"] = counts["train_majority_hit"] / counts["n"]
    for k in (1, 2, 4, topk):
        if k <= topk:
            summary[f"recall_at_{k}"] = counts[f"hit_at_{k}"] / counts["n"]
            summary[f"no_exact_target_recall_at_{k}"] = (
                counts[f"no_exact_target_hit_at_{k}"] / counts["no_exact_target_n"]
                if counts["no_exact_target_n"] else None
            )
    return summary, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    status = json.loads((args.data_dir / "ARTIFACT_STATUS.json").read_text())
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if (status.get("artifact_id") != manifest.get("artifact_type")
            or not status.get("training_allowed") or not manifest.get("training_allowed")):
        raise ValueError("source is not a validated train-ready trace view")
    train, train_source = load_imports(args.data_dir / "train.jsonl")
    reports = {}
    cases = {}
    sources = {"train": train_source}
    for split in ("valid", "test"):
        heldout, source = load_imports(args.data_dir / f"{split}.jsonl")
        summary, split_cases = evaluate(train, heldout, topk=args.topk)
        reports[split] = summary
        cases[split] = split_cases
        sources[split] = source
        print(json.dumps({"phase": "retrieval_done", "split": split,
                          "summary": summary}), flush=True)
    args.output.mkdir(parents=True)
    report = {
        "artifact_type": "system_one_train_only_import_retrieval_diagnostic",
        "scope": "reference_current_states_not_product_start_or_open_vocabulary_generation",
        "source_artifact": manifest["artifact_type"],
        "ranker": "Morgan-radius2-2048-mean-target/current-Tanimoto-nearest-train-state",
        "topk": args.topk,
        "sources": sources,
        "train_import_decisions": len(train),
        "train_distinct_chemical_batches": len({row.batch for row in train}),
        "splits": reports,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for split, split_cases in cases.items():
        with (args.output / f"{split}.cases.jsonl").open("w") as handle:
            for row in split_cases:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"phase": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
