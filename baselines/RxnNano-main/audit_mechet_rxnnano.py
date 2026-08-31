#!/usr/bin/env python3
"""Audit MechET-to-RxnNano preprocessing and prepare overfit slices."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

from src.train.prompts import format_retrosynthesis_prompt


ROOT = Path(__file__).resolve().parent
MECHET_ROOT = ROOT.parents[1] / "MechET"
DATASETS = {
    "flower_full": {
        "suffix": "flower_full",
        "reference": MECHET_ROOT / "data" / "external_baselines" / "flower_full",
    },
    "mech_uspto_31k_full": {
        "suffix": "mech_uspto_31k_full",
        "reference": MECHET_ROOT
        / "data"
        / "external_baselines"
        / "mech_uspto_31k_full",
    },
}
SPLIT_FILES = {"train": "train", "valid": "validation", "test": "test"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def source_snapshot_sha256() -> str:
    """Hash the runnable source snapshot when the copied repository lacks .git."""
    files = [ROOT / "train.py", ROOT / "evaluate.py", ROOT / "requirements.txt"]
    files.extend(sorted((ROOT / "src").rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def molecule(smiles: str) -> Chem.Mol | None:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(str(smiles or "").strip(), params)


def canonical(smiles: str, *, keep_maps: bool) -> str | None:
    mol = molecule(smiles)
    if mol is None:
        return None
    if not keep_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return ".".join(
        sorted(
            Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
            for fragment in fragments
        )
    )


def mapping_coverage(smiles: str) -> tuple[int, int]:
    mol = molecule(smiles)
    if mol is None:
        return 0, 0
    atoms = mol.GetNumAtoms()
    mapped = sum(atom.GetAtomMapNum() > 0 for atom in mol.GetAtoms())
    return mapped, atoms


def extract_answer(assistant_text: str) -> dict[str, Any] | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", assistant_text, re.DOTALL)
    if not match:
        return None
    try:
        answer = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) else None


def adapter_path(dataset: str, split: str) -> Path:
    suffix = DATASETS[dataset]["suffix"]
    return ROOT / "data" / f"{SPLIT_FILES[split]}_{suffix}.jsonl"


def reference_path(dataset: str, split: str) -> Path:
    return DATASETS[dataset]["reference"] / f"{split}.jsonl"


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def preprocess_audit(args: argparse.Namespace) -> int:
    source = adapter_path(args.dataset, args.split)
    reference = reference_path(args.dataset, args.split)
    source_rows = read_jsonl(source)
    reference_rows = read_jsonl(reference)
    selected = source_rows[: args.size]
    if len(selected) != args.size:
        raise ValueError(f"requested {args.size} rows but {source} has {len(selected)}")

    reference_by_id = {str(row["stable_id"]): row for row in reference_rows}
    other_ids = set()
    if args.split == "train":
        for split in ("valid", "test"):
            other_ids.update(
                str(row["stable_id"])
                for row in read_jsonl(reference_path(args.dataset, split))
            )

    failures: dict[str, list[str]] = {
        "missing_reference_id": [],
        "duplicate_id": [],
        "split_leakage": [],
        "invalid_product": [],
        "invalid_precursors": [],
        "product_identity_mismatch": [],
        "precursor_identity_mismatch": [],
        "product_unmapped_identity_mismatch": [],
        "precursor_unmapped_identity_mismatch": [],
        "product_not_fully_mapped": [],
        "precursor_not_fully_mapped": [],
        "formatted_id_lost": [],
        "formatted_product_mismatch": [],
        "formatted_precursor_mismatch": [],
    }
    seen: set[str] = set()
    formatted_rows = []

    for row in selected:
        stable_id = str(row.get("stable_id") or "")
        if stable_id in seen:
            failures["duplicate_id"].append(stable_id)
        seen.add(stable_id)
        if stable_id in other_ids:
            failures["split_leakage"].append(stable_id)

        ref = reference_by_id.get(stable_id)
        if ref is None:
            failures["missing_reference_id"].append(stable_id)
            continue

        product = str(row.get("product") or "")
        precursors = str(row.get("reactants") or "")
        if molecule(product) is None:
            failures["invalid_product"].append(stable_id)
        if molecule(precursors) is None:
            failures["invalid_precursors"].append(stable_id)

        ref_product = str(ref.get("product_mapped") or "")
        ref_precursors = str(ref.get("precursor_mapped") or "")
        if canonical(product, keep_maps=True) != canonical(
            ref_product, keep_maps=True
        ):
            failures["product_identity_mismatch"].append(stable_id)
        if canonical(precursors, keep_maps=True) != canonical(
            ref_precursors, keep_maps=True
        ):
            failures["precursor_identity_mismatch"].append(stable_id)
        if canonical(product, keep_maps=False) != canonical(
            str(ref.get("product_unmapped") or ref_product), keep_maps=False
        ):
            failures["product_unmapped_identity_mismatch"].append(stable_id)
        if canonical(precursors, keep_maps=False) != canonical(
            str(ref.get("precursor_unmapped") or ref_precursors), keep_maps=False
        ):
            failures["precursor_unmapped_identity_mismatch"].append(stable_id)

        mapped, atoms = mapping_coverage(product)
        if atoms == 0 or mapped != atoms:
            failures["product_not_fully_mapped"].append(stable_id)
        mapped, atoms = mapping_coverage(precursors)
        if atoms == 0 or mapped != atoms:
            failures["precursor_not_fully_mapped"].append(stable_id)

        example = dict(row)
        example.update({"variant": "mapped", "prompt_style": args.prompt_style})
        formatted = {**example, **format_retrosynthesis_prompt(example)}
        if formatted.get("stable_id") != stable_id:
            failures["formatted_id_lost"].append(stable_id)
        conversations = formatted["conversations"]
        if f'"{product}"' not in conversations[1]["content"]:
            failures["formatted_product_mismatch"].append(stable_id)
        answer = extract_answer(conversations[2]["content"])
        if answer is None or answer.get("reactants") != precursors:
            failures["formatted_precursor_mismatch"].append(stable_id)

        formatted_rows.append(
            {
                "stable_id": stable_id,
                "source_split": row.get("source_split"),
                "variant": "mapped",
                "product": product,
                "reactants": precursors,
                "conversations": conversations,
            }
        )

    nonzero_failures = {key: value for key, value in failures.items() if value}
    report = {
        "schema_version": 1,
        "artifact_type": "rxnnano_preprocessing_audit",
        "dataset": args.dataset,
        "split": args.split,
        "size": args.size,
        "variant": "mapped",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "reference": str(reference),
        "reference_sha256": sha256_file(reference),
        "selected_stable_ids_sha256": sha256_ids(
            [str(row.get("stable_id") or "") for row in selected]
        ),
        "external_repo_commit": git_commit(ROOT),
        "external_repo_source_snapshot_sha256": source_snapshot_sha256(),
        "environment_lockfile": str(ROOT / "requirements.txt"),
        "environment_lockfile_sha256": sha256_file(ROOT / "requirements.txt"),
        "train_derived_vocab_or_templates": [],
        "formatted_rows": len(formatted_rows),
        "failure_counts": {key: len(value) for key, value in failures.items()},
        "failure_ids": nonzero_failures,
        "passed": not nonzero_failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formatted_path = args.output_dir / "preprocessed.jsonl"
    report_path = args.output_dir / "preprocess_audit.json"
    write_jsonl(formatted_path, formatted_rows)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


def prepare_overfit(args: argparse.Namespace) -> int:
    source = adapter_path(args.dataset, "train")
    rows = read_jsonl(source)[: args.size]
    if len(rows) != args.size:
        raise ValueError(f"requested {args.size} rows but {source} has {len(rows)}")
    stable_ids = [str(row["stable_id"]) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "validation", "test"):
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "schema_version": 1,
        "artifact_type": "rxnnano_overfit_slice",
        "dataset": args.dataset,
        "source_split": "train",
        "size": args.size,
        "variant": "mapped",
        "stable_ids_sha256": sha256_ids(stable_ids),
        "stable_ids": stable_ids,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "files": paths,
        "note": "Validation and test intentionally reuse the training prefix for the overfit gate.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def audit_overfit(args: argparse.Namespace) -> int:
    """Audit loss movement, shared prediction schema, IDs, and candidate validity."""
    manifest_path = args.slice_dir / "manifest.json"
    test_path = args.slice_dir / "test.jsonl"
    training_metrics_path = args.run_dir / "training_metrics.json"
    inference_metrics_path = args.run_dir / "inference_metrics.json"
    predictions_path = args.run_dir / "predictions.jsonl"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    inference_metrics = json.loads(inference_metrics_path.read_text(encoding="utf-8"))
    expected_rows = read_jsonl(test_path)
    prediction_rows = read_jsonl(predictions_path)

    expected_by_id = {str(row["stable_id"]): row for row in expected_rows}
    predicted_ids = [str(row.get("stable_id") or "") for row in prediction_rows]
    expected_ids = [str(row["stable_id"]) for row in expected_rows]

    history = training_metrics.get("log_history", [])
    train_losses = [float(row["loss"]) for row in history if "loss" in row]
    eval_losses = [float(row["eval_loss"]) for row in history if "eval_loss" in row]

    required_fields = {
        "stable_id",
        "product",
        "reference_precursors",
        "candidates",
        "runtime_ms",
        "source_method",
        "checkpoint",
    }
    contract_failure_ids = []
    product_mismatch_ids = []
    reference_mismatch_ids = []
    missing_candidate_ids = []
    invalid_candidate_ids = []
    bad_rank_ids = []
    invalid_runtime_ids = []
    candidate_counts = []
    correct_at = {1: 0, 3: 0, 5: 0, 10: 0}
    correct_unmapped_at = {1: 0, 3: 0, 5: 0, 10: 0}

    for row in prediction_rows:
        stable_id = str(row.get("stable_id") or "")
        expected = expected_by_id.get(stable_id)
        if not required_fields.issubset(row):
            contract_failure_ids.append(stable_id)
        if expected is None:
            continue
        if canonical(str(row.get("product") or ""), keep_maps=True) != canonical(
            str(expected.get("product") or ""), keep_maps=True
        ):
            product_mismatch_ids.append(stable_id)
        reference = str(row.get("reference_precursors") or "")
        expected_reference = str(expected.get("reactants") or "")
        if canonical(reference, keep_maps=True) != canonical(
            expected_reference, keep_maps=True
        ):
            reference_mismatch_ids.append(stable_id)

        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        candidate_counts.append(len(candidates))
        if not candidates:
            missing_candidate_ids.append(stable_id)
        candidate_smiles = []
        for rank, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict) or candidate.get("rank") != rank:
                bad_rank_ids.append(stable_id)
                continue
            smiles = str(candidate.get("precursors") or "")
            candidate_smiles.append(smiles)
            score = candidate.get("score")
            if molecule(smiles) is None or not isinstance(score, (int, float)):
                invalid_candidate_ids.append(stable_id)

        for k in correct_at:
            if canonical(expected_reference, keep_maps=True) in {
                canonical(smiles, keep_maps=True) for smiles in candidate_smiles[:k]
            }:
                correct_at[k] += 1
            if canonical(expected_reference, keep_maps=False) in {
                canonical(smiles, keep_maps=False) for smiles in candidate_smiles[:k]
            }:
                correct_unmapped_at[k] += 1

        runtime = row.get("runtime_ms")
        if (
            not isinstance(runtime, (int, float))
            or not math.isfinite(float(runtime))
            or runtime < 0
        ):
            invalid_runtime_ids.append(stable_id)

    duplicate_ids = sorted(
        stable_id for stable_id in set(predicted_ids) if predicted_ids.count(stable_id) > 1
    )
    missing_ids = sorted(set(expected_ids) - set(predicted_ids))
    extra_ids = sorted(set(predicted_ids) - set(expected_ids))
    denominator = len(expected_rows)
    checks = {
        "train_loss_decreased": bool(train_losses and train_losses[-1] < train_losses[0]),
        "eval_loss_decreased": bool(eval_losses and eval_losses[-1] < eval_losses[0]),
        "prediction_row_count_matches": len(prediction_rows) == denominator,
        "stable_ids_match": not missing_ids and not extra_ids and not duplicate_ids,
        "shared_contract_complete": not contract_failure_ids,
        "products_match": not product_mismatch_ids,
        "references_match": not reference_mismatch_ids,
        "all_rows_have_candidate": not missing_candidate_ids,
        "all_candidates_valid": not invalid_candidate_ids,
        "candidate_ranks_valid": not bad_rank_ids,
        "runtimes_valid": not invalid_runtime_ids,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "rxnnano_overfit_audit",
        "dataset": manifest["dataset"],
        "size": manifest["size"],
        "stable_ids_sha256": manifest["stable_ids_sha256"],
        "artifacts": {
            "manifest": str(manifest_path),
            "test": str(test_path),
            "training_metrics": str(training_metrics_path),
            "inference_metrics": str(inference_metrics_path),
            "predictions": str(predictions_path),
        },
        "losses": {
            "first_train": train_losses[0] if train_losses else None,
            "last_train": train_losses[-1] if train_losses else None,
            "first_eval": eval_losses[0] if eval_losses else None,
            "last_eval": eval_losses[-1] if eval_losses else None,
        },
        "prediction_rows": len(prediction_rows),
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "total": sum(candidate_counts),
        },
        "success_at_k_mapped": {
            str(k): correct_at[k] / denominator if denominator else 0.0
            for k in correct_at
        },
        "success_at_k_unmapped": {
            str(k): correct_unmapped_at[k] / denominator if denominator else 0.0
            for k in correct_unmapped_at
        },
        "failure_ids": {
            "missing": missing_ids,
            "extra": extra_ids,
            "duplicate": duplicate_ids,
            "contract": sorted(set(contract_failure_ids)),
            "product_mismatch": sorted(set(product_mismatch_ids)),
            "reference_mismatch": sorted(set(reference_mismatch_ids)),
            "missing_candidate": sorted(set(missing_candidate_ids)),
            "invalid_candidate": sorted(set(invalid_candidate_ids)),
            "bad_rank": sorted(set(bad_rank_ids)),
            "invalid_runtime": sorted(set(invalid_runtime_ids)),
        },
        "inference_metrics": inference_metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path = args.output or (args.run_dir / "overfit_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    preprocess.add_argument("--split", choices=sorted(SPLIT_FILES), default="train")
    preprocess.add_argument("--size", type=int, default=100)
    preprocess.add_argument(
        "--prompt-style", choices=["with_plan", "without_plan"], default="with_plan"
    )
    preprocess.add_argument("--output-dir", type=Path, required=True)
    preprocess.set_defaults(func=preprocess_audit)

    overfit = subparsers.add_parser("prepare-overfit")
    overfit.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    overfit.add_argument("--size", type=int, choices=[32, 128], required=True)
    overfit.add_argument("--output-dir", type=Path, required=True)
    overfit.set_defaults(func=prepare_overfit)

    audit = subparsers.add_parser("audit-overfit")
    audit.add_argument("--slice-dir", type=Path, required=True)
    audit.add_argument("--run-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(func=audit_overfit)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "size", 1) <= 0:
        raise ValueError("--size must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
