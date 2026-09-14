#!/usr/bin/env python3
"""Validate that bundled baselines are source-only and runnable from raw data."""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "baselines"
MAX_FILE_BYTES = 50 * 1024 * 1024

REQUIRED_FILES = {
    "LocalRetro-main": (
        "README.md",
        "LocalTemplate/template_extractor.py",
        "preprocessing/Extract_from_train_data.py",
        "preprocessing/Run_preprocessing.py",
        "scripts/Train.py",
        "data/configs/default_config.json",
    ),
    "ReactSeq-main": (
        "README.md",
        "prepare_mechet_reactseq.py",
        "preprocess_data.py",
        "train.py",
        "transform.py",
        "onmt/bin/train.py",
        "datasets/vocabs/whole_vocabs.src",
    ),
    "Retro-MTGR-main": (
        "README.md",
        "prepare_mechet_retro_mtgr.py",
        "retro_mtgr_data_processing.py",
        "train_mechet_retro_mtgr.py",
        "retro_mtgr_adapter_model.py",
        "data/USPT-50K/other_data/Bond_Energy.txt",
    ),
    "RetroBridge-main": (
        "README.md",
        "prepare_mechet_retrobridge.py",
        "train.py",
        "sample.py",
        "src/data/retrobridge_dataset.py",
        "src/models/transformer_model.py",
    ),
    "RetroDFM-R-main": (
        "README.md",
        "eval/generate.py",
        "swift_scripts/train_pretrain.sh",
        "swift_scripts/train_cold_start.sh",
        "slime/train.py",
    ),
    "RetroSynFlow-main": (
        "README.md",
        "scripts/preprocess_mechet.py",
        "scripts/train_reaction_center_mechet.py",
        "scripts/train_synthon_flow_mechet.py",
        "src/retflow/datasets/dataset.py",
    ),
    "RxnNano-main": (
        "README.md",
        "train.py",
        "evaluate.py",
        "src/data/dataset_loader.py",
        "src/models/model_loader.py",
    ),
    "editretro-master": (
        "README.md",
        "preprocess/prepare_mechet_data.py",
        "scripts/run_mechet_preprocess.sh",
        "scripts/run_mechet_full_pipeline.sh",
        "editretro/models/editretro_nat.py",
        "fairseq/fairseq_cli/train.py",
    ),
    "retrosynthesis-main": (
        "Readme.md",
        "preprocessing/generate_PtoR_data.py",
        "shell_cmds/train_flower_full.sh",
        "shell_cmds/train_mech_uspto31k_full.sh",
        "train-from-scratch/PtoR/flower-full-aug5-config.yml",
    ),
}

FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".csv",
    ".feather",
    ".idx",
    ".jsonl",
    ".npy",
    ".npz",
    ".o",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".so",
}

FORBIDDEN_TREE_NAMES = {
    ".cache",
    ".conda-envs",
    ".deps",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime",
    "__pycache__",
    "build",
    "dist",
    "unsloth_compiled_cache",
}

FORBIDDEN_BASELINE_ROOTS = {
    "checkpoints",
    "exp",
    "logs",
    "models",
    "output",
    "outputs",
    "results",
    "runs",
    "runtime",
    "trained_models",
    "wandb",
}

ALLOWED_DATA_FILES = {
    "LocalRetro-main": {
        "data/README.md",
        "data/configs/default_config.json",
    },
    "ReactSeq-main": {"datasets/vocabs/whole_vocabs.src"},
    "Retro-MTGR-main": {
        "data/USPT-50K/other_data/Atom_weight.txt",
        "data/USPT-50K/other_data/Atoms_character.txt",
        "data/USPT-50K/other_data/Bond_Energy.txt",
        "data/USPT-50K/other_data/Labels.txt",
    },
    "RxnNano-main": {
        "data/manifest_flower_full.json",
        "data/manifest_mech_uspto_31k_full.json",
        "data/readme.md",
    },
}


def main() -> int:
    errors: list[str] = []
    actual = {path.name for path in ROOT.iterdir() if path.is_dir()}
    expected = set(REQUIRED_FILES)
    if actual != expected:
        errors.append(
            f"baseline directory mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    for baseline, required in REQUIRED_FILES.items():
        baseline_root = ROOT / baseline
        for relative in required:
            required_path = baseline_root / relative
            if not required_path.is_file():
                errors.append(f"missing required source file: {baseline}/{relative}")
            elif required_path.suffix == ".py":
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        ast.parse(
                            required_path.read_text(encoding="utf-8"),
                            filename=str(required_path),
                        )
                except (SyntaxError, UnicodeDecodeError) as exc:
                    errors.append(
                        f"required Python entry point does not parse: "
                        f"{baseline}/{relative}: {exc}"
                    )

        for root_name in FORBIDDEN_BASELINE_ROOTS:
            if (baseline_root / root_name).exists():
                errors.append(f"generated root is present: {baseline}/{root_name}")

        allowed_data = ALLOWED_DATA_FILES.get(baseline, set())
        for data_name in ("data", "dataset", "datasets"):
            data_root = baseline_root / data_name
            if not data_root.exists():
                continue
            for path in data_root.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(baseline_root).as_posix()
                    if relative not in allowed_data:
                        errors.append(f"unexpected bundled data file: {baseline}/{relative}")

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_dir():
            if path.name.startswith(".venv") or path.name in FORBIDDEN_TREE_NAMES:
                errors.append(f"generated environment/cache is present: {relative}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"raw/model artifact is present: {relative}")
        if path.name.endswith(".egg-info"):
            errors.append(f"generated package metadata is present: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(
                f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MiB policy: {relative}"
            )

    if errors:
        print("Baseline source snapshot validation failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1

    total_files = sum(path.is_file() for path in ROOT.rglob("*"))
    total_bytes = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())
    print(
        f"Baseline source snapshot OK: {len(REQUIRED_FILES)} baselines, "
        f"{total_files} files, {total_bytes / (1024 * 1024):.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
