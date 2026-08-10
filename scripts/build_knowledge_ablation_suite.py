#!/usr/bin/env python3
"""Build matched evidence conditions with optional train/valid/test isolation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("install mechet[knowledge]") from exc

from mechet.knowledge_ablation import (
    condition_contract_summary,
    file_sha256,
    make_direct_textbook_condition,
    make_irrelevant_context_control,
    matched_intersection,
    read_jsonl,
    strip_knowledge_messages,
    strip_textbook_keep_anchors,
    validate_alignment,
    write_jsonl,
)

TRANSFORMS = {
    "strip_knowledge": lambda rows: [strip_knowledge_messages(row) for row in rows],
    "strip_textbook_keep_anchors": lambda rows: [
        strip_textbook_keep_anchors(row) for row in rows
    ],
    "length_matched_irrelevant": make_irrelevant_context_control,
    "direct_answer_from_textbook": lambda rows: [
        make_direct_textbook_condition(row) for row in rows
    ],
}


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value.get("conditions"), dict):
        raise ValueError("suite config requires a conditions mapping")
    if value.get("splits") is not None and not isinstance(value.get("splits"), dict):
        raise ValueError("suite config splits must be a mapping")
    return dict(value)


def derive_conditions(
    matched: dict[str, list[dict[str, Any]]],
    specs: dict[str, dict[str, Any]],
    input_meta: dict[str, dict[str, Any]],
) -> None:
    pending = dict(specs)
    while pending:
        progressed = False
        for name, spec in list(pending.items()):
            source_name = str(spec.get("derive_from") or "")
            if source_name not in matched:
                continue
            transform = str(spec.get("transform") or "")
            if transform not in TRANSFORMS:
                raise ValueError(f"unknown transform for {name}: {transform}")
            matched[name] = TRANSFORMS[transform](matched[source_name])
            input_meta[name] = {
                "derive_from": source_name,
                "transform": transform,
                "declared_knowledge": spec.get("knowledge"),
                "declared_environment": spec.get("environment"),
            }
            del pending[name]
            progressed = True
        if not progressed:
            unresolved = {
                name: str(spec.get("derive_from") or "")
                for name, spec in pending.items()
            }
            raise ValueError(f"unresolved derived condition dependencies: {unresolved}")


def _conditions_with_split_inputs(
    base: dict[str, Any], split_spec: dict[str, Any]
) -> dict[str, Any]:
    conditions = deepcopy(base)
    inputs = dict(split_spec.get("inputs") or {})
    if not inputs:
        raise ValueError("each split requires an inputs mapping")
    for name, path in inputs.items():
        if name not in conditions:
            raise ValueError(f"split overrides unknown condition: {name}")
        if conditions[name].get("derive_from"):
            raise ValueError(
                f"split input override must target a source condition, not {name}"
            )
        conditions[name]["input"] = str(path)
    required_sources = {
        name
        for name, spec in conditions.items()
        if not dict(spec or {}).get("derive_from")
    }
    missing = sorted(required_sources - set(inputs))
    if missing:
        raise ValueError(f"split inputs missing source conditions: {missing}")
    return conditions


def build_one_suite(
    *,
    config: dict[str, Any],
    conditions: dict[str, Any],
    output_dir: Path,
    config_path: Path,
    split_name: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    input_meta: dict[str, dict[str, Any]] = {}
    derived_specs: dict[str, dict[str, Any]] = {}
    for name, spec_value in conditions.items():
        spec = dict(spec_value or {})
        if spec.get("input"):
            path = Path(str(spec["input"]))
            if not path.exists():
                raise FileNotFoundError(f"condition input does not exist: {path}")
            loaded[name] = read_jsonl(path)
            input_meta[name] = {
                "input": str(path),
                "input_sha256": file_sha256(path),
                "declared_knowledge": spec.get("knowledge"),
                "declared_environment": spec.get("environment"),
            }
        elif spec.get("derive_from"):
            derived_specs[name] = spec
        else:
            raise ValueError(f"condition {name} requires input or derive_from")

    identifiers, matched = matched_intersection(loaded)
    derive_conditions(matched, derived_specs, input_meta)
    validate_alignment(matched)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    for name in conditions:
        rows = matched[name]
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        written[name] = {
            **input_meta.get(name, {}),
            "output": str(path),
            "output_sha256": file_sha256(path),
            **condition_contract_summary(rows),
        }

    ids_digest = hashlib.sha256("\n".join(identifiers).encode()).hexdigest()
    manifest = {
        "suite_id": str(config.get("suite_id") or config_path.stem),
        "split": split_name,
        "scientific_question": str(config.get("scientific_question") or ""),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "n_matched_ids": len(identifiers),
        "matched_ids": identifiers,
        "matched_ids_sha256": ids_digest,
        "conditions": written,
        "controls": {
            "same_stable_ids": True,
            "same_target_and_expected_precursor": True,
            "knowledge_retrieval_direct_reward": False,
            "irrelevant_context_is_length_matched": any(
                item.get("transform") == "length_matched_irrelevant"
                for item in input_meta.values()
            ),
            "anchors_only_is_derived_from_combined": any(
                item.get("transform") == "strip_textbook_keep_anchors"
                for item in input_meta.values()
            ),
            "direct_open_book_uses_same_bounded_evidence": any(
                item.get("transform") == "direct_answer_from_textbook"
                for item in input_meta.values()
            ),
        },
        "training_contract": dict(config.get("training_contract") or {}),
        "evaluation_contract": dict(config.get("evaluation_contract") or {}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest, identifiers


def _cross_split_overlap(id_sets: dict[str, set[str]]) -> dict[str, int]:
    names = list(id_sets)
    output: dict[str, int] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            output[f"{left}__{right}"] = len(id_sets[left] & id_sets[right])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or Path(
        str(config.get("output_dir") or "data/knowledge_ablation")
    )
    conditions = dict(config["conditions"])
    split_specs = dict(config.get("splits") or {})

    if not split_specs:
        manifest, _ = build_one_suite(
            config=config,
            conditions=conditions,
            output_dir=output_dir,
            config_path=args.config,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    required = {"train", "valid", "test"}
    missing_splits = sorted(required - set(split_specs))
    if missing_splits:
        raise ValueError(
            f"split-aware evidence suite requires train/valid/test; missing {missing_splits}"
        )

    split_manifests: dict[str, Any] = {}
    id_sets: dict[str, set[str]] = {}
    for split_name in ("train", "valid", "test"):
        split_spec = dict(split_specs[split_name] or {})
        split_conditions = _conditions_with_split_inputs(conditions, split_spec)
        split_manifest, identifiers = build_one_suite(
            config=config,
            conditions=split_conditions,
            output_dir=output_dir / split_name,
            config_path=args.config,
            split_name=split_name,
        )
        id_sets[split_name] = set(identifiers)
        split_manifests[split_name] = {
            "manifest": str(output_dir / split_name / "manifest.json"),
            "n_matched_ids": split_manifest["n_matched_ids"],
            "matched_ids_sha256": split_manifest["matched_ids_sha256"],
        }

    overlap = _cross_split_overlap(id_sets)
    if any(overlap.values()):
        raise ValueError(f"evidence train/valid/test stable IDs overlap: {overlap}")

    manifest = {
        "suite_id": str(config.get("suite_id") or args.config.stem),
        "artifact_type": "split_evidence_suite_manifest",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "output_dir": str(output_dir),
        "splits": split_manifests,
        "cross_split_id_overlap": overlap,
        "controls": {
            "train_valid_test_stable_ids_disjoint": True,
            "final_evaluation_split": "test",
            "training_split": "train",
            "model_selection_split": "valid",
        },
        "training_contract": dict(config.get("training_contract") or {}),
        "evaluation_contract": dict(config.get("evaluation_contract") or {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
