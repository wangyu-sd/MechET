#!/usr/bin/env python3
"""Build all matched evidence-layer conditions and a frozen suite manifest."""
from __future__ import annotations

import argparse
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

    loaded: dict[str, list[dict[str, Any]]] = {}
    input_meta: dict[str, dict[str, Any]] = {}
    derived_specs: dict[str, dict[str, Any]] = {}
    for name, spec_value in conditions.items():
        spec = dict(spec_value or {})
        if spec.get("input"):
            path = Path(str(spec["input"]))
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
    written = {}
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
        "suite_id": str(config.get("suite_id") or args.config.stem),
        "scientific_question": str(config.get("scientific_question") or ""),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "n_matched_ids": len(identifiers),
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
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
