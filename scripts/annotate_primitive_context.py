#!/usr/bin/env python3
"""Annotate reaction/tool JSONL with soft mechanistic primitive evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.primitive_library import PrimitiveLibrary


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip(): continue
            value = json.loads(line)
            if not isinstance(value, dict): raise ValueError(f"line {number} is not an object")
            yield value


def first(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    metadata = row.get("metadata") or {}
    for source in (row, metadata if isinstance(metadata, dict) else {}):
        for key in keys:
            if source.get(key) not in (None, ""): return str(source[key])
    return ""


def annotate_step(library: PrimitiveLibrary, step: dict[str, Any], top_k: int) -> dict[str, Any]:
    value = dict(step)
    state = str(value.get("state_smiles") or value.get("state_before") or value.get("reactants") or "")
    target = str(value.get("target_product") or value.get("state_after") or value.get("products") or "")
    if state:
        matches = library.retrieve(state, top_k=top_k)
        value["primitive_candidates"] = [x.to_dict() for x in matches]
        value["primitive_candidate_ids"] = [x.primitive_id for x in matches]
    if state and value.get("moves"):
        value["primitive_move_support"] = library.support_moves(state, list(value.get("moves") or []))
    if state and target:
        value["primitive_reaction_evidence"] = library.annotate_reaction(state, target, top_k=top_k)
    return value


def annotate(library: PrimitiveLibrary, row: dict[str, Any], top_k: int, render: bool) -> dict[str, Any]:
    value = dict(row)
    reactants = first(value, ("reactants", "precursors", "state_smiles", "initial_reactants"))
    products = first(value, ("products", "product", "target_product", "target_smiles", "target"))
    primitive_ids, best = [], 0.0
    if reactants:
        matches = library.retrieve(reactants, top_k=top_k)
        value["primitive_candidates"] = [x.to_dict() for x in matches]
        value["primitive_candidate_ids"] = [x.primitive_id for x in matches]
        if render: value["primitive_context_text"] = library.render_context(reactants, top_k=top_k)
    if reactants and products:
        evidence = library.annotate_reaction(reactants, products, top_k=top_k)
        value["primitive_reaction_evidence"] = evidence
        primitive_ids.extend(evidence["primitive_ids"]); best = float(evidence["best_support"])
    if isinstance(value.get("steps"), list):
        value["steps"] = [annotate_step(library, x, top_k) if isinstance(x, dict) else x for x in value["steps"]]
        for step in value["steps"]:
            if not isinstance(step, dict): continue
            primitive_ids.extend((step.get("primitive_move_support") or {}).get("primitive_ids") or [])
            evidence = step.get("primitive_reaction_evidence") or {}
            primitive_ids.extend(evidence.get("primitive_ids") or [])
            best = max(best, float(evidence.get("best_support") or 0.0))
    primitive_ids = sorted(set(map(str, filter(None, primitive_ids))))
    conditions = dict(value.get("conditions") or {}) if isinstance(value.get("conditions"), dict) else {"raw_conditions": value.get("conditions")}
    conditions.update({
        "_mechet_primitive_ids": primitive_ids,
        "_mechet_primitive_best_support": best,
        "_mechet_primitive_library": library.metadata.get("library_id", "unknown"),
        "_mechet_primitive_evidence_is_soft": True,
    })
    value["conditions"] = conditions
    metadata = dict(value.get("metadata") or {})
    metadata["primitive_augmentation"] = {"library_id": library.metadata.get("library_id", "unknown"), "primitive_ids": primitive_ids, "best_support": best, "soft_evidence_only": True}
    value["metadata"] = metadata
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=REPO / "knowledge/primitives/core_polar_primitives.yaml")
    parser.add_argument("--source-registry", type=Path, default=REPO / "knowledge/source_registry.yaml")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--render-context", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    library = PrimitiveLibrary.load(args.library, source_registry=args.source_registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    read = written = failed = with_ids = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows(args.input):
            read += 1
            if args.limit and read > args.limit: break
            try: value = annotate(library, row, args.top_k, args.render_context)
            except Exception as exc:
                value = dict(row); metadata = dict(value.get("metadata") or {}); metadata["primitive_augmentation_error"] = str(exc); value["metadata"] = metadata; failed += 1
            with_ids += int(bool(((value.get("metadata") or {}).get("primitive_augmentation") or {}).get("primitive_ids")))
            handle.write(json.dumps(value, ensure_ascii=False) + "\n"); written += 1
    manifest = {
        "input": str(args.input), "input_sha256": digest(args.input),
        "output": str(args.output), "output_sha256": digest(args.output),
        "library": str(args.library), "library_sha256": digest(args.library),
        "source_registry": str(args.source_registry), "source_registry_sha256": digest(args.source_registry),
        "top_k": args.top_k, "render_context": args.render_context,
        "read": read, "written": written, "failed": failed,
        "rows_with_primitive_ids": with_ids, "soft_evidence_only": True,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
