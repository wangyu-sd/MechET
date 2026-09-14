#!/usr/bin/env python3
"""Export official RetroDFM-R generations for MechET endpoint evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(row: dict[str, Any]) -> str:
    value = str(row.get("stable_id") or row.get("id") or "").strip()
    if not value:
        raise ValueError("row lacks stable_id/id")
    return value


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-input", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluation-reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inference-server",
        default="vLLM OpenAI-compatible server",
        help="Runtime serving the unchanged official eval/generate.py client.",
    )
    args = parser.parse_args()

    inputs = read_jsonl(args.generation_input)
    generations = read_jsonl(args.generations)
    references = read_jsonl(args.reference)
    if not (len(inputs) == len(generations) == len(references) == args.expected_rows):
        raise ValueError(
            "row-count mismatch: "
            f"inputs={len(inputs)}, generations={len(generations)}, "
            f"references={len(references)}, expected={args.expected_rows}"
        )

    reference_by_id: dict[str, dict[str, Any]] = {}
    for row in references:
        identifier = stable_id(row)
        if identifier in reference_by_id:
            raise ValueError(f"duplicate reference stable ID: {identifier}")
        reference_by_id[identifier] = row

    predictions: list[dict[str, Any]] = []
    evaluation_references: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing_answer_candidates = 0
    for index, (source, generated) in enumerate(zip(inputs, generations, strict=True)):
        identifier = stable_id(source)
        if identifier in seen:
            raise ValueError(f"duplicate generation stable ID: {identifier}")
        seen.add(identifier)
        reference = reference_by_id.get(identifier)
        if reference is None:
            raise ValueError(f"generation ID absent from frozen reference: {identifier}")
        if str(generated.get("input") or "") != str(source.get("input") or ""):
            raise ValueError(f"generation/input order mismatch at row {index}: {identifier}")
        outputs = list(generated.get("output") or [])
        if len(outputs) != args.expected_candidates:
            raise ValueError(
                f"{identifier} has {len(outputs)} outputs; "
                f"expected {args.expected_candidates}"
            )

        candidates: list[dict[str, Any]] = []
        for sample_index, raw_output in enumerate(outputs):
            raw_text = str(raw_output or "")
            precursor = extract_answer(raw_text)
            missing_answer_candidates += int(not precursor)
            candidates.append(
                {
                    "rank": sample_index + 1,
                    "sample_index": sample_index,
                    "precursors": precursor,
                    "prediction": precursor,
                    "raw_output": raw_text,
                    "score": 0.0,
                    "generation_error": "" if precursor else "missing_answer_tag",
                }
            )

        product = str(
            reference.get("product_mapped")
            or reference.get("product_unmapped")
            or source.get("product")
            or ""
        )
        reference_precursors = str(
            reference.get("precursor_mapped")
            or reference.get("precursor_unmapped")
            or ((source.get("label") or {}).get("reactants"))
            or ""
        )
        predictions.append(
            {
                "artifact_type": "prediction",
                "id": identifier,
                "stable_id": identifier,
                "source_index": index,
                "product": product,
                "reference_precursors": reference_precursors,
                "candidates": candidates,
                "runtime_ms": 0.0,
                "source_method": "RetroDFM-R-8B",
                "checkpoint": args.checkpoint,
                "checkpoint_revision": args.model_revision,
                "candidate_budget": args.expected_candidates,
                "generated_candidate_count": len(candidates),
                "valid_candidate_count": sum(bool(item["prediction"]) for item in candidates),
                "candidate_semantics": "independent_stochastic_samples_pass_at_k",
                "ranking": "generation_order",
                "score_type": "unavailable",
            }
        )
        evaluation_references.append(
            {
                "id": identifier,
                "stable_id": identifier,
                "target_smiles": product,
                "structural_precursor": reference_precursors,
            }
        )

    if seen != set(reference_by_id):
        missing = sorted(set(reference_by_id) - seen)
        raise ValueError(f"frozen reference IDs missing from generations: {missing[:10]}")

    write_jsonl(args.predictions, predictions)
    write_jsonl(args.evaluation_reference, evaluation_references)
    manifest = {
        "artifact_type": "retrodfm_r_frozen_checkpoint_inference_v1",
        "dataset": args.dataset,
        "endpoint_only": True,
        "source_method": "RetroDFM-R-8B",
        "checkpoint": args.checkpoint,
        "model_revision": args.model_revision,
        "generation_code": str((Path(__file__).resolve().parent / "generate.py")),
        "inference_server": args.inference_server,
        "generation_input": str(args.generation_input.resolve()),
        "generation_input_sha256": sha256_file(args.generation_input),
        "frozen_reference": str(args.reference.resolve()),
        "frozen_reference_sha256": sha256_file(args.reference),
        "raw_generations": str(args.generations.resolve()),
        "raw_generations_sha256": sha256_file(args.generations),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "evaluation_reference": str(args.evaluation_reference.resolve()),
        "evaluation_reference_sha256": sha256_file(args.evaluation_reference),
        "n_rows": len(predictions),
        "candidates_per_target": args.expected_candidates,
        "missing_answer_tag_candidates": missing_answer_candidates,
        "candidate_semantics": "independent_stochastic_samples_pass_at_k",
        "selection_uses_ground_truth": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "server_seed": args.seed,
        "proof_or_replay_filtering": False,
        "overlap_filtering": False,
        "token_length_filtering": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
