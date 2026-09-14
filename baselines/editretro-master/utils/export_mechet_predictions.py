#!/usr/bin/env python3
"""Export native EditRetro iterative beams to the MechET prediction contract.

Ranking matches ``utils/score.py``: hypotheses are ordered by their mean
negative positional log score inside each root-aligned product view, invalid
SMILES are removed, and canonical duplicates accumulate
``1 / (score_alpha * beam_rank + 1)`` across all augmented views.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterator

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
HYPOTHESIS = re.compile(r"^H-(\d+)\t([^\t]+)\t(.*)$")
POSITIONAL = re.compile(r"^P-(\d+)\t(.*)$")


@dataclass(frozen=True)
class NativeCandidate:
    prediction: str
    canonical: str
    largest_fragment: str
    native_score: float
    beam_position: int


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def stable_id(row: dict[str, Any]) -> str:
    value = str(row.get("stable_id") or row.get("id") or "").strip()
    if not value:
        raise ValueError("reference row missing stable_id/id")
    return value


def product_smiles(row: dict[str, Any]) -> str:
    return str(
        row.get("product_mapped")
        or row.get("target_smiles")
        or row.get("product_unmapped")
        or ""
    ).strip()


def precursor_smiles(row: dict[str, Any]) -> str:
    return str(
        row.get("precursor_mapped")
        or row.get("structural_precursor")
        or row.get("expected_precursor")
        or row.get("precursor_unmapped")
        or ""
    ).strip()


def canonicalize(smiles: str) -> tuple[str, str] | None:
    text = smiles.replace("<unk>", "").replace(" ", "").strip()
    if not text:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        return None
    fragments = canonical.split(".")
    sizes: list[tuple[int, int, str]] = []
    for fragment_index, fragment in enumerate(fragments):
        fragment_molecule = Chem.MolFromSmiles(fragment)
        if fragment_molecule is not None:
            sizes.append((fragment_molecule.GetNumAtoms(), -fragment_index, fragment))
    largest = max(sizes)[2] if sizes else ""
    return canonical, largest


def positional_score(text: str) -> float:
    try:
        values = [float(value) for value in text.strip().split()]
    except ValueError as exc:
        raise ValueError(f"invalid positional scores: {text!r}") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid positional scores: {text!r}")
    negative_count = sum(value < 0 for value in values)
    # The published post_process.py rounds this value before ranking.
    return round(sum(values) / negative_count, 6) if negative_count else 0.0


def parse_generation(
    path: Path, beam_size: int
) -> tuple[dict[int, list[tuple[str, float]]], dict[str, int]]:
    hypotheses: dict[int, list[str]] = defaultdict(list)
    positionals: dict[int, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            hypothesis_match = HYPOTHESIS.match(line.rstrip("\n"))
            if hypothesis_match:
                hypotheses[int(hypothesis_match.group(1))].append(
                    hypothesis_match.group(3)
                )
                continue
            positional_match = POSITIONAL.match(line.rstrip("\n"))
            if positional_match:
                positionals[int(positional_match.group(1))].append(
                    positional_score(positional_match.group(2))
                )

    sample_ids = set(hypotheses) | set(positionals)
    parsed: dict[int, list[tuple[str, float]]] = {}
    incomplete = 0
    for sample_id in sample_ids:
        sample_hypotheses = hypotheses.get(sample_id, [])
        sample_scores = positionals.get(sample_id, [])
        if len(sample_hypotheses) != len(sample_scores):
            raise ValueError(
                f"sample {sample_id} has {len(sample_hypotheses)} hypotheses "
                f"and {len(sample_scores)} positional-score rows"
            )
        if len(sample_hypotheses) != beam_size:
            incomplete += 1
        parsed[sample_id] = list(zip(sample_hypotheses, sample_scores))
    stats = {
        "generation_sample_ids": len(sample_ids),
        "generation_hypotheses": sum(len(value) for value in hypotheses.values()),
        "incomplete_beam_samples": incomplete,
    }
    return parsed, stats


def load_line_map(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    for index, row in enumerate(rows):
        if int(row.get("line_index", -1)) != index:
            raise ValueError(f"non-contiguous line_index at {path}:{index + 1}")
        if not str(row.get("stable_id") or ""):
            raise ValueError(f"missing stable_id at {path}:{index + 1}")
    return rows


def read_runtime_ms(path: Path | None) -> float:
    if path is None:
        return 0.0
    value = path.read_text(encoding="utf-8").strip().splitlines()[0]
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"invalid runtime in {path}: {value!r}")
    return seconds * 1000.0


def command_export(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    references = list(read_jsonl(args.reference))
    if args.profile == "audit100":
        references = references[:100]
    reference_ids = [stable_id(row) for row in references]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("reference contains duplicate stable IDs")
    if args.expected_rows and len(references) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} reference rows, found {len(references)}"
        )
    reference_id_set = set(reference_ids)

    line_map = load_line_map(args.line_map)
    generated, generation_stats = parse_generation(args.fairseq_log, args.beam_size)
    out_of_range = sorted(set(generated) - set(range(len(line_map))))
    if out_of_range:
        raise ValueError(f"generation contains out-of-range sample IDs: {out_of_range[:10]}")

    views: dict[str, dict[int, list[NativeCandidate]]] = defaultdict(dict)
    missing_generation_by_id: dict[str, set[int]] = defaultdict(set)
    incomplete_beam_by_id: dict[str, set[int]] = defaultdict(set)
    invalid_candidates = 0
    missing_generation_views = 0
    for line_index, metadata in enumerate(line_map):
        identifier = str(metadata["stable_id"])
        if identifier not in reference_id_set:
            raise ValueError(f"line map contains unknown stable ID: {identifier}")
        augmentation_index = int(metadata["augmentation_index"])
        if augmentation_index in views[identifier]:
            raise ValueError(
                f"duplicate augmentation {augmentation_index} for {identifier}"
            )
        hypotheses = generated.get(line_index)
        if hypotheses is None:
            missing_generation_views += 1
            missing_generation_by_id[identifier].add(augmentation_index)
            views[identifier][augmentation_index] = []
            continue
        if len(hypotheses) != args.beam_size:
            incomplete_beam_by_id[identifier].add(augmentation_index)
        native_candidates: list[NativeCandidate] = []
        for beam_position, (prediction, score) in enumerate(hypotheses):
            normalized = canonicalize(prediction)
            if normalized is None:
                invalid_candidates += 1
                continue
            canonical, largest = normalized
            native_candidates.append(
                NativeCandidate(
                    prediction=canonical,
                    canonical=canonical,
                    largest_fragment=largest,
                    native_score=score,
                    beam_position=beam_position,
                )
            )
        native_candidates.sort(
            key=lambda item: (-item.native_score, item.beam_position)
        )
        views[identifier][augmentation_index] = native_candidates

    inference_runtime_total_ms = read_runtime_ms(args.runtime_seconds_file)
    inference_runtime_per_target = inference_runtime_total_ms / max(len(references), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp.{os.getpid()}")
    rows_ok = 0
    rows_missing_views = 0
    rows_incomplete_beams = 0
    rows_without_valid_candidates = 0
    try:
        with temporary.open("w", encoding="utf-8") as output_handle:
            for reference in references:
                identifier = stable_id(reference)
                augmented = views.get(identifier, {})
                missing_augmentations = sorted(
                    set(range(args.augmentation)) - set(augmented)
                )
                accumulated: dict[tuple[str, str], dict[str, Any]] = {}
                fusion_first_seen = 0
                for augmentation_index in range(args.augmentation):
                    for native_rank, candidate in enumerate(
                        augmented.get(augmentation_index, [])
                    ):
                        key = (candidate.canonical, candidate.largest_fragment)
                        vote = 1.0 / (args.score_alpha * native_rank + 1.0)
                        if key not in accumulated:
                            accumulated[key] = {
                                "precursors": candidate.prediction,
                                "score": 0.0,
                                # Python's stable dict order is what the
                                # official score.py uses to resolve equal
                                # fused scores after per-view ranking.
                                "first_seen": fusion_first_seen,
                                "best_native_score": candidate.native_score,
                            }
                            fusion_first_seen += 1
                        accumulated[key]["score"] += vote
                        accumulated[key]["best_native_score"] = max(
                            float(accumulated[key]["best_native_score"]),
                            candidate.native_score,
                        )
                ranked = sorted(
                    accumulated.values(),
                    key=lambda item: (-float(item["score"]), int(item["first_seen"])),
                )[: args.top_n]
                missing_native_views = sorted(
                    set(missing_augmentations)
                    | missing_generation_by_id.get(identifier, set())
                )
                incomplete_native_beams = sorted(
                    incomplete_beam_by_id.get(identifier, set())
                )
                # A truncated native decode is a failed target, not a smaller
                # ad-hoc beam/TTA condition.  Keep it in the denominator but
                # do not let partial output receive endpoint credit.
                if missing_native_views or incomplete_native_beams:
                    ranked = []
                candidates: list[dict[str, Any]] = []
                for rank, candidate in enumerate(ranked, 1):
                    prediction = str(candidate["precursors"])
                    candidates.append(
                        {
                            "rank": rank,
                            "precursors": prediction,
                            "prediction": prediction,
                            "score": float(candidate["score"]),
                            "best_native_positional_score": float(
                                candidate["best_native_score"]
                            ),
                        }
                    )
                while len(candidates) < args.top_n:
                    rank = len(candidates) + 1
                    candidates.append(
                        {
                            "rank": rank,
                            "precursors": "",
                            "prediction": "",
                            "score": 0.0,
                        }
                    )

                if missing_native_views:
                    evaluation_status = "missing_augmented_views"
                    rows_missing_views += 1
                elif incomplete_native_beams:
                    evaluation_status = "incomplete_native_beam"
                    rows_incomplete_beams += 1
                elif not ranked:
                    evaluation_status = "no_valid_candidates"
                    rows_without_valid_candidates += 1
                else:
                    evaluation_status = "ok"
                    rows_ok += 1
                output_row = {
                    "id": identifier,
                    "stable_id": identifier,
                    "product": product_smiles(reference),
                    "reference_precursors": precursor_smiles(reference),
                    "candidates": candidates,
                    "runtime_ms": inference_runtime_per_target,
                    "source_method": "EditRetro",
                    "checkpoint": args.checkpoint,
                    "ranking": "native_editretro_tta_reciprocal_rank_fusion",
                    "score_alpha": args.score_alpha,
                    "augmentation": args.augmentation,
                    "native_beam_size": args.beam_size,
                    "evaluation_status": evaluation_status,
                    "forced_error": evaluation_status != "ok",
                }
                output_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    postprocess_runtime_ms = (time.perf_counter() - started) * 1000.0
    report = {
        "schema_version": 1,
        "artifact_type": "editretro_mechet_prediction_export",
        "reference": str(args.reference.resolve()),
        "line_map": str(args.line_map.resolve()),
        "fairseq_log": str(args.fairseq_log.resolve()),
        "output": str(args.output.resolve()),
        "checkpoint": args.checkpoint,
        "profile": args.profile,
        "reference_rows": len(references),
        "line_map_views": len(line_map),
        "expected_views": len(references) * args.augmentation,
        "augmentation": args.augmentation,
        "native_beam_size": args.beam_size,
        "top_n": args.top_n,
        "score_alpha": args.score_alpha,
        "ranking": "native_editretro_tta_reciprocal_rank_fusion",
        "rows_ok": rows_ok,
        "rows_missing_views": rows_missing_views,
        "rows_incomplete_beams": rows_incomplete_beams,
        "rows_without_valid_candidates": rows_without_valid_candidates,
        "missing_generation_views": missing_generation_views,
        "invalid_native_candidates": invalid_candidates,
        "inference_runtime_ms_total": inference_runtime_total_ms,
        "inference_runtime_ms_per_target": inference_runtime_per_target,
        "postprocess_runtime_ms_total": postprocess_runtime_ms,
        "failures_counted_in_full_denominator": True,
        **generation_stats,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--line-map", type=Path, required=True)
    parser.add_argument("--fairseq-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", choices=("full", "audit100"), default="full")
    parser.add_argument("--augmentation", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--score-alpha", type=float, default=0.1)
    parser.add_argument("--runtime-seconds-file", type=Path)
    parser.add_argument("--expected-rows", type=int, default=0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("augmentation", "beam_size", "top_n"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.score_alpha < 0:
        parser.error("--score-alpha cannot be negative")
    return command_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
