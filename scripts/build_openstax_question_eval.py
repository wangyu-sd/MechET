#!/usr/bin/env python3
"""Pair OpenStax problem/answer artifacts into a review-gated evaluation queue."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_chapter(chapter: int) -> str:
    if chapter <= 24:
        return "development"
    if chapter <= 27:
        return "validation"
    return "frozen_test"


def coarse_topics(text: str) -> list[str]:
    families = {
        "spectroscopy": ("nmr", "infrared", "spectrum", "spectra", "chemical shift"),
        "mass_spectrometry": ("mass spectrum", "mass spectrometry", "molecular ion", "m/z"),
        "structure_elucidation": ("propose a structure", "determine the structure", "identify the compound"),
        "mechanism": ("mechanism", "electron", "transition state", "intermediate"),
        "synthesis": ("synthesize", "synthesis", "starting from", "reagent"),
        "stereochemistry": ("stereoisomer", "enantiomer", "diastereomer", "stereochemistry"),
    }
    lowered = text.lower()
    return [key for key, terms in families.items() if any(term in lowered for term in terms)]


def load_items(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    questions: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "pages").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("assessment_items") or []:
            target = questions if item.get("kind") == "question" else answers if item.get("kind") == "answer" else None
            if target is None:
                continue
            problem_id = str(item.get("problem_id") or "")
            if not re.fullmatch(r"\d+-\d+", problem_id):
                continue
            target[problem_id] = {
                **item,
                "page_url": payload.get("canonical_url"),
                "page_revision": payload.get("content_version"),
                "page_slug": payload.get("slug"),
            }
    return questions, answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("knowledge/raw_corpus_v2/openstax_organic_chemistry_10e"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/openstax_organic_eval_v1"))
    args = parser.parse_args()
    questions, answers = load_items(args.raw_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "candidates.jsonl"
    questions_output = args.output_dir / "text_only_questions.jsonl"
    answers_output = args.output_dir / "answer_key.jsonl"
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    with (
        output.open("w", encoding="utf-8") as handle,
        questions_output.open("w", encoding="utf-8") as questions_handle,
        answers_output.open("w", encoding="utf-8") as answers_handle,
    ):
        for problem_id in sorted(set(questions) & set(answers), key=lambda x: tuple(map(int, x.split("-")))):
            question, answer = questions[problem_id], answers[problem_id]
            chapter = int(problem_id.split("-", 1)[0])
            figure_dependent = bool(question.get("has_figure") or answer.get("has_figure"))
            text_only_candidate = bool(question.get("text") and answer.get("text") and not figure_dependent)
            topics = coarse_topics(f"{question.get('text', '')} {answer.get('text', '')}")
            row = {
                "question_id": f"openstax-oc10e-{problem_id}",
                "problem_id": problem_id,
                "chapter": chapter,
                "split": split_for_chapter(chapter),
                "question": question.get("text"),
                "reference_answer": answer.get("text"),
                "topics": topics,
                "figure_dependent": figure_dependent,
                "image_alts": question.get("image_alts") or [],
                "text_only_candidate": text_only_candidate,
                "review_status": "unreviewed",
                "scoring_eligible": False,
                "scoring_blocker": "independent chemistry review required",
                "question_source": question.get("page_url"),
                "answer_source": answer.get("page_url"),
                "source_version": question.get("page_revision"),
                "license": "CC-BY-NC-SA-4.0",
                "question_sha256": question.get("text_sha256"),
                "answer_sha256": answer.get("text_sha256"),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            answers_handle.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "reference_answer": row["reference_answer"],
                        "answer_source": row["answer_source"],
                        "answer_sha256": row["answer_sha256"],
                        "review_status": "unreviewed",
                        "scoring_eligible": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if text_only_candidate:
                questions_handle.write(
                    json.dumps(
                        {
                            "question_id": row["question_id"],
                            "problem_id": row["problem_id"],
                            "chapter": row["chapter"],
                            "split": row["split"],
                            "question": row["question"],
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "Answer the organic chemistry question using explicit chemical reasoning. "
                                        "State assumptions, give the final answer clearly, and do not claim to see "
                                        "a structure or spectrum that is absent from the prompt."
                                    ),
                                },
                                {"role": "user", "content": row["question"]},
                            ],
                            "topics": row["topics"],
                            "review_status": "unreviewed",
                            "scoring_eligible": False,
                            "question_source": row["question_source"],
                            "source_version": row["source_version"],
                            "license": row["license"],
                            "question_sha256": row["question_sha256"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            counts["paired"] += 1
            counts["figure_dependent" if figure_dependent else "text_only"] += 1
            split_counts[row["split"]] += 1
            topic_counts.update(topics)
    manifest = {
        "schema_version": 1,
        "status": "candidate_review_queue",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "OpenStax Organic Chemistry 10e chapter exercises and answer keys",
        "license": "CC-BY-NC-SA-4.0",
        "n_questions_extracted": len(questions),
        "n_answers_extracted": len(answers),
        "n_paired": counts["paired"],
        "n_text_only_candidates": counts["text_only"],
        "n_figure_dependent": counts["figure_dependent"],
        "split_counts": dict(split_counts),
        "topic_counts": dict(topic_counts),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "text_only_questions": str(questions_output),
        "text_only_questions_sha256": sha256_file(questions_output),
        "answer_key": str(answers_output),
        "answer_key_sha256": sha256_file(answers_output),
        "inference_contract": "model inference reads text_only_questions.jsonl only; answer_key.jsonl is evaluator-only",
        "promotion_contract": "scoring_eligible remains false until independent chemistry review and rubric validation",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
