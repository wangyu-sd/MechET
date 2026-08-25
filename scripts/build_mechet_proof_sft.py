#!/usr/bin/env python3
"""Compile MECH_ET v3 SFT JSONL into executable MECH_PROOF v1 SFT JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_sft import convert_mech_et_row_to_proof_sft


def _convert_line(task: tuple[int, str]) -> tuple[bool, int, str | None, str | None]:
    line_number, line = task
    try:
        row = json.loads(line)
        converted = convert_mech_et_row_to_proof_sft(row)
        return True, line_number, json.dumps(converted, ensure_ascii=False), None
    except Exception as exc:
        return False, line_number, None, f"{type(exc).__name__}: {exc}"


def _iter_chunks(reader, *, chunk_size: int):
    while True:
        chunk = list(islice(reader, chunk_size))
        if not chunk:
            return
        yield chunk


def _process_chunk(
    chunk: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    successes: list[tuple[int, str]] = []
    failures: list[tuple[int, str]] = []
    for result in map(_convert_line, chunk):
        ok, line_number, converted, error = result
        if ok:
            assert converted is not None
            successes.append((line_number, converted))
        else:
            assert error is not None
            failures.append((line_number, error))
    return successes, failures


def build_split_parallel(
    src: Path,
    dst: Path,
    limit: int = 0,
    workers: int | None = None,
    chunk_size: int = 64,
) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    skipped = 0
    reasons: Counter[str] = Counter()
    max_workers = workers or max((os.cpu_count() or 1) - 1, 1)

    with src.open("r", encoding="utf-8") as reader, dst.open(
        "w", encoding="utf-8"
    ) as writer, ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = (
            (line_number, line)
            for line_number, line in enumerate(reader, start=1)
            if line.strip()
        )
        chunk_iter = _iter_chunks(tasks, chunk_size=chunk_size)
        for successes, failures in tqdm(
            executor.map(_process_chunk, chunk_iter),
            desc=f"Building {dst.name}",
            unit="chunk",
        ):
            for _line_number, converted in successes:
                writer.write(converted + "\n")
                accepted += 1
            for line_number, error in failures:
                skipped += 1
                reason, _, _message = error.partition(":")
                reasons[reason] += 1
                if skipped <= 10:
                    print(
                        f"[{src.name}:{line_number}] skipped: {error}",
                        file=sys.stderr,
                    )

    return {
        "source": str(src),
        "output": str(dst),
        "accepted": accepted,
        "skipped": skipped,
        "skip_reasons": dict(reasons),
    }


def build_split(src: Path, dst: Path, *, limit: int = 0) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    skipped = 0
    reasons: Counter[str] = Counter()
    with src.open("r", encoding="utf-8") as reader, dst.open(
        "w", encoding="utf-8"
    ) as writer:
        total_lines = sum(1 for _ in reader)
        reader.seek(0)
        for line_number, line in tqdm(
            enumerate(reader, start=1),
            desc=f"Building {dst.name}",
            total=total_lines,
            unit="line",
        ):
            if not line.strip():
                continue
            if limit and accepted >= limit:
                break
            try:
                row = json.loads(line)
                converted = convert_mech_et_row_to_proof_sft(row)
                writer.write(json.dumps(converted, ensure_ascii=False) + "\n")
                accepted += 1
            except Exception as exc:
                skipped += 1
                reasons[type(exc).__name__] += 1
                if skipped <= 10:
                    print(
                        f"[{src.name}:{line_number}] skipped: {exc}",
                        file=sys.stderr,
                    )
    return {
        "source": str(src),
        "output": str(dst),
        "accepted": accepted,
        "skipped": skipped,
        "skip_reasons": dict(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO / "data/mechet_sft",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/mechet_proof_sft",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
    )
    parser.add_argument("--parallel", type=bool, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial artifact instead of failing when any row cannot compile.",
    )
    args = parser.parse_args()

    reports = []
    for split in args.splits:
        src = args.input_dir / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        if args.parallel:
            reports.append(
                build_split_parallel(
                    src,
                    args.output_dir / f"{split}.jsonl",
                    limit=args.limit,
                )
            )
        else:
            reports.append(
                build_split(
                    src,
                    args.output_dir / f"{split}.jsonl",
                    limit=args.limit,
                )
            )

    manifest = {
        "version": "mech_proof_sft_v1",
        "task_type": "mech_proof_retro",
        "answer_channel": "executor_derived",
        "splits": {
            Path(report["output"]).stem: report for report in reports
        },
        "accepted_total": sum(report["accepted"] for report in reports),
        "skipped_total": sum(report["skipped"] for report in reports),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    if manifest["skipped_total"] and not args.allow_incomplete:
        raise RuntimeError(
            "MECH_PROOF coverage is incomplete: "
            f"{manifest['skipped_total']} rows failed compilation; "
            "partial artifacts are not accepted by default"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
