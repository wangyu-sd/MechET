#!/usr/bin/env python3
"""Decode ReactSeq beams and export MechET-compatible top-10 JSONL.

The native ReactSeq test-time augmentation vote is retained: each decoded
precursor receives ``1 / beam_rank**2`` from every augmented product view.
Votes are accumulated in a temporary SQLite database so full FlowER inference
does not require keeping millions of decoded candidates in memory.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} is not an object".format(path, line_number))
            yield value


def stable_id(row: Dict[str, Any]) -> str:
    value = str(row.get("id") or row.get("stable_id") or "").strip()
    if not value:
        raise ValueError("reference row missing id/stable_id")
    return value


def product_smiles(row: Dict[str, Any]) -> str:
    return str(
        row.get("product_mapped")
        or row.get("target_smiles")
        or row.get("product_unmapped")
        or ""
    ).strip()


def precursor_smiles(row: Dict[str, Any]) -> str:
    return str(
        row.get("precursor_mapped")
        or row.get("structural_precursor")
        or row.get("expected_precursor")
        or row.get("precursor_unmapped")
        or ""
    ).strip()


def _decode_one(value: Tuple[str, str]) -> str:
    source, prediction = value
    try:
        from e_smiles import merge_smiles

        return str(merge_smiles(source + ">>>" + prediction) or "").strip()
    except Exception:
        return ""


def _canonical_key(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        from rdkit import Chem

        parts: List[str] = []
        for fragment in smiles.split("."):
            mol = Chem.MolFromSmiles(fragment.strip())
            if mol is None:
                return "RAW:" + smiles
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
                if atom.HasProp("molAtomMapNumber"):
                    atom.ClearProp("molAtomMapNumber")
            parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        return "CANON:" + ".".join(sorted(parts))
    except Exception:
        return "RAW:" + smiles


def _decode_view(value: Tuple[str, List[str]]) -> List[Tuple[str, str]]:
    source, raw_predictions = value
    decoded: List[Tuple[str, str]] = []
    for raw_prediction in raw_predictions:
        prediction = _decode_one((source, raw_prediction))
        decoded.append((prediction, _canonical_key(prediction)))
    return decoded


def _next_nonempty(handle: Any, label: str) -> str:
    line = handle.readline()
    if line == "":
        raise ValueError("{} ended early".format(label))
    return line.rstrip("\n")


def _initialize_votes(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA page_size = 65536")
    connection.execute("PRAGMA journal_mode = MEMORY")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -262144")
    connection.execute(
        """
        CREATE TABLE votes (
          stable_id TEXT NOT NULL,
          candidate_key TEXT NOT NULL,
          prediction TEXT NOT NULL,
          score REAL NOT NULL,
          first_seen INTEGER NOT NULL,
          PRIMARY KEY (stable_id, candidate_key)
        ) WITHOUT ROWID
        """
    )


def _add_votes(
    connection: sqlite3.Connection,
    rows: List[Tuple[str, str, str, float, int]],
) -> None:
    connection.executemany(
        """
        INSERT INTO votes(stable_id, candidate_key, prediction, score, first_seen)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(stable_id, candidate_key) DO UPDATE
        SET score = votes.score + excluded.score
        """,
        rows,
    )


def command_export(args: argparse.Namespace) -> int:
    references: List[Dict[str, Any]] = list(read_jsonl(args.reference))
    reference_ids = [stable_id(row) for row in references]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("reference contains duplicate stable IDs")
    reference_id_set = set(reference_ids)
    view_counts = {identifier: 0 for identifier in reference_ids}

    temporary = tempfile.NamedTemporaryFile(
        prefix="reactseq_votes_", suffix=".sqlite", delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    started = time.time()
    decoded_count = 0
    view_count = 0

    connection = sqlite3.connect(str(temporary_path))
    _initialize_votes(connection)
    pool: Optional[mp.pool.Pool] = None
    if args.workers > 1:
        pool = mp.Pool(args.workers)
    try:
        with args.src.open(encoding="utf-8") as src_handle, args.line_map.open(
            encoding="utf-8"
        ) as map_handle, args.predictions.open(encoding="utf-8") as prediction_handle:
            map_iterator = (line for line in map_handle if line.strip())
            views_since_commit = 0
            while True:
                batch_metadata: List[str] = []
                batch_work: List[Tuple[str, List[str]]] = []
                for _ in range(args.view_batch_size):
                    try:
                        map_line = next(map_iterator)
                    except StopIteration:
                        break
                    metadata = json.loads(map_line)
                    identifier = str(metadata.get("stable_id") or "")
                    if not identifier:
                        raise ValueError("line_map row is missing stable_id")
                    if identifier not in reference_id_set:
                        raise ValueError(
                            "line_map contains unknown stable_id: {}".format(identifier)
                        )
                    source = _next_nonempty(src_handle, "src").replace(" ", "")
                    raw_predictions = [
                        _next_nonempty(prediction_handle, "predictions").replace(
                            " ", ""
                        )
                        for _ in range(args.n_best)
                    ]
                    batch_metadata.append(identifier)
                    batch_work.append((source, raw_predictions))

                if not batch_work:
                    break
                if pool is not None:
                    decoded_batch = pool.map(
                        _decode_view, batch_work, chunksize=args.pool_chunksize
                    )
                else:
                    decoded_batch = list(map(_decode_view, batch_work))

                vote_rows: List[Tuple[str, str, str, float, int]] = []
                for batch_index, (identifier, decoded) in enumerate(
                    zip(batch_metadata, decoded_batch)
                ):
                    view_counts[identifier] += 1
                    first_seen_base = decoded_count + batch_index * args.n_best
                    for beam_index, (prediction, key) in enumerate(decoded):
                        if not key:
                            continue
                        vote_rows.append(
                            (
                                identifier,
                                key,
                                prediction,
                                1.0 / float((beam_index + 1) ** 2),
                                first_seen_base + beam_index,
                            )
                        )
                _add_votes(connection, vote_rows)

                batch_views = len(batch_work)
                decoded_count += batch_views * args.n_best
                view_count += batch_views
                views_since_commit += batch_views
                if views_since_commit >= args.commit_every:
                    connection.commit()
                    views_since_commit = 0
                if view_count % args.progress_every < batch_views:
                    elapsed = time.time() - started
                    rate = view_count / elapsed if elapsed else 0.0
                    remaining = (
                        (args.expected_views - view_count) / rate
                        if args.expected_views and rate
                        else 0.0
                    )
                    print(
                        "Processed {:,} views ({:.1f} views/s), ETA {:.2f} h".format(
                            view_count, rate, max(remaining, 0.0) / 3600.0
                        ),
                        flush=True,
                    )

            if src_handle.readline() != "":
                raise ValueError("src has more rows than line_map")
            if prediction_handle.readline() != "":
                raise ValueError("predictions has more rows than line_map*n_best")
        connection.commit()

        if args.expected_views is not None and view_count != args.expected_views:
            raise ValueError(
                "Expected {} augmented views, found {}".format(
                    args.expected_views, view_count
                )
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        elapsed_ms = (time.time() - started) * 1000.0
        postprocess_runtime_per_target = elapsed_ms / max(len(references), 1)
        rows_with_candidates = 0
        input_preprocessing_failures = 0
        inverse_conversion_failures = 0
        with args.output.open("w", encoding="utf-8") as output_handle:
            for reference in references:
                identifier = stable_id(reference)
                selected = connection.execute(
                    """
                    SELECT prediction, score
                    FROM votes
                    WHERE stable_id = ?
                    ORDER BY score DESC, first_seen ASC
                    LIMIT ?
                    """,
                    (identifier, args.top_n),
                ).fetchall()
                if view_counts[identifier] == 0:
                    evaluation_status = "input_preprocessing_failure"
                    input_preprocessing_failures += 1
                elif not selected:
                    evaluation_status = "inverse_conversion_failure"
                    inverse_conversion_failures += 1
                else:
                    evaluation_status = "ok"
                    rows_with_candidates += 1
                candidates: List[Dict[str, Any]] = []
                for rank, (prediction, score) in enumerate(selected, 1):
                    candidates.append(
                        {
                            "rank": rank,
                            "precursors": prediction,
                            "prediction": prediction,
                            "score": float(score),
                        }
                    )
                while len(candidates) < args.top_n:
                    rank = len(candidates) + 1
                    candidates.append(
                        {"rank": rank, "precursors": "", "prediction": "", "score": 0.0}
                    )
                row = {
                    "id": identifier,
                    "stable_id": identifier,
                    "product": product_smiles(reference),
                    "reference_precursors": precursor_smiles(reference),
                    "candidates": candidates,
                    "runtime_ms": args.inference_runtime_ms_per_target,
                    "postprocess_runtime_ms": postprocess_runtime_per_target,
                    "source_method": "ReactSeq",
                    "checkpoint": args.checkpoint,
                    "ranking": "native_tta_vote_sum_inverse_beam_rank_squared",
                    "evaluation_status": evaluation_status,
                    "forced_error": evaluation_status != "ok",
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        report = {
            "reference_rows": len(references),
            "augmented_views": view_count,
            "decoded_beams": decoded_count,
            "evaluation_denominator_rows": len(references),
            "reference_ids_with_views": sum(value > 0 for value in view_counts.values()),
            "reference_ids_without_views": sum(value == 0 for value in view_counts.values()),
            "rows_with_candidates": rows_with_candidates,
            "input_preprocessing_failures": input_preprocessing_failures,
            "inverse_conversion_failures": inverse_conversion_failures,
            "forced_error_rows": input_preprocessing_failures + inverse_conversion_failures,
            "failures_counted_as_errors": True,
            "reference_ids_filtered_from_evaluation": False,
            "n_best": args.n_best,
            "top_n": args.top_n,
            "output": str(args.output.resolve()),
            "checkpoint": args.checkpoint,
            "inference_runtime_ms_per_target": args.inference_runtime_ms_per_target,
            "postprocess_runtime_ms_total": elapsed_ms,
        }
        report_path = args.output.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--line-map", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-best", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--view-batch-size", type=int, default=1000)
    parser.add_argument("--pool-chunksize", type=int, default=10)
    parser.add_argument("--commit-every", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--expected-views", type=int)
    parser.add_argument("--inference-runtime-ms-per-target", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    positive_values = {
        "--n-best": args.n_best,
        "--top-n": args.top_n,
        "--workers": args.workers,
        "--view-batch-size": args.view_batch_size,
        "--pool-chunksize": args.pool_chunksize,
        "--commit-every": args.commit_every,
        "--progress-every": args.progress_every,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        parser.error("{} must be positive".format(", ".join(invalid)))
    if args.expected_views is not None and args.expected_views <= 0:
        parser.error("--expected-views must be positive")
    return command_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
