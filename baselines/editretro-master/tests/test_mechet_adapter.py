from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_native_preprocessing_keeps_identity_and_stable_id():
    module = load_module(
        "prepare_mechet_data",
        REPOSITORY / "preprocess" / "prepare_mechet_data.py",
    )
    raw = (
        REPOSITORY
        / "datasets"
        / "mech_uspto_31k_full_audit100"
        / "raw"
        / "raw_train.csv"
    )
    import csv

    with raw.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    result = module.process_row(
        {
            "stable_id": row["stable_id"],
            "source_row_index": 0,
            "reaction": row[module.REACTION_COLUMN],
            "split": "train",
            "augmentation": 3,
            "seed": 33,
            "tokenization": "token",
            "dropout": 0.0,
            "shuffle": False,
            "character": False,
        }
    )
    assert result["status"] == "ok"
    assert len(result["views"]) == 3
    assert [view["augmentation_index"] for view in result["views"]] == [0, 1, 2]
    assert {view["stable_id"] for view in result["views"]} == {row["stable_id"]}
    assert len({view["product_canonical"] for view in result["views"]}) == 1
    assert len({view["precursor_canonical"] for view in result["views"]}) == 1


def test_native_preprocessing_retains_unaligned_structural_fragment():
    module = load_module(
        "prepare_mechet_data_unaligned",
        REPOSITORY / "preprocess" / "prepare_mechet_data.py",
    )
    reaction = (
        "[CH3:1][CH2:2][CH2:3][OH:4].[Na+:5]>>"
        "[CH3:1][CH2:2][CH2:3][OH:4]"
    )
    result = module.process_row(
        {
            "stable_id": "mapping-anomaly",
            "source_row_index": 0,
            "reaction": reaction,
            "split": "train",
            "augmentation": 2,
            "seed": 33,
            "tokenization": "token",
            "dropout": 0.0,
            "shuffle": False,
            "character": False,
        }
    )
    assert result["status"] == "ok"
    assert result["unaligned_precursor_fragment_count"] == 1
    assert len(result["views"]) == 2
    assert all(
        view["precursor_canonical"] == "CCCO.[Na+]" for view in result["views"]
    )


def test_export_preserves_native_ranking_and_full_denominator(tmp_path: Path):
    module = load_module(
        "export_mechet_predictions",
        REPOSITORY / "utils" / "export_mechet_predictions.py",
    )
    reference = tmp_path / "test.jsonl"
    references = [
        {
            "stable_id": "a",
            "product_mapped": "CCO",
            "precursor_mapped": "CC.O",
        },
        {
            "stable_id": "b",
            "product_mapped": "CCN",
            "precursor_mapped": "C.CN",
        },
    ]
    reference.write_text(
        "".join(json.dumps(row) + "\n" for row in references), encoding="utf-8"
    )
    line_map = tmp_path / "test.line_map.jsonl"
    metadata = [
        {"line_index": 0, "stable_id": "a", "augmentation_index": 0},
        {"line_index": 1, "stable_id": "a", "augmentation_index": 1},
        {"line_index": 2, "stable_id": "b", "augmentation_index": 0},
        {"line_index": 3, "stable_id": "b", "augmentation_index": 1},
    ]
    line_map.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    generation = tmp_path / "generation.txt"
    generation.write_text(
        """H-2\t-1.0\tnot_smiles
P-2\t-1.0 -1.0
H-0\t-1.0\tCC . O
P-0\t-0.1 -0.1
H-0\t-2.0\tC . C
P-0\t-0.2 -0.2
H-1\t-1.0\tO . CC
P-1\t-0.1 -0.1
H-1\t-2.0\tC . O
P-1\t-0.2 -0.2
H-2\t-2.0\talso_bad
P-2\t-2.0 -2.0
H-3\t-1.0\tstill_bad
P-3\t-1.0 -1.0
H-3\t-2.0\tbad_again
P-3\t-2.0 -2.0
""",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.txt"
    runtime.write_text("2.0\n", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    args = argparse.Namespace(
        reference=reference,
        line_map=line_map,
        fairseq_log=generation,
        output=output,
        checkpoint="checkpoint.pt",
        profile="full",
        augmentation=2,
        beam_size=2,
        top_n=3,
        score_alpha=0.1,
        runtime_seconds_file=runtime,
        expected_rows=2,
    )
    assert module.command_export(args) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["stable_id"] for row in rows] == ["a", "b"]
    assert len(rows[0]["candidates"]) == 3
    assert rows[0]["candidates"][0]["precursors"] == "CC.O"
    assert rows[0]["candidates"][0]["score"] == 2.0
    assert rows[0]["runtime_ms"] == 1000.0
    assert rows[0]["evaluation_status"] == "ok"
    assert rows[1]["evaluation_status"] == "no_valid_candidates"
    assert all(not candidate["precursors"] for candidate in rows[1]["candidates"])


def test_export_treats_truncated_native_decode_as_failure(tmp_path: Path):
    module = load_module(
        "export_mechet_predictions_truncated",
        REPOSITORY / "utils" / "export_mechet_predictions.py",
    )
    reference = tmp_path / "test.jsonl"
    reference.write_text(
        json.dumps({"stable_id": "a", "product_mapped": "CCO"}) + "\n",
        encoding="utf-8",
    )
    line_map = tmp_path / "test.line_map.jsonl"
    line_map.write_text(
        json.dumps(
            {"line_index": 0, "stable_id": "a", "augmentation_index": 0}
        )
        + "\n",
        encoding="utf-8",
    )
    generation = tmp_path / "generation.txt"
    generation.write_text(
        "H-0\t-1.0\tCC.O\nP-0\t-0.10000004 -0.1\n", encoding="utf-8"
    )
    output = tmp_path / "predictions.jsonl"
    args = argparse.Namespace(
        reference=reference,
        line_map=line_map,
        fairseq_log=generation,
        output=output,
        checkpoint="checkpoint.pt",
        profile="full",
        augmentation=1,
        beam_size=2,
        top_n=2,
        score_alpha=0.1,
        runtime_seconds_file=None,
        expected_rows=1,
    )
    assert module.positional_score("-0.10000004 -0.1") == -0.1
    assert module.command_export(args) == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["evaluation_status"] == "incomplete_native_beam"
    assert row["forced_error"] is True
    assert all(not candidate["prediction"] for candidate in row["candidates"])
