"""Distributed endpoint denominator checks for the opt-in hybrid."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from mechet.crn_graph_bridge import file_sha256
from scripts.summarize_crn_qwen_hybrid import aggregate
from scripts.run_natural_language_value_search import read_selected


def test_all_shards_and_official_unsupported_denominator() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "test.jsonl"
        rows = [{"id": f"test_{i}"} for i in range(5)]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        digest = file_sha256(source)
        selected_all = read_selected(source, 5, 17)
        baseline = root / "baseline"
        baseline.mkdir()
        for rank in range(2):
            contract = {
                "source": str(source.resolve()), "sample_reactions": 5,
                "seed": 17, "rank": rank, "world": 2,
                "source_sha256": digest,
            }
            (root / f"contract.shard-{rank:02d}-of-02.json").write_text(json.dumps(contract))
            selected = selected_all[rank::2]
            (root / f"results.shard-{rank:02d}-of-02.jsonl").write_text(
                "".join(json.dumps({"id": row["id"],
                                    "top1_structural_exact": row["id"] == "test_0",
                                    "top_terminal": True, "rejected": {},
                                    "attempts": [{"name": "import_fragments",
                                                  "arguments": {"fragment": row["id"]}}]}) + "\n"
                        for row in selected)
            )
            (baseline / f"results.shard-{rank:02d}-of-02.jsonl").write_text(
                "".join(json.dumps({"id": row["id"],
                                    "top1_structural_exact": row["id"] == "test_1",
                                    "attempts": [{"name": "import_fragments",
                                                  "arguments": {"fragment": (
                                                      "other" if row["id"] == "test_2"
                                                      else row["id"])}}]}) + "\n"
                        for row in selected)
            )
        report = aggregate(source=source, output=root, sample_reactions=5,
                           seed=17, world=2, source_sha256=digest,
                           official_denominator=9, baseline=baseline)
        assert report["evaluated_strict_executable"] == 5
        assert report["top1_structural_exact"] == 1
        assert report["upstream_corrupt_graph_unsupported"] == 4
        assert report["top1_structural_rate_official_with_unsupported_failures"] == 1 / 9
        assert report["paired_baseline"]["baseline_only_correct"] == 1
        assert report["paired_baseline"]["hybrid_only_correct"] == 1
        assert report["paired_baseline"]["first_call_identical"] == 4
        assert report["paired_baseline"]["first_call_different_ids"] == ["test_2"]
        assert report["paired_baseline"]["baseline_contract_verified"] is False
        for rank in range(2):
            control = json.loads(
                (root / f"contract.shard-{rank:02d}-of-02.json").read_text()
            )
            control.update(matched_v2=True, product_only_remap=True, greedy=True)
            (baseline / f"contract.shard-{rank:02d}-of-02.json").write_text(
                json.dumps(control)
            )
        verified = aggregate(source=source, output=root, sample_reactions=5,
                             seed=17, world=2, source_sha256=digest,
                             official_denominator=9, baseline=baseline)
        assert verified["paired_baseline"]["baseline_contract_verified"] is True
        changed = json.loads((baseline / "contract.shard-01-of-02.json").read_text())
        changed["source_sha256"] = "wrong"
        (baseline / "contract.shard-01-of-02.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="differs on source_sha256"):
            aggregate(source=source, output=root, sample_reactions=5,
                      seed=17, world=2, source_sha256=digest, baseline=baseline)
        (root / "results.shard-01-of-02.jsonl").write_text("")
        with pytest.raises(ValueError, match="incomplete evaluation"):
            aggregate(source=source, output=root, sample_reactions=5,
                      seed=17, world=2, source_sha256=digest)


def test_official_four_rows_are_explicitly_accounted_for() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        strict = root / "strict_test.jsonl"
        strict_rows = [
            {"id": f"strict_{i}", "source_id": f"flower_mech_proof_test_{i}"}
            for i in range(1, 6)
        ]
        strict.write_text("".join(json.dumps(row) + "\n" for row in strict_rows))
        official = root / "official_test.jsonl"
        official_rows = [
            {"id": f"official_{i}", "source_id": str(i)} for i in range(1, 6)
        ] + [
            {"id": f"official_{name}", "source_id": name}
            for name in ("PC", "PM", "RC", "RS")
        ]
        official.write_text("".join(json.dumps(row) + "\n" for row in official_rows))
        for rank in range(2):
            (root / f"contract.shard-{rank:02d}-of-02.json").write_text(json.dumps({
                "source": str(strict.resolve()), "sample_reactions": 5,
                "seed": 17, "rank": rank, "world": 2,
                "source_sha256": file_sha256(strict),
            }))
            selected = read_selected(strict, 5, 17)[rank::2]
            (root / f"results.shard-{rank:02d}-of-02.jsonl").write_text(
                "".join(json.dumps({"id": row["id"], "top1_structural_exact": False,
                                    "top_terminal": False, "rejected": {}}) + "\n"
                        for row in selected)
            )
        report = aggregate(
            source=strict, output=root, sample_reactions=5, seed=17, world=2,
            source_sha256=file_sha256(strict), official_denominator=9,
            official_source=official, official_source_sha256=file_sha256(official),
        )
        assert report["official_unsupported_source_ids"] == ["PC", "PM", "RC", "RS"]
        assert len(report["official_unsupported_records"]) == 4
