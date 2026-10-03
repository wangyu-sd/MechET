from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/merge_iclr_sampled_shards.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference = tmp_path / "reference.jsonl"
    generation = tmp_path / "generation"
    output = tmp_path / "merged"
    generation.mkdir()
    _write_jsonl(reference, [{"id": f"r{index}", "answer": "private"} for index in range(4)])
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    for shard in range(2):
        path = generation / f"predictions.shard-{shard:03d}.jsonl"
        rows = [
            {
                "id": f"r{index}",
                "prediction_mode": "direct",
                "model": {
                    "adapter_sha256": "adapter-hash",
                    "model_revision": "revision",
                    "seed": 17,
                    "samples_per_target": 2,
                },
                "candidates": [{"sample_index": 0}, {"sample_index": 1}],
            }
            for index in range(shard, 4, 2)
        ]
        _write_jsonl(path, rows)
        (generation / f"predictions.shard-{shard:03d}.jsonl.manifest.json").write_text(
            json.dumps(
                {
                    "shard_index": shard,
                    "shard_count": 2,
                    "data_sha256": reference_sha,
                    "n_predictions_written": 1,
                    "n_predictions_skipped_by_resume": 1,
                    "mode": "direct",
                    "condition_name": "frozen",
                    "model_revision": "revision",
                    "tokenizer_revision": "revision",
                    "adapter_sha256": "adapter-hash",
                    "seed": 17,
                    "backend": "vllm",
                }
            ),
            encoding="utf-8",
        )
    return reference, generation, output


def _run(reference: Path, generation: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--generation-dir",
            str(generation),
            "--output-dir",
            str(output),
            "--baseline",
            "outcome_only",
            "--expected-rows",
            "4",
            "--k",
            "2",
            "--shards",
            "2",
            "--evaluation-scope",
            "nmi_h2",
            "--direct-sample-batch-size",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_merge_preserves_reference_order_and_resume_counts(tmp_path: Path) -> None:
    reference, generation, output = _fixture(tmp_path)
    completed = _run(reference, generation, output)
    assert completed.returncode == 0, completed.stderr
    merged = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in merged] == ["r0", "r1", "r2", "r3"]
    assert json.loads((output / "manifest.json").read_text())["n_candidates"] == 8
    repeated = _run(reference, generation, output)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr


def test_merge_rejects_candidate_model_lineage_mismatch(tmp_path: Path) -> None:
    reference, generation, output = _fixture(tmp_path)
    path = generation / "predictions.shard-001.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["model"]["adapter_sha256"] = "wrong-adapter"
    _write_jsonl(path, rows)
    completed = _run(reference, generation, output)
    assert completed.returncode != 0
    assert "row/model lineage differs" in completed.stderr
    assert not (output / "predictions.jsonl").exists()


def test_merge_prepartitioned_trace_shards(tmp_path: Path) -> None:
    reference, generation, output = _fixture(tmp_path)
    full_reference_sha = "full-test-hash"
    for shard in range(2):
        path = generation / f"predictions.shard-{shard:03d}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["prediction_mode"] = "trace"
        _write_jsonl(path, rows)
        manifest_path = generation / f"predictions.shard-{shard:03d}.jsonl.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update({
            "shard_index": 0,
            "shard_count": 1,
            "data_sha256": full_reference_sha,
            "mode": "trace",
        })
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--reference", str(reference),
            "--generation-dir", str(generation),
            "--output-dir", str(output),
            "--baseline", "closed_loop",
            "--expected-rows", "4", "--k", "2", "--shards", "2",
            "--evaluation-scope", "nmi_h2",
            "--prediction-mode", "trace",
            "--shard-layout", "prepartitioned",
            "--manifest-data-sha256", full_reference_sha,
            "--task-shard-count", "2", "--task-shard-index", "1",
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    merged = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in merged] == ["r0", "r1", "r2", "r3"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["artifact_type"] == "nmi_h2_closed_loop_sampled_test_manifest"
    assert manifest["task_shard_index"] == 1


def test_validate_only_checks_rows_without_writing_output(tmp_path: Path) -> None:
    reference, generation, output = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--reference", str(reference),
            "--generation-dir", str(generation),
            "--output-dir", str(output),
            "--baseline", "outcome_only",
            "--expected-rows", "4", "--k", "2", "--shards", "2",
            "--evaluation-scope", "nmi_h2",
            "--validate-only",
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validation_only"] is True
    assert not output.exists()
