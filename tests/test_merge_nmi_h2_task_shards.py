import hashlib
import json
from pathlib import Path

import pytest

from scripts.merge_nmi_h2_task_shards import merge_task_shards


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    reference = tmp_path / "test.jsonl"
    rows = [
        {"id": f"r{index}", "source_id": f"source-{index}", "target_smiles": "C"}
        for index in range(5)
    ]
    reference.write_text("".join(json.dumps(row) + "\n" for row in rows))
    full_sha = _sha(reference)
    directories = [tmp_path / f"task-{index}" for index in range(2)]
    model = {
        "adapter_sha256": "adapter-sha",
        "model_revision": "model-revision",
        "samples_per_target": 2,
        "seed": 17,
        "data_sha256": full_sha,
    }
    for shard_index, directory in enumerate(directories):
        directory.mkdir()
        selected = rows[shard_index::2]
        selected_reference = tmp_path / f"selected-{shard_index}.jsonl"
        selected_reference.write_text("".join(json.dumps(row) + "\n" for row in selected))
        predictions = directory / "predictions.jsonl"
        predictions.write_text("".join(json.dumps({
            "id": row["id"],
            "source_id": row["source_id"],
            "target_smiles": row["target_smiles"],
            "prediction_mode": "trace",
            "condition_name": "nmi_h2_closed_loop_seed17_k2",
            "candidates": [{"sample_index": 0}, {"sample_index": 1}],
            "model": model,
        }) + "\n" for row in selected))
        (directory / "manifest.json").write_text(json.dumps({
            "artifact_type": "nmi_h2_closed_loop_sampled_test_manifest",
            "task_shard_count": 2,
            "task_shard_index": shard_index,
            "n_targets": len(selected),
            "samples_per_target": 2,
            "n_candidates": 2 * len(selected),
            "reference_sha256": _sha(selected_reference),
            "shard_data_sha256": full_sha,
            "predictions_sha256": _sha(predictions),
            "shard_adapter_sha256": "adapter-sha",
            "shard_model_revision": "model-revision",
        }))
    return reference, directories, tmp_path / "merged"


def test_merges_complete_task_shards_in_original_reaction_order(tmp_path: Path) -> None:
    reference, directories, output = _fixture(tmp_path)
    result = merge_task_shards(
        reference, directories[::-1], output,
        expected_rows=5, k=2, task_shard_count=2,
    )
    assert result["n_targets"] == 5
    assert result["n_candidates"] == 10
    assert result["predictions_sha256"] == _sha(output / "predictions.jsonl")
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in predictions] == [f"r{index}" for index in range(5)]
    assert json.loads((output / "manifest.json").read_text())["reference_sha256"] == _sha(reference)


def test_validate_only_checks_without_creating_output(tmp_path: Path) -> None:
    reference, directories, output = _fixture(tmp_path)
    result = merge_task_shards(
        reference, directories, output,
        expected_rows=5, k=2, task_shard_count=2, validate_only=True,
    )
    assert result["validation_only"] is True
    assert not output.exists()


@pytest.mark.parametrize("corruption", [
    "duplicate_index", "wrong_adapter", "missing_row", "wrong_reference",
    "wrong_product", "wrong_condition", "wrong_model_reference",
])
def test_rejects_incomplete_or_mismatched_task_shards(tmp_path: Path, corruption: str) -> None:
    reference, directories, output = _fixture(tmp_path)
    manifest_path = directories[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "duplicate_index":
        manifest["task_shard_index"] = 0
    elif corruption == "wrong_adapter":
        manifest["shard_adapter_sha256"] = "different-adapter"
    elif corruption == "missing_row":
        predictions = directories[1] / "predictions.jsonl"
        predictions.write_text(predictions.read_text().splitlines(keepends=True)[0])
        manifest["predictions_sha256"] = _sha(predictions)
    elif corruption == "wrong_reference":
        manifest["reference_sha256"] = "wrong-selected-reference"
    elif corruption in ("wrong_product", "wrong_condition", "wrong_model_reference"):
        predictions = directories[1] / "predictions.jsonl"
        rows = [json.loads(line) for line in predictions.read_text().splitlines()]
        if corruption == "wrong_product":
            rows[0]["target_smiles"] = "O"
        elif corruption == "wrong_condition":
            rows[0]["condition_name"] = "different-condition"
        else:
            rows[0]["model"]["data_sha256"] = "wrong-reference"
        predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["predictions_sha256"] = _sha(predictions)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        merge_task_shards(
            reference, directories, output,
            expected_rows=5, k=2, task_shard_count=2,
        )
    assert not (output / "predictions.jsonl").exists()
