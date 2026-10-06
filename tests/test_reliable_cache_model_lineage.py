import pytest

from scripts.train_tool_sft import attach_arrow_length_column, validate_cache_model_lineage


def test_frozen_cache_model_revision_and_length_are_checked():
    cache = {
        "model_name_or_path": "Qwen/Qwen3-0.6B",
        "model_revision": "frozen-commit",
        "max_length": 4096,
    }
    validate_cache_model_lineage(
        cache, model_name="Qwen/Qwen3-0.6B", revision="frozen-commit", max_length=4096
    )
    with pytest.raises(ValueError, match="model_revision"):
        validate_cache_model_lineage(
            cache, model_name="Qwen/Qwen3-0.6B", revision="other-commit", max_length=4096
        )
    with pytest.raises(ValueError, match="max_length"):
        validate_cache_model_lineage(
            cache, model_name="Qwen/Qwen3-0.6B", revision="frozen-commit", max_length=2048
        )


def test_arrow_vectorized_length_column_preserves_sampler_lengths():
    from datasets import Dataset, concatenate_datasets
    from transformers.trainer_pt_utils import LengthGroupedSampler

    data = concatenate_datasets([
        Dataset.from_dict({"input_ids": [[1, 2, 3], [4]], "labels": [[1, 2, 3], [4]]}),
        Dataset.from_dict({"input_ids": [[5, 6]], "labels": [[5, 6]]}),
    ])
    with_lengths = attach_arrow_length_column(data)
    assert with_lengths["length"] == [3, 1, 2]
    assert with_lengths["input_ids"] == data["input_ids"]
    sampler = LengthGroupedSampler(2, dataset=with_lengths, lengths=with_lengths["length"])
    assert sampler.lengths == [3, 1, 2]
