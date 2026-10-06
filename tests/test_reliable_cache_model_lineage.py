import pytest

from scripts.train_tool_sft import validate_cache_model_lineage


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
