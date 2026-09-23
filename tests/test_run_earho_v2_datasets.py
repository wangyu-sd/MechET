import json

import pytest

from scripts.run_earho_v2 import _dataset_contract, _sample_reactions


def test_streaming_sample_is_deterministic_and_unique(tmp_path):
    source = tmp_path / "reactions.jsonl"
    source.write_text("".join(json.dumps({"source_id": str(i)}) + "\n" for i in range(20)))
    first = _sample_reactions(source, 20, 7, 17)
    second = _sample_reactions(source, 20, 7, 17)
    assert first == second
    assert len({row["source_id"] for row in first}) == 7
    with pytest.raises(ValueError, match="row count changed"):
        _sample_reactions(source, 21, 7, 17)


def test_only_pinned_dataset_contracts_are_accepted():
    assert _dataset_contract({"dataset_id": "flower_strict_executable"})["reaction_denominator"]["train"] == 257167
    assert _dataset_contract({"dataset_id": "mech_uspto31k_current_compiler"})["reaction_denominator"]["train"] == 10152
    with pytest.raises(ValueError, match="unknown EARHO dataset_id"):
        _dataset_contract({"dataset_id": "unfiltered"})
