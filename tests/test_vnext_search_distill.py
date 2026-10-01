import pytest

from scripts.build_vnext_search_distill import build_rows


def test_search_teacher_compiles_to_history_without_reference_in_prompt():
    sources = [
        {"source_id": str(i), "target_smiles": "[CH4:1]",
         "expected_precursor": "[CH4:1]", "metadata": {}}
        for i in range(50)
    ]
    search = [
        {"source_id": str(i), "id": str(i), "target": "C",
         "expected_precursor": "C", "successful_full_exact": True,
         "successful_actions": [{
             "state_before": "[CH4:1]", "name": "finish_trace",
             "arguments": {}, "result": {"ok": True, "code": "PASS", "derived_precursor": "C"},
         }]}
        for i in range(50)
    ]
    rows, stats = build_rows(sources, search)
    assert stats["exact_teacher_reactions"] == 50
    assert rows["train"] and rows["valid"]
    sample = rows["train"][0]
    assert "TRAJECTORY HISTORY" in sample["messages"][1]["content"]
    assert "expected_precursor" not in sample["messages"][1]["content"]
    assert sample["metadata"]["search_private_reference_selected"]


def test_non_train_search_id_rejected():
    with pytest.raises(ValueError, match="train-only"):
        build_rows([], [{"source_id": "test", "successful_full_exact": True}])
