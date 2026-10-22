import json
from pathlib import Path


def test_stereo_failure_review_preserves_both_states_and_never_relabels_it_as_success():
    from scripts.review_pr96_quarantine import reproduce_failure
    legacy, _ = json.loads((Path(__file__).parent / "fixtures/mech31k_main_only_stereo_pairs.json").read_text())[0]
    result = reproduce_failure(legacy)
    assert result["source_id"] == "313"
    assert result["classification"] == "endpoint_stereo_mismatch"
    assert result["strict_chemical_equal"] is False
    assert result["source_state"] and result["executed_state"]
    assert result["source_state"] != result["executed_state"]
    assert result["training_allowed"] is False


def test_quarantine_review_refuses_to_discard_a_now_replayable_reference():
    from scripts.review_pr96_quarantine import reproduce_failure
    from mechet.inverse_trace_data import build_inverse_tool_sft_row
    from test_mech31k_main_product_root import forward_sn2
    import pytest
    source = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")
    with pytest.raises(ValueError, match="NOW_REPLAYABLE"):
        reproduce_failure(source)
