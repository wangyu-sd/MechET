import json
from pathlib import Path
import pytest


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


def test_missing_source_bond_review_records_actual_graph_and_original_arrow_without_relabeling():
    from scripts.review_pr96_quarantine import reproduce_failure
    source = json.loads((Path(__file__).parent / "fixtures/pr96_missing_source_bond.json").read_text())
    result = reproduce_failure(source)
    assert result["source_id"] == "flower_mech_proof_test_21379"
    assert result["classification"] == "source_bond_absent_after_public_prefix_replay"
    assert result["source_state"] and result["executed_state"]
    assert result["strict_chemical_equal"] is False
    assert result["context"]["missing_runtime_source_bonds"] == [[2, 14]]
    assert result["context"]["original_source_moves"]
    assert result["training_allowed"] is False


@pytest.mark.parametrize("workers", [1, 2])
def test_partial_quarantine_review_preserves_raw_states_and_progress_but_never_publishes_approval(tmp_path, monkeypatch, workers):
    import hashlib
    import pytest
    from mechet.inverse_trace_data import build_inverse_tool_sft_row
    from test_mech31k_main_product_root import forward_sn2
    from scripts import review_pr96_quarantine as module
    failed, _ = json.loads((Path(__file__).parent / "fixtures/mech31k_main_only_stereo_pairs.json").read_text())[0]
    replayable = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")
    rows = [{"source_row": row, "source_row_sha256": hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
             "error": "ValueError:natural-language trace endpoint changed"} for row in (failed, replayable)]
    source = tmp_path / "unresolved.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {"unresolved_path": str(source), "unresolved_sha256": digest,
              "unresolved_ids": [str(failed["source_id"]), str(replayable["source_id"])]}
    (tmp_path / "manifest.json").write_text(json.dumps({"track": "mech31k", "reports": {"train": report}}))
    output = tmp_path / "review.json"
    monkeypatch.setattr("sys.argv", ["review", "--data-dir", str(tmp_path), "--output", str(output), "--workers", str(workers)])
    with pytest.raises(ValueError, match="NOW_REPLAYABLE"):
        module.main()
    assert not output.exists()
    cases = [json.loads(l) for l in (tmp_path / "review.cases.jsonl").read_text().splitlines()]
    assert len(cases) == 1 and cases[0]["source_state"] and cases[0]["executed_state"]
    assert cases[0]["source_id"] == "313" and cases[0]["training_allowed"] is False
    progress = json.loads((tmp_path / "review.progress.jsonl").read_text().splitlines()[-1])
    assert progress["processed"] == 1 and progress["total"] == 2 and progress["percent"] == 50
