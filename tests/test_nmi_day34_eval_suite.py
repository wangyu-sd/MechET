"""Unit tests for NMI Day 3-4 six-dimensional evaluator and panel freezer."""
import json
from pathlib import Path
import pytest

from scripts.prepare_nmi_day34_eval_suite import freeze_panels, score_six_dimensions


def _write_group(root: Path, name: str, payload: dict):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "scored_group.json").write_text(json.dumps([payload]))


def test_six_dim_scoring_and_panel_freeze(tmp_path):
    gdir = tmp_path / "groups"
    _write_group(gdir, "t1", dict(
        task_id="t1", root="C[C@H](O)CC", status="stock_solved",
        stock_leaves=["C[C@H](O)C", "CBr"],
        reactions=[
            dict(occurrence="0", product="C[C@H](O)CC", precursors=["C[C@H](O)C", "CBr"]),
            dict(occurrence="0.0", product="C[C@H](O)C", precursors=["C[C@H](O)H"]),
            dict(occurrence="0.1", product="CBr", precursors=["CH4"]),
        ],
        reward_evidence=dict(structural_stock_closed=True),
        stepwise=dict(
            backtracks=1, selected_path=["r1"],
            slates=[dict(candidates=[dict(request_id="r1", teacher_score=0.2, teacher_z={"chemical_forward": 0.1})])],
        ),
    ))
    _write_group(gdir, "t2", dict(
        task_id="t2", root="CCCC", status="rejected_action",
        reactions=[],
        reward_evidence=dict(structural_stock_closed=False),
        stepwise=dict(
            backtracks=0, selected_path=[],
            slates=[dict(candidates=[dict(request_id="r2", accepted=False, error="ROUTE_CYCLE:parent")])],
        ),
    ))
    metrics, reports = score_six_dimensions(gdir, "test_arm")
    assert metrics["nonzero_stock_closed"] == 1
    assert metrics["mech_gated_closed"] == 1
    assert metrics["stereo_preservation_rate"] == 1.0
    assert metrics["convergent_closed"] == 1
    assert metrics["backtrack_closed"] == 1
    assert metrics["cycle_failed_targets"] == 1

    with pytest.raises(ValueError, match="Refusing to freeze"):
        freeze_panels(reports, tmp_path / "bad_out", split_role="internal_valid")

    manifest = freeze_panels(reports, tmp_path / "out", split_role="public_test")
    assert manifest["split_role"] == "public_test"
    assert manifest["failure_recovery_count"] == 1
    assert manifest["supply_disruption_count"] == 1
    assert manifest["competing_forks_count"] == 0
