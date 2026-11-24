"""Unit tests for NMI elevation scripts: URSA export, Test-Time Scaling, Emergent Strategy decoding, and panel freezing."""
import json
from pathlib import Path

from scripts.analyze_emergent_strategy import compute_test_time_scaling, decode_emergent_strategies
from scripts.export_retroe_to_ursa import convert_report_to_ursa_route, export_groups_dir
from scripts.freeze_elevation_panels import freeze_cbyd_stratified, freeze_steerable_sar_panel


def _sample_report(task_id: str, calls: int, min_pr18: float, convergent: bool = True):
    rxns = [
        {"occurrence": "0", "product": "CC(=O)Nc1ccccc1", "precursors": ["CC(=O)O", "Nc1ccccc1"], "actions": ["FLOW_1", "FINISH"]},
        {"occurrence": "0.0", "product": "CC(=O)O", "precursors": ["CCO"], "actions": ["FLOW_1", "FLOW_2", "FINISH"]},
    ]
    if convergent:
        rxns.append({"occurrence": "0.1", "product": "Nc1ccccc1", "precursors": ["O=[N+]([O-])c1ccccc1"], "actions": ["FLOW_1", "FINISH"]})
    cands = [{"request_id": f"r{i}", "teacher_score": min_pr18 if i == 0 else 0.5} for i in range(calls)]
    return {
        "task_id": task_id,
        "root": "CC(=O)Nc1ccccc1",
        "status": "stock_solved",
        "stock_leaves": ["CCO", "O=[N+]([O-])c1ccccc1"],
        "reactions": rxns,
        "reward_evidence": {"structural_stock_closed": True},
        "stepwise": {
            "total_calls": calls,
            "selected_path": ["r0"],
            "slates": [{"key": "s0", "candidates": cands}],
        },
    }


def test_ursa_export_and_forward_steps(tmp_path: Path):
    gdir = tmp_path / "groups"
    (gdir / "t1").mkdir(parents=True)
    rep = _sample_report("t1", calls=12, min_pr18=0.2)
    (gdir / "t1" / "scored_group.json").write_text(json.dumps(rep))

    ursa_route, steps = convert_report_to_ursa_route(rep)
    assert ursa_route["schema_version"] == "2"
    assert len(ursa_route["target"]["product_of"]["reactants"]) == 2
    assert len(steps) == 3

    summary = export_groups_dir(gdir, tmp_path / "out", label="demo")
    assert summary["exported_closed_routes"] == 1
    assert summary["exported_forward_steps"] == 3


def test_scaling_and_emergent_strategy():
    reps = [
        _sample_report("t1", calls=10, min_pr18=0.3, convergent=True),
        _sample_report("t2", calls=40, min_pr18=-0.6, convergent=False),
    ]
    scaling = compute_test_time_scaling(reps, "Retro-E-Plan", budgets=(8, 16, 64))
    by_b = {r["budget"]: r for r in scaling}
    assert by_b[8]["solv0_count"] == 0
    assert by_b[16]["solv0_count"] == 1 and by_b[16]["mv_solv_count"] == 1
    assert by_b[64]["solv0_count"] == 2 and by_b[64]["mv_solv_count"] == 1

    strat = decode_emergent_strategies(reps, "Retro-E-Plan")
    assert strat["closed_routes"] == 2
    assert strat["convergent_routes"] == 1
    assert strat["mean_flow_actions_per_step"] > 1.0


def test_freeze_cbyd_and_sar_panels(tmp_path: Path):
    cbyd = tmp_path / "cbyd.jsonl"
    def make_chain(depth):
        node = {"type": "mol", "smiles": "CC"}
        for _ in range(depth):
            node = {"type": "mol", "children": [{"type": "reaction", "children": [node]}]}
        return node
    rows = []
    for i in range(6):
        n = make_chain(6 if i < 2 else (11 if i < 4 else 16))
        n["id"] = f"np_{i}"
        n["target"] = "c1ccccc1"
        rows.append(json.dumps(n))
    cbyd.write_text("\n".join(rows) + "\n")

    cmeta = freeze_cbyd_stratified(cbyd, tmp_path / "frozen", per_stratum=1)
    assert cmeta["total"] == 3
    assert cmeta["strata"] == {"lls_5_9": 1, "lls_10_14": 1, "lls_ge_15": 1}

    sar_src = tmp_path / "drugs.jsonl"
    sar_src.write_text("\n".join([
        json.dumps({"task_id": "d1", "smiles": "Cc1ccccc1"}),
        json.dumps({"task_id": "d2", "smiles": "CCc1ccccc1"}),
        json.dumps({"task_id": "d3", "smiles": "Oc1ccccc1"}),
    ]) + "\n")
    smeta = freeze_steerable_sar_panel([sar_src], tmp_path / "frozen", num_series=1, max_per_series=3)
    assert smeta["total_molecules"] == 3
    assert smeta["num_series"] == 1
