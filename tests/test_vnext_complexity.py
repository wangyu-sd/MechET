from scripts.audit_vnext_complexity import aggregate, reaction_features


def test_complexity_and_missing_prediction_denominator():
    row = {
        "source_id": "r1",
        "metadata": {"trace_plan": {"steps": [{
            "state_before": "[CH3:1][OH:2]", "state_after": "[CH3+:1].[OH-:2]",
            "imports": [],
            "moves": [{"source": {"kind": "BOND", "atoms": [1, 2]},
                       "sink": {"kind": "ATOM", "atoms": [2]}}],
        }]}}
    }
    assert reaction_features(row) == {
        "reacting_atoms": 2, "changed_bonds": 1, "ring_change": False,
        "electron_flows": 1, "fragment_imports": 0, "trajectory_length": 1,
    }
    missing = {**row, "source_id": "r2"}
    report = aggregate([row, missing], [{"source_id": "r1", "top1_exact": True,
                                        "top_terminal": True, "pass_at_beam": True}])
    assert report["all"]["all"] == {
        "source_reactions": 2, "evaluated": 1, "top1_endpoint_exact": 1,
        "top1_full_exact": 0, "pass_at_beam": 1, "top_terminal": 1,
    }
