"""Held-out generation never consumes GT and never drops rejected roots."""
from copy import deepcopy
import json

import pytest

from test_pr96_evaluation import OrderedBackend
from test_pr96_history import state_rows


def root():
    from mechet.pr96_evaluation import public_root
    node = public_root({"target_smiles": "CO"})
    return {"source_id": "toy", "product_smiles": "CO",
            "initial_mapped_product_state": node.state}


def test_test_rollout_uses_only_main_product_and_its_own_import_flow_finish():
    from scripts.evaluate_pr96_test import rollout
    calls = [r["messages"][2]["tool_calls"][0]["function"] for r in state_rows()]
    backend = OrderedBackend(calls)
    journal = []
    values = rollout([root()], backend, compact_history=True, journal=journal.append)
    value = values[0]
    assert value["terminal"]
    assert [a["name"] for a in value["actions"]] == ["import_fragments", "apply_electron_flow", "finish_trace"]
    assert "[Br-]" not in backend.received[0][1]["content"]
    assert len(backend.received) == 3  # no hidden teacher-forced inference
    assert all("TRAJECTORY HISTORY" in m[1]["content"] for m in backend.received)
    assert [r["depth"] for r in journal if r["mode"] == "closed_loop"] == [0, 1, 2]
    assert journal[-1]["mode"] == "episode_complete"
    assert not any("gt_endpoint" in r for r in journal)


def test_public_contract_rejects_hidden_gold_and_changed_private_root():
    from scripts.evaluate_pr96_test import rollout
    bad = root(); bad["gt_endpoint"] = "CBr.CO"
    with pytest.raises(ValueError, match="PUBLIC_FIELDS"):
        rollout([bad], OrderedBackend([]), compact_history=False)
    bad = root(); bad["initial_mapped_product_state"] += ".[Br-:99]"
    with pytest.raises(ValueError, match="PUBLIC_ROOT"):
        rollout([bad], OrderedBackend([]), compact_history=False)
    bad = root(); bad["product_smiles"] = "COC.[Br-]"
    with pytest.raises(ValueError, match="MAIN_ONLY"):
        rollout([bad], OrderedBackend([]), compact_history=False)


def test_truncation_stays_in_full_denominator_and_is_immediately_journaled():
    from scripts.evaluate_pr96_test import rollout, aggregate_results
    class Truncated:
        def generate(self, messages):
            return [{"raw": "<tool_call>", "truncated": True} for _ in messages]
    journal = []
    values = rollout([root()], Truncated(), compact_history=False, journal=journal.append)
    assert len(values) == 1 and values[0]["error"] == "OUTPUT_TRUNCATED"
    assert values[0]["generated_decisions"] == 1
    assert journal[-1]["reaction_id"] == "toy"
    scored = [{**values[0], "structural_exact": False, "full_exact": False}]
    metrics = aggregate_results([scored], ["toy"])
    assert metrics["denominator"] == 1 and metrics["structural_hits"] == 0
    assert metrics["failure_reasons"] == {"OUTPUT_TRUNCATED": 1}
    with pytest.raises(ValueError, match="IDS"):
        aggregate_results([scored, scored], ["toy"])
    with pytest.raises(ValueError, match="IDS"):
        aggregate_results([scored], ["toy", "missing"])


def test_endpoint_scoring_preserves_all_components_charge_and_stereo():
    from scripts.evaluate_pr96_test import score_endpoint
    base = {"reaction_id": "one", "target": "CO", "initial_state": "[CH3:1][OH:2]",
            "terminal": True, "predicted_state": "[CH3:1][OH:2].[Cl-:3]"}
    gold = {"target_smiles": "[CH3:50][OH:60].[Br-:70]",
            "full_precursor_state": "[Cl-:80].[OH:60][CH3:50]"}
    score = score_endpoint(base, gold)
    assert score["structural_exact"] and score["full_exact"]
    missing = deepcopy(base); missing["predicted_state"] = "[CH3:1][OH:2]"
    score = score_endpoint(missing, gold)
    assert score["structural_exact"] and not score["full_exact"]
    neutral = deepcopy(base); neutral["predicted_state"] = "[CH3:1][OH:2].[Cl:3]"
    assert not score_endpoint(neutral, gold)["full_exact"]
    opposite = {"reaction_id": "stereo", "target": "FC(Cl)Br", "initial_state": "[F:1][CH:2]([Cl:3])[Br:4]",
                "terminal": True, "predicted_state": "[F:1][C@@H:2]([Cl:3])[Br:4]"}
    stereo_gold = {"target_smiles": "[F:11][CH:12]([Cl:13])[Br:14]",
                   "full_precursor_state": "[F:11][C@H:12]([Cl:13])[Br:14]"}
    assert not score_endpoint(opposite, stereo_gold)["full_exact"]
    stopped = deepcopy(base); stopped["terminal"] = False
    assert not score_endpoint(stopped, gold)["full_exact"]
