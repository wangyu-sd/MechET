import hashlib

from scripts.build_vnext_reachability_value import build_rows
from mechet.successor_value import reachability_value_prompt


def test_reachability_prompt_has_budget_but_no_reference_endpoint():
    prompt = reachability_value_prompt("CO", "CO", "C.O", remaining_decisions=5, terminal=False)
    assert "REMAINING DECISIONS: 5" in prompt
    assert "reference precursor" not in prompt.lower()


def test_builder_uses_observed_outcome_not_off_reference_as_negative():
    state = "[CH3:1][OH:2]"
    anchor = hashlib.sha256(state.encode()).hexdigest()
    sources = [{"source_id": str(i), "target_smiles": state} for i in range(100)]
    rollouts = []
    for i in range(100):
        terminal = i % 2 == 0
        rollouts.append({
            "id": str(i), "kind": "rl", "anchor": {"state_hash": anchor},
            "score": {"first_successor_state": "CO", "first_successor_terminal": terminal,
                      "correct": i % 5 == 0, "failure": "DECISION_BUDGET",
                      "trajectory": [{"state_before": state}]},
        })
    rows, stats = build_rows(sources, rollouts, decision_budget=40, min_failed_rollouts=2)
    flattened = rows["train"] + rows["valid"]
    assert stats["ambiguous_nonterminal_skipped"] > 0
    assert any(r["metadata"]["label"] == "P" for r in flattened)
    assert any(r["metadata"]["label"] == "N" for r in flattened)
    assert all(r["metadata"]["negative_means_chemically_impossible"] is False for r in flattened)
    assert all("expected_precursor" not in r["messages"][1]["content"] for r in flattened)
