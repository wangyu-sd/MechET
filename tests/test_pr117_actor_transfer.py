import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from scripts import pr117_actor_transfer_score as scorer
from scripts.pr117_actor_transfer import matched_actor_panel, normalized_adapter_config
from scripts.pr117_actor_transfer_score import native_endpoints
from scripts.pr117_controllers import C0, C2, C4


def test_trained_actor_preserves_every_target_stock_seed_and_request_ceiling():
    baseline = dict(actor={"adapter_sha256": "stage2"}, arms=[C0, C2, C4],
                    targets=[{"target_id": "test", "target": "CC", "seed": 10}],
                    seed=17, stocks={"official": {"size": 100}}, request_ceiling_per_target_arm=64)
    trained = matched_actor_panel(baseline, C0, "learned")
    assert {k: v for k, v in trained.items() if k not in {"arms", "actor"}} == {
        k: v for k, v in baseline.items() if k not in {"arms", "actor"}}
    assert trained["arms"] == [C0] and trained["actor"] == {"adapter_sha256": "learned"}
    assert baseline["arms"] == [C0, C2, C4] and baseline["actor"]["adapter_sha256"] == "stage2"
    with pytest.raises(ValueError, match="distinct"):
        matched_actor_panel(baseline, C0, "stage2")
    with pytest.raises(ValueError, match="same controller"):
        matched_actor_panel(baseline, "different", "learned")


def test_adapter_module_order_does_not_change_execution_config():
    left = {"target_modules": ["q_proj", "v_proj"], "r": 16}
    right = {"target_modules": ["v_proj", "q_proj"], "r": 16}
    assert normalized_adapter_config(left) == normalized_adapter_config(right)
    assert normalized_adapter_config(left) != normalized_adapter_config(dict(right, r=8))


def test_actor_endpoints_read_first_native_action_and_charge_only_physical_root_calls():
    receipts = [dict(request_id="preflight", generated_token_ids=[], resource_error="PHYSICAL_CONTEXT_LIMIT"),
                dict(request_id="first-cuda", generated_token_ids=[1, 2], prompt_token_ids=[17]),
                dict(request_id="later-cuda", generated_token_ids=[3, 4, 5])]
    raw = dict(cohort="long", seconds=5.2, result=dict(decision_receipts=receipts, independent_attempts=[
        {"records": [{"request_id": "first-cuda", "accepted": True}]}]))
    audit = dict(solved=False, status="resource_censored", gpu_requests=2, requests=3, generated_tokens=5,
                 first_solution_requests=None, first_solution_tokens=None,
                 distinct_flow_transitions=1,
                 commits=[dict(receipt="later-cuda", occurrence="0")])
    outcome = native_endpoints(raw, audit)
    assert outcome["first_action_accepted"]
    assert outcome["first_root_reaction_call"] == 2
    assert outcome["first_solution"] is None
    assert outcome["native_calls"] == 2


def test_actor_comparison_counts_complete_proofs_at_common_native_call_limits(monkeypatch):
    def outcome(first_solution_calls):
        first_solution = None
        if first_solution_calls is not None:
            first_solution = dict(native_calls=first_solution_calls, generated_tokens=1000, terminal_seconds=12.5)
        return dict(cohort="long", stock_closed=first_solution is not None, first_solution=first_solution,
                    first_action_accepted=True, first_root_reaction_call=4,
                    first_request_id="first", first_prompt_sha256="same-prompt")

    raw_outcomes = {
        "parent-results": {("long", "earlier"): outcome(11), ("long", "boundary"): outcome(20),
                           ("long", "last"): outcome(64)},
        "trained-results": {("long", "earlier"): outcome(10), ("long", "boundary"): outcome(31),
                            ("long", "last"): outcome(None)},
    }
    proof = dict(primary_arm=C0, targets=3, per_target_arm_request_cap=64)
    version = (None, None, None, proof)
    monkeypatch.setattr(scorer, "bind_runtime", lambda args: version)
    monkeypatch.setattr(scorer, "bind_trained_comparison", lambda args: version)
    monkeypatch.setattr(scorer, "replay_arm",
                        lambda results, *args: (raw_outcomes[results], {"final_summary_verified": True}))

    def measure_transitions(pairs, trained, parent, success):
        return dict(Counter(f"{int(success(row[parent]))}->{int(success(row[trained]))}" for row in pairs))

    # This fixture isolates the archive-replay and statistics boundaries. The scorer
    # still builds its paired population and chooses the success criterion for every cap.
    monkeypatch.setitem(sys.modules, "analyze_paired_reference", SimpleNamespace(estimate=measure_transitions))
    results = scorer.evaluate(SimpleNamespace(parent_results="parent-results", trained_results="trained-results"))
    effects = results["effects"]
    assert effects["inventory_closed_by_10_calls"] == {"0->1": 1, "0->0": 2}
    assert effects["inventory_closed_by_20_calls"] == {"1->1": 1, "1->0": 1, "0->0": 1}
    assert effects["inventory_closed_by_30_calls"] == {"1->1": 1, "1->0": 1, "0->0": 1}
    assert effects["inventory_closed_by_64_calls"] == effects["full_inventory_closed"] == {"1->1": 2, "1->0": 1}
    assert effects["productive_root_by_10_calls"] == {"1->1": 3}
