import pytest

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
