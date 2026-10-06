from types import SimpleNamespace

from scripts.run_natural_language_value_search import (
    Action, Node, generation_sampling_policy, private_product_state,
    product_only_private_state,
    rollout, select_successful_terminals, visible,
)


def test_product_only_private_mapping_and_unmapped_endpoint():
    product = "CO.[Cl-]"
    mapped = private_product_state(product)
    assert ":1" in mapped and visible(mapped) == visible(product)
    assert private_product_state(mapped) == mapped


def test_reliable_product_start_discards_source_atom_maps():
    left = product_only_private_state("[CH3:77][OH:42]")
    right = product_only_private_state("[CH3:9][OH:501]")
    assert left == right
    assert ":77" not in left and ":501" not in right


def test_reliable_rollout_observes_only_fresh_product_mapping():
    class Runtime:
        pointer_invalid_handles = 0

        def proposals(self, node, **kwargs):
            assert node.state == product_only_private_state("[CH3:77][OH:42]")
            return [Action("finish_trace", {}, "", -0.1, 1)]

        def values(self, *args, **kwargs):
            return [0.0]

    args = SimpleNamespace(max_decisions=1, branching=1, max_new_tokens=1,
                           compact_history=False, max_imports=2,
                           reject_target_retained_finish=False, early_beam=1,
                           late_beam=1, early_depth=1, value_weight=0.0,
                           pointer_weight=0.0, product_only_remap=True)
    row = {"id": "test", "source_id": "test", "target_smiles": "[CH3:77][OH:42]",
           "expected_precursor": "[CH3:77][OH:42]"}
    result = rollout(Runtime(), row, args)
    assert result["top1_full_exact"]
    assert result["attempts"][0]["accepted"]
    assert result["attempts"][0]["state_after"] == result["target"]


def test_reliable_rollout_keeps_no_call_failure_in_denominator():
    class Runtime:
        pointer_invalid_handles = 0
        last_proposal_error = "CONTEXT_BUDGET_EXCEEDED"

        def proposals(self, node, **kwargs):
            return []

        def values(self, *args, **kwargs):
            return []

    args = SimpleNamespace(max_decisions=1, branching=1, max_new_tokens=1,
                           compact_history=False, max_imports=2,
                           reject_target_retained_finish=False, early_beam=1,
                           late_beam=1, early_depth=1, value_weight=0.0,
                           pointer_weight=0.0, product_only_remap=True)
    row = {"id": "test", "source_id": "test", "target_smiles": "[CH3:77][OH:42]",
           "expected_precursor": "[CH3:77][OH:42]"}
    result = rollout(Runtime(), row, args)
    assert not result["top1_exact"]
    assert result["rejected"] == {"CONTEXT_BUDGET_EXCEEDED": 1}
    assert result["attempts"][0]["error"] == "CONTEXT_BUDGET_EXCEEDED"


def test_pointer_bonus_is_independent_of_executor_value():
    node = Node(target="CO", state="[CH3:1][OH:2]", next_map=3,
                actions=[{"name": "apply_electron_flow"}], logprob=-4.0,
                tokens=4, value=0.5, pointer_score=2.0)
    assert node.score(0.0, 0.0) == -1.0
    assert node.score(0.2, 0.0) == -0.9
    assert node.score(0.0, 0.5) == 0.0


def test_unmapped_reference_has_full_endpoint_metric():
    class Runtime:
        pointer_invalid_handles = 0

        def proposals(self, node, **kwargs):
            return [Action("finish_trace", {}, "", -0.1, 1)]

        def values(self, *args, **kwargs):
            return [0.0]

    args = SimpleNamespace(max_decisions=1, branching=1, max_new_tokens=1,
                           compact_history=True, max_imports=2,
                           reject_target_retained_finish=False, early_beam=1,
                           late_beam=1, early_depth=1, value_weight=0.0,
                           pointer_weight=0.0)
    row = {"id": "test", "source_id": "test", "target_smiles": "CO",
           "expected_precursor": "CO"}
    result = rollout(Runtime(), row, args)
    assert result["endpoint_metric"] == "full_unmapped"
    assert result["top1_full_exact"]


def test_full_endpoint_teacher_is_not_lost_to_earlier_structural_hit():
    structural_only = Node(target="CO", state="C.[Na+]", next_map=3, terminal=True)
    full_exact = Node(target="CO", state="C.O", next_map=3, terminal=True)
    structural, full = select_successful_terminals(
        [structural_only, full_exact], lambda node: "C" in visible(node.state), "C.O"
    )
    assert structural is structural_only
    assert full is full_exact


def test_vnext_k1_is_greedy_but_k4_is_stochastic_expansion():
    assert generation_sampling_policy(
        matched_v2=False, vnext_v2_prefix=True, candidates=1
    ) == {"do_sample": False}
    assert generation_sampling_policy(
        matched_v2=False, vnext_v2_prefix=True, candidates=4
    )["do_sample"] is True
