from types import SimpleNamespace

import pytest

from scripts.run_natural_language_value_search import (
    Action, Node, generated_transition_scores, generation_sampling_policy,
    private_product_state, product_only_private_state,
    rollout, select_successful_terminals, visible,
    search_unlabeled,
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
    assert generation_sampling_policy(
        matched_v2=True, vnext_v2_prefix=False, candidates=1,
        planning_sample=True,
    )["do_sample"] is True
    assert generation_sampling_policy(
        matched_v2=True, vnext_v2_prefix=False, candidates=3,
        beam_search=True,
    ) == {"do_sample": False, "temperature": 1.0, "top_p": 1.0,
          "top_k": 50, "num_beams": 3}


def test_nll_ranking_uses_raw_logits_not_temperature_top_p_scores():
    raw = (object(),)
    warped = (object(),)
    output = SimpleNamespace(sequences="tokens", logits=raw, scores=warped)
    received = []
    model = SimpleNamespace(compute_transition_scores=lambda seq, scores, normalize_logits: (
        received.append((seq, scores, normalize_logits)) or scores
    ))
    assert generated_transition_scores(model, output, raw_model_nll=True) is raw
    assert received[-1] == ("tokens", raw, True)
    assert generated_transition_scores(model, output, raw_model_nll=False) is warped
    assert received[-1] == ("tokens", warped, True)
    output.logits = None
    with pytest.raises(ValueError, match="lacks the requested"):
        generated_transition_scores(model, output, raw_model_nll=True)


def test_beam_transition_scores_follow_hf_beam_indices():
    output = SimpleNamespace(
        sequences="tokens", logits="raw", scores="warped",
        beam_indices="ancestry",
    )
    received = []
    model = SimpleNamespace(compute_transition_scores=lambda seq, scores, **kwargs: (
        received.append((seq, scores, kwargs)) or "ranked"
    ))
    assert generated_transition_scores(model, output, raw_model_nll=True) == "ranked"
    assert received == [("tokens", "raw", {
        "normalize_logits": True, "beam_indices": "ancestry",
    })]


def test_state_beam_preserves_pruned_nodes_rejections_and_surviving_path():
    class Runtime:
        pointer_invalid_handles = 0

        def proposals(self, node, **_kwargs):
            state = visible(node.state)
            if not node.actions:
                return [
                    Action("import_fragments", {"fragments": [{"smiles": "O", "count": 1}]}, "O", -0.1, 1),
                    Action("import_fragments", {"fragments": [{"smiles": "N", "count": 1}]}, "N", -0.2, 1),
                    Action("import_fragments", {"fragments": [{"smiles": "F", "count": 1}]}, "F", -0.9, 1),
                ]
            if ".N" in state:
                return [Action("finish_trace", {}, "finish", -0.1, 1)]
            return [Action("invalid_tool", {}, "bad", -0.1, 1)]

        def values(self, _target, states, **_kwargs):
            return [0.0] * len(states)

    args = SimpleNamespace(
        max_decisions=2, branching=3, max_new_tokens=8,
        compact_history=False, max_imports=3,
        reject_target_retained_finish=False,
        early_beam=2, late_beam=2, early_depth=2,
        value_weight=0.0, pointer_weight=0.0,
        product_only_remap=True,
    )
    result = search_unlabeled(Runtime(), "C", args)
    tree = result["search_tree"]
    attempts = result["attempts"]
    assert result["top"].terminal
    assert visible(result["top"].state) == "C.N"
    assert len(tree) == 5  # root, three imports, one terminal
    assert sorted(node["status"] for node in tree).count("pruned") == 1
    failed = [edge for edge in attempts if edge["error"] == "ValueError:UNKNOWN_TOOL"]
    assert len(failed) == 1
    assert failed[0]["child_node_id"] is None
    for node in tree[1:]:
        edge = attempts[node["via_attempt_id"]]
        assert edge["parent_node_id"] == node["parent_node_id"]
        assert edge["child_node_id"] == node["node_id"]


def test_state_beam_does_not_merge_same_visible_state_with_different_aliases():
    class Runtime:
        pointer_invalid_handles = 0

        def proposals(self, node, **_kwargs):
            if not node.actions:
                fragments = ("O", "N")
            elif node.actions[0]["arguments"]["fragments"][0]["smiles"] == "O":
                fragments = ("N",)
            else:
                fragments = ("O",)
            return [Action(
                "import_fragments", {"fragments": [{"smiles": fragment, "count": 1}]},
                fragment, -0.1, 1,
            ) for fragment in fragments]

        def values(self, _target, states, **_kwargs):
            return [0.0] * len(states)

    args = SimpleNamespace(
        max_decisions=2, branching=2, max_new_tokens=8, compact_history=False,
        max_imports=2, reject_target_retained_finish=False,
        early_beam=2, late_beam=2, early_depth=2,
        value_weight=0.0, pointer_weight=0.0, product_only_remap=True,
    )
    searched = search_unlabeled(Runtime(), "C", args)
    frontier = searched["frontier_node_ids"]
    assert len(frontier) == 2
    assert len({searched["search_tree"][node_id]["state"] for node_id in frontier}) == 1
