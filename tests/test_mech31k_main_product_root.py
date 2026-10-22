"""The public root must not preload missing electron-flow participants."""
import json
from pathlib import Path

import pytest
from mechet.in_place_grounded_flow import mapped_state_signature

from mechet.inverse_trace_data import build_inverse_tool_sft_row, select_mapped_target
from scripts.build_natural_language_event_sft import convert_row
from scripts.run_natural_language_value_search import (
    Action, Node, execute, mapped_atom_numbers, policy_prompt,
    product_only_private_state, visible,
)


def forward_sn2():
    return {
        "id": "sn2_main_only", "source": "mech_uspto_31k", "split": "valid",
        "initial_state": "[CH3:1][Br:3].[OH-:2]",
        "final_state": "[Br-:3].[CH3:1][OH:2]",
        "steps": [{
            "step_index": 0,
            "state_smiles": "[CH3:1][Br:3].[OH-:2]",
            "target_product": "[Br-:3].[CH3:1][OH:2]",
            "moves": [
                {"source": {"kind": "LP", "atoms": [2]},
                 "sink": {"kind": "BOND", "atoms": [1, 2]}},
                {"source": {"kind": "BOND", "atoms": [1, 3]},
                 "sink": {"kind": "ATOM", "atoms": [3]}},
            ],
        }],
    }


def test_mapped_signature_is_stereo_sensitive_but_independent_of_smiles_traversal():
    left = '[F:1][C@H:2]([Cl:3])[Br:4]'
    same = '[Br:4][C@@H:2]([Cl:3])[F:1]'
    opposite = '[Br:4][C@H:2]([Cl:3])[F:1]'
    assert mapped_state_signature(left) == mapped_state_signature(same)
    assert mapped_state_signature(left) != mapped_state_signature(opposite)


def test_main_product_excludes_reference_counterion_and_keeps_it_as_import():
    target, imports, metadata = select_mapped_target(
        "[Br-:3].[CH3:1][OH:2]", "CO.[Br-]", main_product_only=True,
    )
    assert target == "[CH3:1][OH:2]"
    assert imports == ("[Br-:3]",)
    assert metadata["input_condition"] == "main_product_only_v1"


def test_public_and_private_roots_omit_counterion_and_reference_endpoint_is_unchanged():
    row = build_inverse_tool_sft_row(
        forward_sn2(), product_reference="CO.[Br-]", main_product_only=True,
    )
    assert row["target_smiles"] == "[CH3:1][OH:2]"
    assert row["expected_precursor"] == "[CH3:1][Br:3].[OH-:2]"
    assert row["metadata"]["source_product_reference"] == "CO.[Br-]"
    calls = [c["function"] for m in row["messages"] for c in m.get("tool_calls", [])]
    assert calls[0] == {"name": "import_fragment", "arguments": {"fragment_smiles": "[Br-:3]"}}
    assert "[Br-:3]" not in row["messages"][1]["content"]
    assert row["metadata"]["executor_replayed"] is True


def test_first_use_sft_imports_missing_counterion_before_flow_and_replays_online():
    inverse = build_inverse_tool_sft_row(
        forward_sn2(), product_reference="CO.[Br-]", main_product_only=True,
    )
    decisions = convert_row(inverse)
    calls = [r["messages"][2]["tool_calls"][0]["function"] for r in decisions]
    assert [c["name"] for c in calls] == ["import_fragments", "apply_electron_flow", "finish_trace"]
    assert calls[0]["arguments"] == {"fragments": [
        {"smiles": "[Br-]", "count": 1, "purpose": "electron_participant"},
    ]}
    assert all(r["target_smiles"] == "CO" for r in decisions)
    initial = product_only_private_state("CO")
    node = Node(target="CO", state=initial, next_map=max(mapped_atom_numbers(initial)) + 1,
                visited={"CO"})
    for decision, call in zip(decisions, calls, strict=True):
        assert policy_prompt("CO", node.state, include_inventory=True, actions=node.actions,
                             compact_history=False) == decision["messages"][1]["content"]
        child, error = execute(node, Action(call["name"], call["arguments"], "", 0, 0),
                               max_imports=32)
        assert child is not None, error
        node = child
    assert node.terminal and visible(node.state) == "CBr.[OH-]"


@pytest.mark.parametrize("reference", ["[Na+].[Cl-]", "CC.CO"])
def test_main_product_selection_does_not_guess_missing_or_ambiguous_organic_product(reference):
    with pytest.raises(ValueError, match="MAIN_PRODUCT"):
        select_mapped_target("[CH3:1][OH:2]", reference, main_product_only=True)


def test_legacy_product_side_is_explicitly_preserved_without_main_only_option():
    target, imports, _ = select_mapped_target("[Br-:3].[CH3:1][OH:2]", "CO.[Br-]")
    assert target == "[Br-:3].[CH3:1][OH:2]"
    assert imports == ()


def test_full_reference_audit_replays_import_flow_finish_from_public_root():
    from scripts.build_mech31k_main_product_only_sft import process_pair

    forward = forward_sn2()
    legacy = build_inverse_tool_sft_row(forward, product_reference="CO.[Br-]")
    result = process_pair((json.dumps(legacy), json.dumps(forward)))
    assert result["audit"]["reference_endpoint_exact"] is True
    assert result["audit"]["model_prefix_parity"] is True
    assert result["audit"]["flow_successors_match_source"] == 1
    assert result["audit"]["initial_components"] == 1
    assert result["audit"]["first_tool"] == "import_fragments"
    assert result["audit"]["decision_count"] == 3
    assert result["inverse"]["full_precursor_state"] == legacy["full_precursor_state"]


def test_full_reference_audit_does_not_silently_pair_different_reactions():
    from scripts.build_mech31k_main_product_only_sft import process_pair

    forward = forward_sn2()
    legacy = build_inverse_tool_sft_row(forward, product_reference="CO.[Br-]")
    forward["id"] = "wrong-reaction"
    with pytest.raises(ValueError, match="SOURCE_ID"):
        process_pair((json.dumps(legacy), json.dumps(forward)))


@pytest.mark.parametrize('index', [0, 1, 2])
def test_real_stereo_references_replay_from_fresh_unmapped_public_product(index):
    from scripts.build_mech31k_main_product_only_sft import process_pair
    pairs = json.loads((Path(__file__).parent / 'fixtures' /
                       'mech31k_main_only_stereo_pairs.json').read_text())
    legacy, forward = pairs[index]
    if index == 0:
        # Existing executor has no attack-face/stereo action. Do not silently
        # label its traversal-dependent substitution as a correct reference.
        with pytest.raises(ValueError, match='endpoint changed'):
            process_pair((json.dumps(legacy), json.dumps(forward)))
        return
    result = process_pair((json.dumps(legacy), json.dumps(forward)))
    assert result['audit']['reference_endpoint_exact']
    assert result['audit']['model_prefix_parity']


def test_unmapped_comparison_keeps_stereo_charge_hydrogen_and_stoichiometry():
    from scripts.build_mech31k_main_product_only_sft import same_chemical_state
    assert same_chemical_state('[F:1][C@H:2]([Cl:3])[Br:4]',
                               'Br[C@@H](Cl)F')
    assert not same_chemical_state('F[C@H](Cl)Br', 'F[C@@H](Cl)Br')
    assert not same_chemical_state('F[C@H](Cl)Br', 'FC(Cl)Br')
    assert not same_chemical_state('[NH4+]', 'N')
    assert not same_chemical_state('[Cl-].[Cl-]', '[Cl-]')
    assert not same_chemical_state('[H]O[H]', 'O')


def test_endpoint_accepts_identical_unmapped_chemistry_despite_symmetric_map_assignment():
    from scripts.build_mech31k_main_product_only_sft import process_pair
    legacy, forward = json.loads((Path(__file__).parent / 'fixtures' /
                                  'mech31k_main_only_14046_pair.json').read_text())
    result = process_pair((json.dumps(legacy), json.dumps(forward)))
    assert result['audit']['reference_endpoint_exact']
