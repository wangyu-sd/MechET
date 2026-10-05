from mechet.electron_pointer import parse_pointer_example, parse_pointer_observation
from mechet.natural_language_electron_flow import execute_event_arguments
from mechet.system_one_replay import (
    execute_pair_indices,
    pair_indices_to_arguments,
    reconstruct_mapped_state,
)
from scripts.eval_system_one_successor import select_flow_counts
from scripts.eval_pr71_pointer_successor import coupled_target_indices
from scripts.compare_system_one_pr71_successor import backoff, cluster_bootstrap


def test_visible_state_reconstruction_and_pair_execution():
    row = {
        "id": "example::event", "tools": [],
        "messages": [
            {"role": "user", "content": (
                "ANNOTATED CURRENT STATE: <A01>C<A02>O.<A03>[Cl-]"
            )},
            {"role": "assistant", "tool_calls": [{"function": {
                "name": "apply_electron_flow", "arguments": {
                    "direction": "retrosynthetic", "electron_flow": [
                        {"source": "a lone pair on atom A03",
                         "destination": "the bond to form between atoms A01 and A03"},
                        {"source": "the bond between atoms A01 and A02",
                         "destination": "atom A02"},
                    ], "bond_order_changes": [], "charge_changes": [],
                },
            }}]},
        ],
    }
    observation = parse_pointer_example(row)
    assert observation is not None
    mapped = reconstruct_mapped_state(observation)
    assert ":1]" in mapped
    # Source A03 has index 2; sink A01-A03 has index 4 among three atoms
    # followed by three unordered atom-pair sinks.
    pair_indices = [2 * 6 + 4, 3 * 6 + 1]
    assert coupled_target_indices(observation) == sorted(pair_indices)
    generated = pair_indices_to_arguments(observation, pair_indices)
    assert generated["electron_flow"][0]["source"] == "a lone pair on atom A03"
    assert generated["electron_flow"][0]["destination"] == (
        "the bond to form between atoms A01 and A03"
    )
    expected = execute_event_arguments(
        mapped, observation.assistant_message["tool_calls"][0]["function"]["arguments"]
    )
    actual = execute_pair_indices(mapped, observation, pair_indices)
    assert actual["ok"]
    assert actual == expected


def test_pair_replay_rejects_duplicates_and_out_of_range():
    observation = parse_pointer_observation(
        "ANNOTATED CURRENT STATE: <A01>C<A02>O.<A03>[Cl-]"
    )
    import pytest

    with pytest.raises(ValueError, match="unique"):
        pair_indices_to_arguments(observation, [1, 1])
    with pytest.raises(ValueError, match="outside"):
        pair_indices_to_arguments(observation, [1000])


def test_reconstruction_accepts_rdkit_equivalent_redundant_stereo_only():
    # This real train-state serialization flips one @/@@ pair during RDKit's
    # map-free canonicalization while retaining the same isomer and Axx graph.
    annotated = (
        "<A01>C<A02>N(<A03>C<A04>[C@H]1<A05>C<A06>[C@@H](<A07>O<A08>S("
        "<A09>C)(=<A10>O)(<A11>[O-])<A12>Cl)<A13>C1)<A14>C(=<A15>O)"
        "<A16>O<A17>C(<A18>C)(<A19>C)<A20>C"
    )
    observation = parse_pointer_observation("ANNOTATED CURRENT STATE: " + annotated)
    assert reconstruct_mapped_state(observation)


def test_executor_validity_backoff_does_not_use_reference_count():
    failed_two = {1: {"ok": True}, 2: {"ok": False}}
    valid_two = {1: {"ok": True}, 2: {"ok": True}}
    for reference_count in (1, 2):
        assert select_flow_counts(reference_count, failed_two)[
            "validity_backoff_2_to_1"
        ] == 1
        assert select_flow_counts(reference_count, valid_two)[
            "validity_backoff_2_to_1"
        ] == 2
        assert select_flow_counts(reference_count, valid_two)["oracle_count"] == reference_count


def test_paired_comparison_uses_executor_validity_only():
    one = {"execute_ok": True, "successor_exact": True, "successor": "CO"}
    two = {"execute_ok": False, "successor_exact": False, "successor": None}
    row = {"id": "example", "policies": {"fixed1": one, "fixed2": two}}
    assert backoff(row) is one
    two["execute_ok"] = True
    assert backoff(row) is two
    assert cluster_bootstrap([(1, 1, 0)] * 3, repetitions=100) == (1.0, 1.0)
