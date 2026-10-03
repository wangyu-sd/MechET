import random

from rdkit import DataStructs

from scripts.audit_nmi_structural_overlap import _candidate_bitcounts, is_near_duplicate
from mechet.structural_overlap import reaction_center_context_features, reaction_center_context_signature


def test_bitcount_pruning_has_no_threshold_false_negatives():
    rng = random.Random(17)
    vectors = []
    for _ in range(80):
        vector = DataStructs.ExplicitBitVect(128)
        for bit in rng.sample(range(128), rng.randint(4, 90)):
            vector.SetBit(bit)
        vectors.append(vector)
    for probe in vectors:
        buckets = {}
        for candidate in vectors:
            buckets.setdefault(candidate.GetNumOnBits(), []).append(candidate)
        for threshold in (0.5, 0.8, 0.9, 1.0):
            expected = any(
                DataStructs.TanimotoSimilarity(probe, candidate) >= threshold
                for candidate in vectors
            )
            assert is_near_duplicate(probe, buckets, threshold) == expected
    assert list(_candidate_bitcounts(10, 0.9)) == [9, 10, 11]


def test_explicit_hydrogen_center_uses_mapped_h_instead_of_dropping_it():
    row = {"metadata": {"trace_plan": {"target_smiles": "[O:1][H:2]", "expected_precursor": "[O-:1].[H+:2]", "initial_imports": [], "steps": [{"step_index": 0, "state_before": "[O:1][H:2]", "state_after": "[O-:1].[H+:2]", "imports": [], "moves": [{"source": {"kind": "BOND", "atoms": [1, 2]}, "sink": {"kind": "ATOM", "atoms": [1]}, "electrons": 2}]}]}}}
    assert len(reaction_center_context_signature(row, preserve_mapped_h=True)) == 64


def test_be_delta_center_uses_bond_and_charge_atoms():
    row = {"metadata": {"trace_plan": {"target_smiles": "[C:1][O:2]", "expected_precursor": "[C:1].[O:2]", "initial_imports": [], "steps": [{"step_index": 0, "state_before": "[C:1][O:2]", "state_after": "[C:1].[O:2]", "imports": [], "moves": [{"mode": "BE_DELTA", "bond_deltas": [{"atoms": [1, 2], "delta": -1}], "charge_actions": [{"atom_map": 2, "q0": 0, "q1": -1}]}]}]}}}
    assert len(reaction_center_context_signature(row, preserve_mapped_h=True)) == 64


def test_local_center_context_does_not_encode_move_direction_or_program_label():
    base = {"metadata": {"trace_plan": {"target_smiles": "[CH3:1][OH:2]", "expected_precursor": "[CH3:1].[OH:2]", "initial_imports": [], "steps": [{"step_index": 0, "state_before": "[CH3:1][OH:2]", "state_after": "[CH3:1].[OH:2]", "imports": [], "moves": [{"source": {"kind": "BOND", "atoms": [1, 2]}, "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2}]}]}}}
    alternate = {"metadata": {"trace_plan": {**base["metadata"]["trace_plan"], "steps": [{**base["metadata"]["trace_plan"]["steps"][0], "moves": [{"source": {"kind": "BOND", "atoms": [1, 2]}, "sink": {"kind": "ATOM", "atoms": [1]}, "electrons": 2}]}]}}}
    left = reaction_center_context_features(base, preserve_mapped_h=True)
    right = reaction_center_context_features(alternate, preserve_mapped_h=True)
    assert left["ordered_context_key"] != right["ordered_context_key"]
    assert left["local_structural_centers"] == right["local_structural_centers"]
