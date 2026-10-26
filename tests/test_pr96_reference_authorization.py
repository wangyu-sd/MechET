from copy import deepcopy

import pytest

from mechet.inverse_trace_data import build_inverse_tool_sft_row
from scripts.build_main_product_training_data import prepare_reference
from test_mech31k_main_product_root import forward_sn2


def test_authorization_checks_actual_source_actions_and_endpoint_not_just_replay_flags():
    from scripts.finalize_pr96_reference import verify_source_identity
    source = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")
    reference = prepare_reference(source)
    verify_source_identity(source, reference)
    bad = deepcopy(reference)
    bad["metadata"]["trace_plan"]["steps"][0]["moves"][0]["electrons"] = 1
    with pytest.raises(ValueError, match="REFERENCE_STEPS_CHANGED"):
        verify_source_identity(source, bad)
    bad = deepcopy(reference)
    bad["full_precursor_state"] = "[CH3:1][F:3].[OH-:2]"
    with pytest.raises(ValueError, match="ENDPOINT_CHANGED"):
        verify_source_identity(source, bad)


def test_authorization_rejects_hidden_product_counterion_even_if_metadata_claims_main_only():
    from scripts.finalize_pr96_reference import verify_source_identity
    source = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")
    reference = prepare_reference(source)
    reference["target_smiles"] = source["target_smiles"]
    with pytest.raises(ValueError, match="PUBLIC_MAIN_ROOT_CHANGED"):
        verify_source_identity(source, reference)
