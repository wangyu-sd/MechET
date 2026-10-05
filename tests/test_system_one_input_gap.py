from scripts.audit_system_one_full_endpoint_input_gap import audit, component_counter


def test_input_gap_distinguishes_final_mixture_from_principal_product():
    strict = {"1": {"target_smiles": "CCO.[Br-]", "expected_precursor": "CCBr.O"}}
    full = {"1": {"product_unmapped": "CCO", "reactants_unmapped": "CCBr.O",
                  "precursor_unmapped": "CCBr"}}
    counts = audit(strict, full)["counts"]
    assert counts["strict_reactions"] == 1
    assert counts["target_exact"] == 0
    assert counts["strict_target_more_components"] == 1
    assert counts["principal_product_components_contained_in_strict_target"] == 1
    assert counts["strict_precursor_equals_full_reactants"] == 1
    assert counts["strict_precursor_equals_full_structural"] == 0


def test_product_component_containment_uses_multisets():
    assert component_counter("C.C") - component_counter("C")


def test_stereo_only_gap_and_residual_context_are_separate():
    strict = {"1": {"target_smiles": "F[C@H](Cl)Br.[I-]",
                    "expected_precursor": "F[C@H](Cl)Br.[I-]"}}
    full = {"1": {"product_unmapped": "FC(Cl)Br",
                  "reactants_unmapped": "FC(Cl)Br.[I-]",
                  "precursor_unmapped": "FC(Cl)Br"}}
    result = audit(strict, full)
    assert result["counts"]["principal_product_components_contained_in_strict_target"] == 0
    assert result["counts"]["principal_product_components_contained_without_stereo"] == 1
    assert result["residual_context_batches_without_stereo"] == [
        {"components": ["[I-]"], "reactions": 1}
    ]
