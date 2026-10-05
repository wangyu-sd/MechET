from mechet.proof_program import sides_equal
from scripts.audit_system_one_output_gap import components_without_stereo


def test_output_gap_normalization_handles_explicit_hydrogen_and_stereo():
    assert sides_equal("[H]OCC.[Na+]", "CCO.[Na+]", ignore_maps=True)
    assert components_without_stereo("[H]OCC.[Na+]") == components_without_stereo("CCO.[Na+]")
    assert not sides_equal("F[C@H](Cl)Br", "FC(Cl)Br", ignore_maps=True)
    assert components_without_stereo("F[C@H](Cl)Br") == components_without_stereo("FC(Cl)Br")
    assert components_without_stereo("CCO") != components_without_stereo("COC")
