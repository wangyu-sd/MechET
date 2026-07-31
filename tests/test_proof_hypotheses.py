from mechet.proof_curriculum import corrupt_proof
from mechet.proof_hypotheses import (
    deduplicate_hypotheses,
    score_hypothesis,
    summarize_hypotheses,
    survival_curve,
)
from tests.test_proof_curriculum import substitution_proof


def test_hypothesis_scoring_uses_structural_endpoint_and_deduplicates():
    proof = substitution_proof()
    first = score_hypothesis(
        proof,
        source_index=0,
        expected_precursor="[CH3:1][Br:3].[OH-:2]",
        model_logprob=-0.2,
    )
    second = score_hypothesis(
        proof,
        source_index=1,
        expected_precursor="[CH3:1][Br:3].[OH-:2]",
        model_logprob=-0.4,
    )
    invalid_text = corrupt_proof(proof, corruption_type="LP_WRONG_DELTA").corrupted_proof
    invalid = score_hypothesis(invalid_text, source_index=2)
    assert first.execute_ok and first.endpoint_exact
    assert "Na" not in first.derived_core_precursor
    unique = deduplicate_hypotheses([first, second, invalid])
    assert sum(item.execute_ok for item in unique) == 1
    summary = summarize_hypotheses([first, second, invalid])
    assert summary.n_generated == 3
    assert summary.n_equivalence_classes == 1
    survival = survival_curve([first, second, invalid])
    assert survival["executable"] == 2
    assert survival["endpoint_compatible"] == 2
