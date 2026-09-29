"""First executor rejection is not the same as first reference divergence."""

from scripts.autoresearch.audit_r3_executor_rejection_baseline import classify


def test_r3_first_rejection_position_classes() -> None:
    assert classify(2, 2) == "first_rejection_at_mutation"
    assert classify(4, 2) == "first_rejection_after_mutation"
    assert classify(1, 2) == "first_rejection_before_mutation"
    assert classify(None, 2) == "no_rejection"
