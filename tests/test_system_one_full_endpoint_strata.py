from __future__ import annotations

import pytest

from scripts.stratify_system_one_full_endpoint import summarize_cases


def _case(reaction_id: str, *, product: str, predicted: str, exact: bool,
          electron_events: int = 1, imports: int = 0) -> dict:
    return {
        "id": reaction_id,
        "completed": True,
        "terminal": "FINISHED",
        "structural_exact": exact,
        "principal_product_input": product,
        "predicted_structural_precursor": predicted,
        "electron_events": electron_events,
        "import_batches": imports,
    }


def test_compiler_coverage_and_no_transform_are_separate() -> None:
    cases = [
        _case("1", product="CO", predicted="C.O", exact=True),
        _case("2", product="CCO", predicted="CCO", exact=False, imports=1),
        _case("3", product="CCC", predicted="CC.C", exact=False),
    ]
    contexts = {str(i): {"top1_exact": i != 3} for i in range(1, 4)}
    groups = summarize_cases(cases, contexts, {"1"}, {"1", "2"}, {"2"})
    assert groups["stitched_strict_trace"]["structural_exact"] == 1
    assert groups["all_steps_executable_but_unstitched"]["finished_wrong_no_transform"] == 1
    assert groups["all_steps_executable_but_unstitched"]["context_exact_but_endpoint_wrong"] == 1
    assert groups["all_steps_executable_but_unstitched"]["min_equ_target_changed"] == 1
    assert groups["stitched_strict_trace"]["min_equ_target_changed"] == 0
    assert groups["incomplete_elementary_steps"]["reactions"] == 1
    assert groups["all"]["reactions"] == 3


def test_stitched_ids_must_be_all_step_executable() -> None:
    with pytest.raises(ValueError, match="subset"):
        summarize_cases([], {}, {"1"}, set())
