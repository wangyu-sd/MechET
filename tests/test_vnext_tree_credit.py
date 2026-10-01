import math

import pytest

from mechet.vnext_tree_credit import (
    assign_sibling_advantages,
    chemical_state_key,
    search_teacher_distribution,
)


def branch(index, state, *, positive=False, reference=False, reward=0.0, failure=""):
    return {
        "id": "reaction-1", "prompt_mode": "unified", "anchor": {"state_hash": "same-parent"},
        "candidate_index": index, "action_fingerprint": f"a{index}", "reward": reward,
        "score": {"first_successor_state": state, "first_successor_terminal": False,
                  "correct": positive, "reference_first_successor_exact": reference,
                  "failure": failure},
    }


def test_equivalent_successors_pool_and_all_negative_group_has_zero_credit():
    rows = [branch(0, "CCO"), branch(1, "OCC"), branch(2, "CCN")]
    summary = assign_sibling_advantages(rows, method="tree")
    assert summary["unique_successors"] == 2
    assert summary["equivalent_duplicates"] == 1
    assert summary["all_negative"] and summary["dynamic_resample"]
    assert all(r["advantage"] == 0.0 for r in rows)
    assert search_teacher_distribution(rows) == []
    assert chemical_state_key("[CH3:1][CH2:2][OH:3]") == chemical_state_key("OCC")


@pytest.mark.parametrize("method", ["grpo", "gspo", "tree"])
def test_verified_sibling_positive_and_negative(method):
    rows = [branch(0, "CCO", positive=True, reward=1.0),
            branch(1, "CCN", reward=0.0),
            branch(2, "NCC", reward=0.0)]
    summary = assign_sibling_advantages(rows, method=method)
    assert summary["effective"] and summary["unique_successors"] == 2
    assert rows[0]["advantage"] > 0 and rows[1]["advantage"] < 0
    assert rows[1]["advantage"] == rows[2]["advantage"]
    assert all(math.isfinite(row["advantage"]) for row in rows)
    assert rows[0]["ratio_mode"] == ("sequence" if method == "gspo" else "token")
    teacher = search_teacher_distribution(rows)
    assert len(teacher) == 1 and teacher[0]["candidate_index"] == 0


def test_private_reference_is_ignored_without_opt_in_and_invalid_branch_excluded():
    rows = [branch(0, "CCO", reference=True, reward=0.5),
            branch(1, "CCN", reward=0.0),
            branch(2, "", reward=-0.1, failure="STATE_CYCLE")]
    unprivileged = assign_sibling_advantages(rows)
    assert unprivileged["all_negative"] and all(r["advantage"] == 0 for r in rows)
    summary = assign_sibling_advantages(rows, allow_private_reference=True)
    assert summary["invalid_branches"] == 1
    assert rows[2]["advantage"] == 0.0
    assert search_teacher_distribution(rows)[0]["training_private_reference"]


def test_mixed_parent_state_forbidden():
    rows = [branch(0, "CCO"), branch(1, "CCN")]
    rows[1]["anchor"] = {"state_hash": "different"}
    with pytest.raises(ValueError, match="one reaction"):
        assign_sibling_advantages(rows)
