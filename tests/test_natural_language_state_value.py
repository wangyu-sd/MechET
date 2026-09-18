import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_natural_language_state_value import (
    remove_model_visible_label_conflicts,
    value_row,
)


def test_model_visible_conflict_filter_drops_only_counterfactual(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    common = dict(
        reaction_id="rxn",
        target="[CH3:1][CH3:2]",
        state="[CH3:1][CH3:2]",
    )
    rows = [
        value_row(key="positive", label="A", provenance="reference_prefix", **common),
        value_row(key="negative", label="C", provenance="counterfactual", **common),
        value_row(
            reaction_id="rxn",
            key="other_negative",
            target="[CH3:1][CH3:2]",
            state="[CH3:1].[CH3:2]",
            label="C",
            provenance="counterfactual",
        ),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    audit = remove_model_visible_label_conflicts(path)
    cleaned = [json.loads(line) for line in path.read_text().splitlines()]

    assert audit == {
        "dropped_conflicting_counterfactuals": 1,
        "label_A": 1,
        "rows": 2,
        "label_C": 1,
        "model_visible_conflicts_after": 0,
    }
    assert [row["metadata"]["label"] for row in cleaned] == ["A", "C"]
