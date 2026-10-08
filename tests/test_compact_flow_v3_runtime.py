"""V3 rollout uses distinct schema, frozen v2 remains the default."""
from argparse import Namespace
import json

import pytest

from scripts.build_compact_electron_flow_sft import SYSTEM, TOOLS
from scripts import eval_reliable_independent_episodes as evaluator
from scripts.run_natural_language_value_search import (
    Action, Node, execute, validate_compact_flow_v3_adapter_manifest,
)
from mechet.compact_electron_flow import render_compact_event_arguments
from mechet.natural_language_electron_flow import render_event_arguments


def test_v3_schema_and_policy_args_do_not_change_v2_defaults():
    assert "DELTA|" in SYSTEM
    assert set(TOOLS[1]["function"]["parameters"]["properties"]) == {"flow"}
    args = Namespace(stage="state", adapter="adapter", dtype="bfloat16",
                     no_4bit=True, max_new_tokens=128, max_context=4096,
                     record_attempts=False)
    assert evaluator.policy_args(args).compact_flow_v3 is False
    args.compact_flow_v3 = True
    actor = evaluator.policy_args(args)
    assert actor.compact_flow_v3 is True
    assert actor.matched_v2 is True
    assert actor.compact_history is False


def test_separate_adapter_lineage_guard(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    good = {
        "environment_revision": "compact_electron_flow_v3",
        "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
        "base_model": "Qwen/Qwen3-0.6B",
        "base_model_revision": "c" * 40,
    }
    (adapter / "adapter_manifest.json").write_text(json.dumps(good))
    assert validate_compact_flow_v3_adapter_manifest(
        adapter, expected_model="Qwen/Qwen3-0.6B", expected_revision="c" * 40,
    ) == good
    good["environment_revision"] = "natural_language_electron_event_v2"
    (adapter / "adapter_manifest.json").write_text(json.dumps(good))
    with pytest.raises(ValueError, match="compact v3"):
        validate_compact_flow_v3_adapter_manifest(adapter)


def test_real_executor_accepts_compact_and_verbose_same_successor():
    state = "[CH3:1][OH:2].[Br-:3]"
    moves = [
        {"source": {"kind": "BOND", "atoms": [1, 2]},
         "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [3]},
         "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
    ]
    root = Node(target="CO", state=state, next_map=4, visited={"CO"})
    verbose, error_a = execute(
        root, Action("apply_electron_flow", render_event_arguments(state, moves),
                     "", -1.0, 1), max_imports=32,
    )
    compact, error_b = execute(
        root, Action("apply_electron_flow", render_compact_event_arguments(state, moves),
                     "", -1.0, 1), max_imports=32,
    )
    assert error_a == error_b == ""
    assert verbose is not None and compact is not None
    assert verbose.state == compact.state
    assert verbose.actions[-1]["result"] == compact.actions[-1]["result"]
