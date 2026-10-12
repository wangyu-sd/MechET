"""No-GPU causal rollout tests exercising the ACTUAL Graph-EARHO functions.

Load the function definitions through Python AST rather than importing the
heavy MechET/RDKit/transformers trainer during the lightweight CI workflow.
This detects future regressions in model-owned anchors and cross-model
mixed-NLL search selection, not just in a separate mock algorithm.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pure_graph_earho_runtime", ROOT / "src/mechet/graph_earho.py"
)
pure = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pure
spec.loader.exec_module(pure)


@dataclass
class FakeAction:
    name: str
    target: str
    logprob: float = 0.0


@dataclass
class FakeNode:
    target: str
    state: str
    next_map: int
    visited: set
    actions: list = field(default_factory=list)
    terminal: bool = False
    policy_score: float = 0.0


def execute(node, action, *, max_imports, reject_target_retained_finish=True):
    if action.target == "INVALID":
        return None, "REJECTED"
    child = FakeNode(
        target=node.target, state=action.target,
        next_map=node.next_map, visited=set(node.visited) | {action.target},
        actions=node.actions + [{"name": action.name}],
        terminal=action.name == "finish_trace",
        policy_score=action.logprob,
    )
    return child, ""


class FakeRuntime:
    def __init__(self, state_to_actions):
        self.state_to_actions = state_to_actions
        self.last_proposal_error = ""

    def proposals(self, node, *, candidates, max_new_tokens, compact_history):
        return self.state_to_actions.get(node.state, [])


ARGS = SimpleNamespace(
    max_import_tokens=256, compact_history=False,
    max_imports=32, max_decisions=40, rollout_beam_width=2,
)


def source_functions():
    script = ROOT / "scripts/train_crn_graph_earho.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    needed = {"observe_policy_divergence", "bounded_continuation"}
    selected = [
        item for item in tree.body if isinstance(item, ast.FunctionDef)
        and item.name in needed
    ]
    selected += [
        item for item in tree.body if isinstance(item, ast.ClassDef)
        and item.name == "ContinuationResult"
    ]
    if len(selected) != 3:
        raise AssertionError("Graph-EARHO runtime functions are missing")
    program = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    scope = {
        "Node": FakeNode,
        "dataclass": dataclass,
        "visible": lambda text: text,
        "mapped_atom_numbers": lambda _: (1,),
        "product_only_private_state": lambda value: value,
        "Transition": pure.Transition,
        "first_executed_divergence": pure.first_executed_divergence,
        "execute": execute,
    }
    exec(compile(ast.fix_missing_locations(program), str(script), "exec"), scope)
    return scope


class ActualGraphEARHORuntimeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fns = source_functions()

    def test_first_graph_error_returns_real_model_state(self):
        ref = [
            pure.Transition("import_fragments", True, "IMPORTED"),
            pure.Transition(pure.FLOW, True, "GOLD"),
        ]
        runtime = FakeRuntime({
            "PRODUCT": [FakeAction("import_fragments", "IMPORTED")],
            "IMPORTED": [FakeAction(pure.FLOW, "WRONG")],
        })
        diff, node = self.fns["observe_policy_divergence"](
            runtime, {"original_target": "PRODUCT", "reference": ref}, ARGS,
        )
        self.assertEqual(diff.index, 1)
        self.assertTrue(diff.graph_correctable)
        self.assertEqual(node.state, "IMPORTED")
        self.assertEqual(len(node.actions), 1)
        self.assertEqual(node.actions[0]["name"], "import_fragments")

    def test_first_wrong_qwen_import_is_not_graph_trainable(self):
        reference = [
            pure.Transition("import_fragments", True, "IMPORTED"),
            pure.Transition(pure.FLOW, True, "GOLD"),
        ]
        runtime = FakeRuntime({
            "PRODUCT": [FakeAction("import_fragments", "WRONG_IMPORT")]
        })
        diff, node = self.fns["observe_policy_divergence"](
            runtime, {"original_target": "PRODUCT", "reference": reference}, ARGS,
        )
        self.assertEqual(diff.index, 0)
        self.assertFalse(diff.graph_correctable)
        self.assertEqual(node.state, "PRODUCT")

    def test_graph_empty_slate_counts_as_graph_correctable(self):
        ref = [pure.Transition(pure.FLOW, True, "GOLD")]
        runtime = FakeRuntime({})
        runtime.last_proposal_error = "GRAPH_NO_FLOW_CANDIDATE"
        diff, node = self.fns["observe_policy_divergence"](
            runtime, {"original_target": "PRODUCT", "reference": ref}, ARGS,
        )
        self.assertTrue(diff.graph_correctable)
        self.assertEqual(diff.reason, "GRAPH_FLOW_REJECTED")
        self.assertEqual(node.state, "PRODUCT")

    def test_mixed_qwen_graph_logprob_does_not_rank_cross_parent_beam(self):
        # Model-score sorting would favor the high-scoring later parent B,
        # but balanced bounded breadth-first selection keeps the earlier A
        # parent's top two chemically distinct next states.
        actions = {
            "P": [FakeAction(pure.FLOW, "A", 0.1),
                  FakeAction(pure.FLOW, "B", -0.1)],
            "A": [FakeAction("import_fragments", "C", -1000),
                  FakeAction("import_fragments", "E", -900)],
            "B": [FakeAction(pure.FLOW, "D", 1000)],
            "C": [FakeAction("finish_trace", "C_final", -1000)],
            "E": [FakeAction("finish_trace", "E_final", -900)],
            "D": [FakeAction("finish_trace", "D_final", 1000)],
        }
        output = self.fns["bounded_continuation"](
            FakeRuntime(actions),
            FakeNode("P", "P", 1, {"P"}),
            ARGS, decisions=3,
        )
        self.assertEqual(
            {x.state for x in output.terminals},
            {"C_final", "E_final"},
        )
        self.assertTrue(output.censored)  # D branch was pruned by budget


if __name__ == "__main__":
    unittest.main()
