"""CPU-only contracts for the new graph-EARHO scientific algorithm.

Tests refuse a fake on-policy graph frontier when the first executed error
belongs to the frozen Qwen controller. They also verify successor-class credit,
curriculum expansion and all-negative gating, without RDKit or model weights.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

MODULE = Path(__file__).resolve().parents[1] / "src/mechet/graph_earho.py"
spec = importlib.util.spec_from_file_location("graph_earho_contract", MODULE)
earho = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = earho
spec.loader.exec_module(earho)


def event(name, successor, *, accepted=True, terminal=False):
    return earho.Transition(name, accepted, successor, terminal)


class GraphEARHOContracts(unittest.TestCase):
    def test_first_chemically_wrong_flow_is_correctable(self):
        ref = [
            event("import_fragments", "C.O"),
            event(earho.FLOW, "CC.O"),
            event("finish_trace", "CC.O", terminal=True),
        ]
        generated = [
            event("import_fragments", "C.O"),
            event(earho.FLOW, "C.O"),
        ]
        difference = earho.first_executed_divergence(ref, generated)
        self.assertEqual(difference.index, 1)
        self.assertTrue(difference.graph_correctable)
        self.assertEqual(difference.reason, "GRAPH_EXECUTED_SUCCESSOR_DIVERGED")

    def test_qwen_fragment_failure_is_not_graph_earho_gradient(self):
        ref = [
            event("import_fragments", "C.O"),
            event(earho.FLOW, "CC.O"),
        ]
        generated = [event("import_fragments", "C.N")]
        diff = earho.first_executed_divergence(ref, generated)
        self.assertEqual(diff.index, 0)
        self.assertFalse(diff.graph_correctable)
        self.assertIsNone(earho.select_receding_flow_anchor(
            divergence=diff,
            reference_names=[item.name for item in ref],
            horizon=2,
        ))

    def test_wrong_action_type_not_attributed_to_graph(self):
        ref = [event(earho.FLOW, "CC")]
        gen = [event("finish_trace", "C", terminal=True)]
        diff = earho.first_executed_divergence(ref, gen)
        self.assertFalse(diff.graph_correctable)
        self.assertEqual(diff.reason, "QWEN_ROUTING_IMPORT_OR_FINISH")

    def test_reference_aligned_has_no_arbitrary_frontier(self):
        ref = [event(earho.FLOW, "CC"),
               event("finish_trace", "CC", terminal=True)]
        delta = earho.first_executed_divergence(ref, ref)
        self.assertIsNone(delta.index)
        self.assertEqual(delta.reason, "REFERENCE_ALIGNED")

    def test_receding_horizon_goes_back_toward_actual_flow_error(self):
        ref = [earho.FLOW] * 7 + ["finish_trace"]
        diff = earho.Divergence(0, "GRAPH_FLOW_REJECTED", True, 0)
        small = earho.select_receding_flow_anchor(
            divergence=diff, reference_names=ref, horizon=2,
        )
        large = earho.select_receding_flow_anchor(
            divergence=diff, reference_names=ref, horizon=6,
        )
        self.assertEqual(small.decision_index, 6)
        self.assertEqual(large.decision_index, 2)
        self.assertEqual(earho.select_receding_flow_anchor(
            divergence=diff, reference_names=ref, horizon=2,
            full_product=True,
        ).decision_index, 0)

    def test_successor_equivalence_pools_actions_not_invalid_events(self):
        slate = [
            earho.ExecutedAlternative(0, "C.O", True, -1.5),
            earho.ExecutedAlternative(1, "C.O", True, -0.6),
            earho.ExecutedAlternative(2, "CC", True, -2.0),
            earho.ExecutedAlternative(3, None, False, -3.0, "valence"),
            earho.ExecutedAlternative(4, None, False, -3.4, "valence"),
        ]
        groups = earho.pool_executed_successors(slate)
        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0].member_indices, (0, 1))
        self.assertEqual(groups[0].representative_index, 1)
        self.assertNotEqual(groups[2].key, groups[3].key)

    def test_successor_replay_vs_complete_endpoint_has_priority(self):
        groups = [
            earho.VerifiedOutcome(True, endpoint_exact=True),
            earho.VerifiedOutcome(True, reference_successor=True),
            earho.VerifiedOutcome(True),
        ]
        a = earho.successor_group_advantages(groups)
        self.assertGreater(a[0], a[1])
        self.assertGreater(a[1], 0)
        self.assertLess(a[2], 0)
        self.assertAlmostEqual(sum(a), 0.0)

    def test_all_negative_has_zero_gradient(self):
        outcomes = [earho.VerifiedOutcome(False),
                    earho.VerifiedOutcome(True)]
        self.assertEqual(
            earho.successor_group_advantages(outcomes), (0.0, 0.0)
        )

    def test_all_successful_but_endpoint_vs_replay_has_contrast(self):
        outcomes = [earho.VerifiedOutcome(True, endpoint_exact=True),
                    earho.VerifiedOutcome(True, reference_successor=True)]
        a = earho.successor_group_advantages(outcomes)
        self.assertGreater(a[0], 0)
        self.assertLess(a[1], 0)

    def test_curriculum_promotes_only_with_verifiable_support(self):
        c = earho.AdaptiveHorizon(
            current=2, maximum=4, min_effective_groups=2,
            min_productive_fraction=.5,
        )
        same, promoted = c.advance(
            total_groups=4, effective_groups=1, productive_groups=4
        )
        self.assertFalse(promoted)
        self.assertEqual(same.current, 2)
        step, promoted = c.advance(
            total_groups=4, effective_groups=2, productive_groups=2
        )
        self.assertTrue(promoted)
        self.assertEqual(step.current, 3)
        step, promoted = step.advance(
            total_groups=4, effective_groups=2, productive_groups=4
        )
        self.assertEqual(step.current, 4)
        maxed, promoted = step.advance(
            total_groups=4, effective_groups=4, productive_groups=4
        )
        self.assertFalse(promoted)
        self.assertEqual(maxed.current, 4)


if __name__ == "__main__":
    unittest.main()
