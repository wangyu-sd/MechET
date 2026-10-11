"""CPU-only contracts for the opt-in CRN-JEPA/Qwen hybrid and RL gating."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechet.crn_graph_bridge import (
    GRAPH_PRETRAIN_SHA256, TOOLS, _shared_prefix_length, tool_prefix,
)
from mechet.graph_endpoint_rl import VerifiedBranch, success_gated_advantages


class HybridContracts(unittest.TestCase):
    def test_tool_scope(self):
        self.assertEqual(
            TOOLS, ("import_fragments", "apply_electron_flow", "finish_trace")
        )
        self.assertIn('"name": "import_fragments"', tool_prefix("import_fragments"))
        self.assertRaises(ValueError, tool_prefix, "submit_proof")
        self.assertEqual(len(GRAPH_PRETRAIN_SHA256), 64)

    def test_lcp_handles_variable_length_names(self):
        self.assertEqual(
            _shared_prefix_length([[11, 12, 1, 2], [11, 12, 3]]), 2
        )
        self.assertRaises(ValueError, _shared_prefix_length, [[1, 2], [1, 2]])

    def test_no_false_positive_all_negative(self):
        group = [VerifiedBranch(accepted=False), VerifiedBranch(accepted=True)]
        self.assertEqual(success_gated_advantages(group), (0.0, 0.0))

    def test_success_gated_endpoint_priority(self):
        group = [
            VerifiedBranch(accepted=True, exact_terminal=True),
            VerifiedBranch(accepted=True, reference_successor=True),
            VerifiedBranch(accepted=True),
            VerifiedBranch(accepted=False),
        ]
        advantages = success_gated_advantages(group)
        self.assertGreater(advantages[0], advantages[1])
        self.assertGreater(advantages[1], 0)
        self.assertTrue(all(a < 0 for a in advantages[2:]))
        self.assertAlmostEqual(sum(advantages), 0.0, places=7)

    def test_all_positive_has_no_relative_gradient(self):
        positive = [VerifiedBranch(accepted=True, reference_successor=True)] * 3
        self.assertEqual(success_gated_advantages(positive), (0.0,) * 3)

    def test_illegal_reward_claims_rejected(self):
        with self.assertRaises(ValueError):
            _ = VerifiedBranch(accepted=False, exact_terminal=True).reward
        with self.assertRaises(ValueError):
            _ = VerifiedBranch(accepted=False, reference_successor=True).reward


if __name__ == "__main__":
    unittest.main()
