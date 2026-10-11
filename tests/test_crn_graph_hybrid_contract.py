"""CPU-only contracts for the opt-in CRN-JEPA/Qwen hybrid and RL gating."""
from __future__ import annotations

import sys
from pathlib import Path
import json
import subprocess
import unittest

import importlib.util

SRC = Path(__file__).resolve().parents[1] / "src" / "mechet"


def _load_pure_module(name: str, filename: str):
    # Avoid importing mechet.__init__ and its optional RDKit dependencies.
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_pure_module("crn_graph_bridge_contract", "crn_graph_bridge.py")
rl = _load_pure_module("graph_endpoint_rl_contract", "graph_endpoint_rl.py")
GRAPH_PRETRAIN_SHA256 = bridge.GRAPH_PRETRAIN_SHA256
TOOLS = bridge.TOOLS
_shared_prefix_length = bridge._shared_prefix_length
tool_prefix = bridge.tool_prefix
graph_event_coordinates = bridge.graph_event_coordinates
VerifiedBranch = rl.VerifiedBranch
success_gated_advantages = rl.success_gated_advantages


class HybridContracts(unittest.TestCase):
    def test_rl_cli_inherits_single_source_sha_option(self):
        command = [sys.executable,
                   str(Path(__file__).resolve().parents[1] / "scripts" /
                       "train_crn_graph_endpoint_rl.py"), "--help"]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source-sha256 SOURCE_SHA256", completed.stdout)

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

    def test_graph_coordinates_use_only_accepted_action_history(self):
        actions = [
            {"name": "import_fragments"},
            {"name": "apply_electron_flow"},
            {"name": "import_fragments"},
            {"name": "apply_electron_flow"},
        ]
        self.assertEqual(graph_event_coordinates(actions, 4), (2, 2))
        self.assertEqual(graph_event_coordinates(actions, 1), (2, 1))
        with self.assertRaises(ValueError):
            graph_event_coordinates(actions, 0)

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

    def test_distributed_shards_preserve_every_selected_id(self):
        from scripts.run_crn_qwen_hybrid import selected_shard

        rows = [{"id": f"reaction_{number}"} for number in range(29)]
        shards = [selected_shard(rows, rank, 8) for rank in range(8)]
        flattened = [row["id"] for shard in shards for row in shard]
        self.assertEqual(len(flattened), len(rows))
        self.assertEqual(set(flattened), {row["id"] for row in rows})
        with self.assertRaises(ValueError):
            selected_shard(rows, 8, 8)

    def test_resume_rejects_duplicate_or_out_of_shard_records(self):
        from scripts.run_crn_qwen_hybrid import completed_ids
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps({"id": "reaction_1"}) + "\n")
            self.assertEqual(completed_ids(path, {"reaction_1"}), {"reaction_1"})
            with self.assertRaises(ValueError):
                completed_ids(path, {"reaction_2"})
            path.write_text((json.dumps({"id": "reaction_1"}) + "\n") * 2)
            with self.assertRaises(ValueError):
                completed_ids(path, {"reaction_1"})


if __name__ == "__main__":
    unittest.main()
