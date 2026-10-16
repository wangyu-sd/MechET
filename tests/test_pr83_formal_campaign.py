"""Unit checks for formal-job sequencing, without model sampling."""
import importlib.util
import json
from pathlib import Path
import unittest

FILE = Path(__file__).resolve().parents[1] / "scripts/run_pr83_formal_campaign.py"
spec = importlib.util.spec_from_file_location("pr83_campaign", FILE)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class FormalCampaignTests(unittest.TestCase):
    def test_waiting_and_running_cannot_advance_to_training(self):
        for state in ("READY", "TRAINING_RESOURCE_WAITING", "PENDING", "TRAINING_RUNNING"):
            self.assertEqual(campaign.next_stage(0, {"state": state}), 0)

    def test_failed_or_killed_job_cannot_advance(self):
        for success in (False, None):
            with self.assertRaises(RuntimeError):
                campaign.next_stage(0, {"state": "END", "is_succeed": success})

    def test_success_advances_exactly_one_stage(self):
        self.assertEqual(campaign.next_stage(1, {"state": "END", "is_succeed": True}), 2)

    def test_only_common_status_is_returned_from_private_detail(self):
        raw = 'task instance detail:\n' + json.dumps({"common": {"state": "END"}})
        raw += '\n "instance_config": {"init_cmd": "private"}'
        self.assertEqual(campaign.read_common(raw), {"state": "END"})


if __name__ == "__main__":
    unittest.main()
