from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_endpoint_process_rlvr import promotion_decision, rates


def test_monitor_rates_keep_all_product_start_rows_in_denominator():
    value = rates(
        {
            "n": 128,
            "endpoint_exact": 8,
            "formal_terminal": 64,
            "explicit_finish": 80,
            "rejected_proposals": 256,
            "generated_tokens": 12_800,
            "committed_events": 384,
        }
    )
    assert value["n"] == 128
    assert value["endpoint_pass_at_1"] == 0.0625
    assert value["invalid_actions_per_target"] == 2.0
    assert value["mean_generated_tokens"] == 100.0


def test_promotion_requires_above_five_percent_and_parent_gain():
    assert promotion_decision(
        {"endpoint_pass_at_1": 0.0}, {"endpoint_pass_at_1": 0.0625}
    )
    assert not promotion_decision(
        {"endpoint_pass_at_1": 0.0625}, {"endpoint_pass_at_1": 0.0625}
    )
    assert not promotion_decision(
        {"endpoint_pass_at_1": 0.0}, {"endpoint_pass_at_1": 0.05}
    )
