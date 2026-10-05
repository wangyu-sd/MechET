from scripts.eval_jev_style_successor import select_counts


def test_typed_successor_backoff_does_not_read_gold_flow_count():
    executable_two = {1: {"ok": True}, 2: {"ok": True}}
    invalid_two = {1: {"ok": True}, 2: {"ok": False}}
    for gold_count in (1, 2, 3):
        assert select_counts(gold_count, executable_two)["validity_backoff_2_to_1"] == 2
        assert select_counts(gold_count, invalid_two)["validity_backoff_2_to_1"] == 1
        assert select_counts(gold_count, executable_two)["oracle_count"] == gold_count
