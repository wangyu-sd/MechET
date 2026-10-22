import json

from test_pr96_history import state_rows
from test_main_product_training_data import legacy_sn2
from scripts.build_main_product_training_data import prepare_reference


class OrderedBackend:
    def __init__(self, functions):
        self.functions = iter(functions)
        self.received = []

    def generate(self, messages):
        self.received.extend(messages)
        return [{"raw": '<tool_call>' + json.dumps(next(self.functions)) + '</tool_call>',
                 "truncated": False, "generated_token_ids": [], "prompt_token_ids": []}
                for _ in messages]


def test_greedy_starts_from_main_product_and_reaches_exact_endpoint_without_gold_imports():
    from mechet.pr96_evaluation import evaluate_reactions
    source = prepare_reference(legacy_sn2())
    rows = state_rows()
    calls = [r["messages"][2]["tool_calls"][0]["function"] for r in rows]
    # TF next-action calls, followed by autonomous closed-loop calls.
    backend = OrderedBackend(calls + calls)
    result = evaluate_reactions([source], {source["source_id"]: rows}, backend,
                                compact_history=False, max_decisions=64)
    episode = result["episodes"][0]
    assert episode["structural_exact"] and episode["full_exact"] and episode["terminal"]
    assert episode["first_error_depth"] is None
    assert [r["name"] for r in episode["actions"]] == ["import_fragments", "apply_electron_flow", "finish_trace"]
    assert "[Br-]" not in backend.received[0][1]["content"]
    assert result["teacher_forced"][1]["score"]["successor_chemical_exact"]
    assert result["metrics"]["closed_loop_completed"] == 1
    assert result["metrics"]["root_import_fragment"] == {"correct": 1, "total": 1}


def test_premature_finish_is_legal_but_not_an_endpoint_hit_and_stays_in_denominator():
    from mechet.pr96_evaluation import evaluate_reactions
    source = prepare_reference(legacy_sn2())
    rows = state_rows()
    calls = [r["messages"][2]["tool_calls"][0]["function"] for r in rows]
    backend = OrderedBackend(calls + [{"name": "finish_trace", "arguments": {}}])
    result = evaluate_reactions([source], {source["source_id"]: rows}, backend,
                                compact_history=True, max_decisions=64)
    episode = result["episodes"][0]
    assert episode["terminal"] and not episode["structural_exact"]
    assert episode["first_error_depth"] == 0
    assert result["metrics"]["structural_hits"] == 0
    assert result["metrics"]["closed_loop_completed"] == 1
    assert all("TRAJECTORY HISTORY" in m[1]["content"] for m in backend.received)


def test_truncated_generation_is_preserved_as_failure_not_silently_dropped():
    from mechet.pr96_evaluation import evaluate_reactions
    source = prepare_reference(legacy_sn2())
    class Truncated:
        def generate(self, messages):
            return [{"raw": "<tool_call>", "truncated": True} for _ in messages]
    result = evaluate_reactions([source], {source["source_id"]: state_rows()}, Truncated(),
                                compact_history=False, max_decisions=64)
    assert result["episodes"][0]["error"] == "OUTPUT_TRUNCATED"
    assert result["metrics"]["closed_loop_completed"] == 1
    assert result["metrics"]["parse"]["correct"] == 0
    assert result["metrics"]["truncated_calls"] == 4
