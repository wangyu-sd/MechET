"""Bounded controller checks with scripted actor responses and a route-state fixture."""
from copy import deepcopy
from types import SimpleNamespace

from scripts import pr117_controllers as controllers


class Episode:
    def __init__(self):
        self.uid = "route"
        self.public_root = "A"
        self.current = dict(product="A", occurrence="0", ancestors=[])
        self.node = SimpleNamespace(state="a0", actions=[])
        self.pending = []
        self.stock_leaves = []
        self.reactions = []
        self.records = []
        self.cfg = {}
        self.status = self.error = None

    def next_decision(self):
        return None if self.status else (self.current["occurrence"], self.node.state)

    def fork(self, namespace):
        copy = deepcopy(self)
        copy.uid = namespace
        return copy

    def accept(self, receipt):
        name = receipt["raw"]
        self.records.append(receipt)
        if name == "finish_cycle":
            self.status = "rejected_action"
            self.error = "ROUTE_CYCLE:A"
        elif name.startswith("flow"):
            self.node.state = name
            self.node.actions.append(dict(name="apply_electron_flow"))
        elif name == "finish_b":
            self.reactions.append(dict(occurrence="0", product="A", precursors=["B", "S"]))
            self.stock_leaves.append("S")
            self.current = dict(product="B", occurrence="0.0", ancestors=["A"])
            self.node = SimpleNamespace(state="b0", actions=[])
        elif name == "finish_stock":
            self.reactions.append(dict(occurrence="0", product="A", precursors=["T"]))
            self.stock_leaves.append("T")
            self.status = "stock_solved"

    def finish_report(self):
        return dict(status=self.status, reactions=self.reactions, stock_leaves=self.stock_leaves)


def run(names, cap):
    agenda = iter(names)
    seen = []

    def sample(request):
        seen.append(request)
        name = next(agenda)
        return dict(raw=name, generated_token_ids=[1], request_id=str(len(seen)))

    result = controllers.electron_frontier(
        Episode(), controllers.RequestBudget(sample, cap),
        action=lambda raw: "apply_electron_flow" if raw.startswith("flow") else "finish_trace",
        grounded_key=lambda state, raw: raw,
    )
    return result, seen


def test_cycle_backs_up_to_an_authentic_pre_finish_sibling():
    result, seen = run(["flow_b", "flow_good", "finish_b", "finish_cycle", "finish_stock"], 5)
    assert seen == [("0", "a0"), ("0", "a0"), ("0", "flow_b"), ("0.0", "b0"), ("0", "flow_good")]
    assert result["requests"] == 5
    assert result["report"]["stock_leaves"] == ["T"]
    assert result["report"]["reactions"][0]["precursors"] == ["T"]
    assert result["backjumps"][0]["from_occurrence"] == "0.0"
    assert result["backjumps"][0]["to_occurrence"] == "0"
    assert result["backjumps"][0]["blocked_precursors"] == ["B", "S"]
    assert result["backjumps"][0]["replacement_receipt"] == "2"


def test_duplicate_flow_is_charged_and_a_new_draw_obeys_the_global_cap():
    result, seen = run(["flow_b", "flow_b", "finish_b", "finish_cycle", "flow_good"], 5)
    assert len(seen) == result["requests"] == 5
    assert result["duplicate_draws"] == 1
    assert seen[-1] == ("0", "a0")
    assert result["status"] == "resource_censored"
    assert len(result["report"]["reactions"]) == 0


def test_compact_context_uses_actual_route_relationships_only():
    e = Episode()
    e.accept({"raw": "finish_b"})
    e.pending = [{"product": "X", "occurrence":"0.2", "future_reference":"forbidden"}]
    memory = controllers.compact_context(e)
    assert memory["root"] == "A" and memory["ancestors"] == ["A"]
    assert memory["current"] == {"product": "B", "occurrence": "0.0"}
    assert memory["pending"] == [{"product": "X", "occurrence": "0.2"}]
    assert memory["completed"] == [{"product": "A", "precursors": ["B", "S"], "occurrence": "0"}]
    assert memory["stock_leaves"] == ["S"]
