"""Controller state-machine tests. Scripted responses stand in for actor calls."""
from copy import deepcopy
from types import SimpleNamespace

from scripts import pr117_reaction_first as runner


class Episode:
    def __init__(self):
        self.uid = "route"
        self.reactions = []
        self.records = []
        self.status = None
        self.node = SimpleNamespace(actions=[])
        self.current = {"occurrence": "0"}

    def next_decision(self):
        return None if self.status else (len(self.reactions), len(self.node.actions))

    def fork(self, namespace):
        assert not self.node.actions, "Reaction-first can only branch at reaction starts"
        result = deepcopy(self)
        result.uid = namespace
        return result

    def accept(self, receipt):
        self.records.append(receipt)
        kind = receipt["kind"]
        if kind == "reject":
            self.status = "rejected_action"
        elif kind == "finish":
            self.reactions.append({"precursors": receipt["precursors"]})
            self.node.actions = []
            self.status = receipt.get("end")
        else:
            self.node.actions.append(kind)

    def finish_report(self):
        return dict(status=self.status, reactions=self.reactions)


def response(kind, *precursors, end=None):
    return dict(kind=kind, precursors=precursors, end=end,
                generated_token_ids=[1, 2], behavior_logprobs=[-0.1, -0.2])


def test_only_completed_reactions_are_forked_and_backtracked():
    # Two independent complete reactions at the root. The first completed
    # precursor route stalls and the second finishes two reactions in total.
    agenda = iter([response("flow"), response("finish", "A"), response("flow"),
                   response("finish", "B"), response("reject"), response("reject"),
                   response("flow"), response("finish", "stock", end="stock_solved")])
    result = runner.search(Episode(), lambda _: next(agenda), request_cap=8)
    assert result["report"]["status"] == "stock_solved"
    assert [x["precursors"] for x in result["report"]["reactions"]] == [("B",), ("stock",)]
    assert result["requests"] == 8
    assert result["generated_tokens"] == 16
    assert result["backtracks"] == 1


def test_budget_cannot_convert_unfinished_flow_to_a_reaction():
    agenda = iter([response("flow")])
    result = runner.search(Episode(), lambda _: next(agenda), request_cap=1)
    assert result["status"] == "resource_censored"
    assert result["requests"] == 1
    assert not result["report"]["reactions"]


def test_duplicate_completed_precursors_are_one_route_choice():
    agenda = iter([response("flow"), response("finish", "A"), response("flow"),
                   response("finish", "A"), response("reject"), response("reject")])
    result = runner.search(Episode(), lambda _: next(agenda), request_cap=7)
    assert result["requests"] == 6
    assert result["status"] == "rejected_action"
    assert result["duplicate_precursor_sets"] == 1
    assert result["backtracks"] == 0
