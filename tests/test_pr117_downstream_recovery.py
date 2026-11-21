from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.pr117_downstream_recovery import recover


@dataclass
class Request:
    request_id: str
    episode_id: str
    messages: list
    behavior_version: str = "actor"


class Episode:
    def __init__(self):
        self.uid = "route"
        self.node = SimpleNamespace(state="s")
        self.current = dict(occurrence="0.1", product="B")
        self.reactions = [dict(occurrence="0", product="A")]
        self.records = []
        self.status = self.error = None

    def fork(self, name):
        ep = deepcopy(self)
        ep.uid = name
        return ep

    def next_decision(self):
        if self.status:
            return None
        return Request(f"{self.uid}/{len(self.records)}", self.uid, [self.node.state])

    def accept(self, receipt):
        action = receipt["raw"]
        self.records.append(dict(receipt, accepted=True))
        if action == "finish":
            self.reactions.append(dict(occurrence="0.1", product="B"))
            self.status = "stock_solved"
        elif action == "alternative":
            self.node.state = "next"
            self.records[-1]["state_after"] = "next"
        else:
            raise AssertionError("The duplicate should never execute")

    def finish_report(self):
        return dict(status=self.status, error=self.error, reactions=self.reactions, records=self.records)


def run(episode, answers, cap):
    seen = []
    slate = iter(answers)

    def native_sample(req):
        seen.append(req)
        raw, prompt = next(slate)
        return dict(request_id=req.request_id, behavior_version="actor", raw=raw,
                    prompt_token_ids=prompt, generated_token_ids=[1])

    original = dict(request_id="route/0", behavior_version="actor", raw="bad",
                    prompt_token_ids=[11], generated_token_ids=[1])
    result = recover(episode, original, native_sample, cap=cap,
                     action=lambda raw: "finish_trace" if raw == "finish" else "apply_electron_flow",
                     grounded_key=lambda state, raw: (state, raw))
    return result, seen


def test_duplicate_counts_against_budget_and_preserves_same_input_for_retry():
    ep = Episode()
    result, seen = run(ep, [("bad", [11]), ("alternative", [11]), ("finish", [22])], 3)
    assert result["status"] == "stock_solved"
    assert result["recovered_reaction"] and result["requests"] == 3
    assert [x.messages for x in seen] == [["s"], ["s"], ["next"]]
    assert len({x.request_id for x in seen}) == 3
    assert ep.records == [] and ep.node.state == "s"


def test_two_identical_draws_stop_without_reexecuting_the_failure():
    result, seen = run(Episode(), [("bad", [11]), ("bad", [11])], 5)
    assert result["status"] == "no_distinct_executable_retry"
    assert not result["recovered_reaction"] and result["requests"] == len(seen) == 2


def test_a_retry_prompt_change_is_a_protocol_failure():
    with pytest.raises(ValueError, match="original prompt tokens"):
        run(Episode(), [("alternative", [99])], 3)
