"""Try a different native FLOW at a replay-authenticated downstream failure."""

from dataclasses import asdict
import json

from scripts.pr117_controllers import RequestBudget, terminal_report


FLOW = "apply_electron_flow"


def conditioning(request):
    return {key: value for key, value in asdict(request).items() if key not in ("request_id", "episode_id")}


def with_executor_feedback(messages, error):
    if not messages or messages[0].get("role") != "system" or not error:
        raise ValueError("Expected an original system message and a native executor error")
    text = json.dumps(error, ensure_ascii=False).replace("<", r"\u003c").replace(">", r"\u003e")
    added = "\n\nEXECUTOR FEEDBACK\nThe previous proposal was rejected. No state change was committed.\n" + text
    return [dict(messages[0], content=messages[0]["content"] + added), *messages[1:]]


def condition_retry(episode, error):
    prefix_records = len(episode.records)

    class FeedbackEpisode(type(episode)):
        def _messages(self):
            messages = super()._messages()
            if len(self.records) == prefix_records:
                return with_executor_feedback(messages, error)
            return messages

    episode.__class__ = FeedbackEpisode


def recover(snapshot, failure, sample, *, cap, action, grounded_key, executor_error=None):
    original = snapshot.next_decision()
    occurrence = snapshot.current["occurrence"]
    product = snapshot.current["product"]
    if (original is None or not snapshot.reactions or occurrence == "0"
            or original.request_id != failure["request_id"] or action(failure["raw"]) != FLOW):
        raise ValueError("A rejected native FLOW at an authenticated downstream state is required")
    budget = RequestBudget(sample, cap)
    state = snapshot.node.state
    observed_actions = {grounded_key(state, failure["raw"])}
    tried = []
    active = None
    expected = conditioning(original)
    if executor_error is not None:
        expected["messages"] = with_executor_feedback(expected["messages"], executor_error)
    conditioned_tokens = None
    for index in range(1, 3):
        if budget.exhausted:
            break
        candidate = snapshot.fork(f"{original.episode_id}/pr117-recovery-{index}")
        if executor_error is not None:
            condition_retry(candidate, executor_error)
        request = candidate.next_decision()
        if (conditioning(request) != expected
                or request.request_id == original.request_id or candidate.node.state != state):
            raise ValueError("A retry must preserve the electronic state and registered policy input")
        receipt = budget.sample(request)
        if (receipt["request_id"] != request.request_id
                or receipt["behavior_version"] != original.behavior_version):
            raise ValueError("Retry receipt has a different request or policy identity")
        tokens = receipt["prompt_token_ids"]
        if executor_error is None and tokens != failure["prompt_token_ids"]:
            raise ValueError("Retry must use the original prompt tokens")
        if executor_error is not None:
            if tokens == failure["prompt_token_ids"] or (conditioned_tokens is not None and tokens != conditioned_tokens):
                raise ValueError("Each retry must use the registered feedback-conditioned token prefix")
            conditioned_tokens = tokens
        trial = dict(request_id=receipt["request_id"], outcome="not_a_complete_flow")
        tried.append(trial)
        if receipt.get("truncated") or receipt.get("resource_error") or action(receipt["raw"]) != FLOW:
            continue
        key = grounded_key(state, receipt["raw"])
        if key in observed_actions:
            trial["outcome"] = "duplicate_grounded_action"
            continue
        observed_actions.add(key)
        candidate.accept(receipt)
        record = candidate.records[-1]
        if not record.get("accepted") or record.get("state_after") in (None, state):
            trial.update(outcome="executor_rejected_or_unchanged", error=record.get("error"))
            continue
        trial["outcome"] = "distinct_executable_flow"
        active = candidate
        break

    first_commit = None
    first_stock = None
    if active is None:
        report = dict(status="no_distinct_executable_retry", reactions=snapshot.reactions,
                      records=snapshot.records)
    else:
        while not budget.exhausted:
            request = active.next_decision()
            if request is None:
                break
            active.accept(budget.sample(request))
            if first_commit is None and any(edge["occurrence"] == occurrence and edge["product"] == product
                                            for edge in active.reactions):
                first_commit = budget.requests
            if active.status == "stock_solved":
                first_stock = budget.requests
                break
        report = terminal_report(active, budget)
    return dict(status=report["status"], recovered_reaction=first_commit is not None,
                first_recovered_reaction_call=first_commit, first_stock_solution_call=first_stock,
                requests=budget.requests, generated_tokens=budget.tokens, report=report,
                proposals=tried, decision_receipts=budget.receipts)
