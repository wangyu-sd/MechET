"""Fixed-actor PR117 controls over native, executor-owned chemistry."""
from dataclasses import dataclass, field
import json

from scripts import pr117_reaction_first

C0 = "C0_original"
C1 = "C1_compact_route_context"
C2 = "C2_early_FLOW_and_backjump"
C4 = "C4_completed_reaction_first"


class RequestBudget:
    def __init__(self, backend, cap):
        if type(cap) is not int or cap < 1:
            raise ValueError("positive request ceiling required")
        self.backend = backend
        self.cap = cap
        self.requests = self.tokens = 0
        self.receipts = []

    @property
    def exhausted(self):
        return self.requests >= self.cap

    def sample(self, request):
        if self.exhausted:
            raise RuntimeError("request ceiling exceeded")
        self.requests += 1
        receipt = self.backend(request)
        self.receipts.append(receipt)
        self.tokens += len(receipt["generated_token_ids"])
        return receipt


def compact_context(episode):
    """Project committed, public route evidence into model input."""
    def obligation(value):
        return {key: value[key] for key in ("product", "occurrence")}

    return dict(root=episode.public_root, current=obligation(episode.current),
                ancestors=list(episode.current["ancestors"]),
                completed=[{key: reaction[key] for key in ("product", "precursors", "occurrence")}
                           for reaction in episode.reactions],
                pending=[obligation(item) for item in episode.pending],
                stock_leaves=list(episode.stock_leaves),
                conflict=episode.cfg.get("pr117_conflict"))


def contextual_episode_type(native_type):
    class ContextEpisode(native_type):
        def _messages(self):
            messages = super()._messages()
            memory = json.dumps(compact_context(self), ensure_ascii=False, separators=(",", ":"))
            memory = memory.replace("<", r"\u003c").replace(">", r"\u003e")
            messages[0]["content"] += (
                "\n\nROUTE CONTEXT (observed executor history)\n" + memory
                + "\nUse only the current atom inventory for electron actions. "
                  "A completed reaction must advance this branch without regenerating an ancestor."
            )
            return messages

    return ContextEpisode


def terminal_report(episode, budget):
    if episode.status is not None:
        return episode.finish_report()
    if not budget.exhausted:
        raise ValueError("open route without a resource stop")
    return dict(status="resource_censored", error="PR117_REQUEST_BUDGET", reactions=episode.reactions,
                pending=episode.pending, current=episode.current, stock_leaves=episode.stock_leaves,
                records=episode.records)


def policy_path(episode, budget):
    while not budget.exhausted:
        request = episode.next_decision()
        if request is None:
            break
        episode.accept(budget.sample(request))
    report = terminal_report(episode, budget)
    return dict(status=report["status"], report=report)


@dataclass
class FlowFrontier:
    parent: object
    occurrence: str
    actions: set = field(default_factory=set)
    states: set = field(default_factory=set)
    unused: list = field(default_factory=list)
    draws: int = 0


def electron_frontier(episode, budget, *, action, grounded_key):
    """Fork at the root's first proposed FLOW and backjump on a causal ancestor cycle."""
    frontier = None
    backjumps = []
    proposals = []
    duplicate_draws = 0
    root_uid = episode.uid

    def trial(parent):
        return parent.fork(f"{root_uid}/pr117-c2-{budget.requests + 1}")

    def propose(frame, branch, receipt):
        nonlocal duplicate_draws
        frame.draws += 1
        raw = receipt["raw"]
        record = dict(receipt=receipt["request_id"], occurrence=frame.occurrence, alternative=False)
        proposals.append(record)
        if action(raw) != "apply_electron_flow":
            # Keep the authentic reply and its cost even outside the FLOW domain.
            if not receipt.get("resource_error") and not receipt.get("truncated"):
                record["reason"] = "not_a_flow_proposal"
                return False
        key = grounded_key(frame.parent.node.state, raw)
        if key in frame.actions:
            record["reason"] = "duplicate_action"
            duplicate_draws += 1
            return False
        frame.actions.add(key)
        branch.accept(receipt)
        if branch.status is not None:
            record["reason"] = branch.error or branch.status
            return False
        successor = branch.node.state
        if successor in frame.states:
            record["reason"] = "duplicate_executed_state"
            duplicate_draws += 1
            return False
        frame.states.add(successor)
        frame.unused.append((branch, receipt["request_id"]))
        record.update(alternative=True, executable_successor=successor)
        return True

    while True:
        request = episode.next_decision()
        if episode.status == "rejected_action" and frontier is not None:
            error = episode.error or ""
            # The first pilot expands only the root before FINISH. A descendant
            # must actually return to that ancestor to invalidate the root edge.
            if error.startswith("ROUTE_CYCLE:") and episode.current:
                repeated = set(error.removeprefix("ROUTE_CYCLE:").split(","))
                if frontier.parent.current["product"] in repeated:
                    origin = next((r for r in episode.reactions if r["occurrence"] == frontier.occurrence), None)
                    if origin is not None:
                        conflict = dict(from_occurrence=episode.current["occurrence"],
                                       to_occurrence=frontier.occurrence, repeated=sorted(repeated),
                                       blocked_precursors=list(origin["precursors"]))
                        if not frontier.unused and frontier.draws < 3 and not budget.exhausted:
                            retry = trial(frontier.parent)
                            retry.cfg["pr117_conflict"] = conflict
                            propose(frontier, retry, budget.sample(retry.next_decision()))
                        if frontier.unused:
                            episode, receipt_id = frontier.unused.pop(0)
                            episode.cfg["pr117_conflict"] = conflict
                            backjumps.append(dict(conflict, replacement_receipt=receipt_id))
                            continue
        if request is None or budget.exhausted:
            break
        if frontier is None and episode.current["occurrence"] == "0":
            branch = trial(episode)
            receipt = budget.sample(branch.next_decision())
            if action(receipt["raw"]) == "apply_electron_flow":
                frontier = FlowFrontier(episode, episode.current["occurrence"])
                primary = propose(frontier, branch, receipt)
                if not budget.exhausted:
                    sibling = trial(episode)
                    propose(frontier, sibling, budget.sample(sibling.next_decision()))
                if frontier.unused:
                    episode, _ = frontier.unused.pop(0)
                elif not primary:
                    episode = branch
            else:
                branch.accept(receipt)
                episode = branch
        else:
            episode.accept(budget.sample(request))

    report = terminal_report(episode, budget)
    return dict(status=report["status"], report=report, requests=budget.requests,
                generated_tokens=budget.tokens, backjumps=backjumps,
                duplicate_draws=duplicate_draws, frontier_proposals=proposals)


def solve(arm, episode, backend, *, cap, action=None, grounded_key=None):
    budget = RequestBudget(backend, cap)
    if arm == C4:
        result = pr117_reaction_first.search(episode, budget.sample, request_cap=cap)
    elif arm == C2:
        result = electron_frontier(episode, budget, action=action, grounded_key=grounded_key)
    elif arm in (C0, C1):
        result = policy_path(episode, budget)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return dict(result, requests=budget.requests, generated_tokens=budget.tokens,
                decision_receipts=budget.receipts)
