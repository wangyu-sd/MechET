"""Completed-reaction-first control using the native MechET episode and sampler.

Each trial runs one reaction independently from its parent. Only committed
precursor multisets become alternatives. All samples share a target-level
budget, including rejected actions, duplicate completions and unfinished trials.
"""


def search(episode, sample, *, request_cap, width=2):
    if request_cap < 1 or width < 1:
        raise ValueError("request_cap and width must be positive")
    root = episode.uid
    requests = tokens = backtracks = duplicates = trial_number = 0
    budget_stop = False
    alternatives = []
    attempts = []
    last = episode

    while True:
        if episode.status in ("stock_solved", "already_stock"):
            last = episode
            break
        request = episode.next_decision()
        if request is None:
            last = episode
        elif requests >= request_cap:
            budget_stop = True
            last = episode
        else:
            if episode.node.actions:
                raise ValueError("C4 may branch only at a reaction boundary")
            completed = {}
            starting_count = len(episode.reactions)
            for _ in range(width):
                if requests >= request_cap:
                    budget_stop = True
                    break
                trial_number += 1
                trial = episode.fork(f"{root}/pr117-c4-trial-{trial_number}")
                tried = []
                while len(trial.reactions) == starting_count:
                    current = trial.next_decision()
                    if current is None:
                        break
                    if requests >= request_cap:
                        budget_stop = True
                        break
                    requests += 1
                    receipt = sample(current)
                    tried.append(receipt)
                    tokens += len(receipt["generated_token_ids"])
                    trial.accept(receipt)
                last = trial
                committed = len(trial.reactions) > starting_count
                attempts.append(dict(trial=trial_number, starting_reactions=starting_count,
                                     receipts=tried, committed=committed, status=trial.status))
                if not committed:
                    continue
                precursors = tuple(sorted(trial.reactions[-1]["precursors"]))
                if precursors in completed:
                    duplicates += 1
                    continue
                completed[precursors] = trial
                if trial.status == "stock_solved":
                    break
            if completed:
                # Fixed proposal order breaks ties. Whole-route closure, if
                # observed within an executed trial, takes precedence.
                candidates = sorted(completed.values(), key=lambda t: t.status != "stock_solved")
                episode, *others = candidates
                alternatives.append(others)
                continue
        while alternatives and not alternatives[-1]:
            alternatives.pop()
        if not alternatives:
            break
        episode = alternatives[-1].pop(0)
        backtracks += 1

    status = last.status
    if status not in ("stock_solved", "already_stock") and budget_stop:
        status = "resource_censored"
    if last.status is None:
        report = dict(status=status, reactions=last.reactions,
                      pending=getattr(last, "pending", []), current=getattr(last, "current", None))
    else:
        report = last.finish_report()
    return dict(status=status, report=report, requests=requests, generated_tokens=tokens,
                duplicate_precursor_sets=duplicates, backtracks=backtracks, attempts=attempts)
