"""Reconstruct every nonzero historical comparison from the original learner inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def scored_closure_groups(folder, contract_sha, learner_roots, stock_closed):
    # The original publisher puts the root status ahead of the large action records.
    pattern = re.compile(rb'"behavior_version"\s*:\s*"[^"]+"\s*,\s*"status"\s*:\s*"([^"]+)"')
    claims = [row for line in (folder / "claims.jsonl").read_bytes().splitlines()
              if (row := json.loads(line))["event"] == "finished"]
    if len(claims) != learner_roots or len({row["task_id"] for row in claims}) != learner_roots:
        raise ValueError("Broker group population differs from the durable update")
    counts = Counter()
    for claim in claims:
        path = (folder / claim["result_path"]).resolve()
        if not path.is_relative_to(folder / "groups") or claim["contract_sha"] != contract_sha:
            raise ValueError("Wrong broker group origin")
        with path.open("rb") as f:
            header = f.read(4096)
            match = pattern.search(header.split(b'"reactions"', 1)[0])
            if match is not None and match.group(1) not in (b"stock_solved", b"already_stock"):
                counts["other_statuses"] += 1
                continue
            raw = header + f.read()
        if sha(raw) != claim["result_sha"]:
            raise ValueError("Content hash mismatch in a closure candidate")
        (report,) = json.loads(raw)
        if report["task_id"] != claim["task_id"] or report["behavior_version"] != claim["actor_version"]:
            raise ValueError("Closure report and scheduler claim differ")
        counts[report["status"]] += 1
        if report["status"] in ("stock_solved", "already_stock"):
            yield report, claim["result_sha"]
    if counts["stock_solved"] != stock_closed:
        raise ValueError("Header-based filtering disagrees with checkpoint closure totals")


def signaled_slates(report, spec):
    from mechet.route_action_feedback import semantic_action
    from mechet.route_adaptive_credit import adaptive_slates

    slate_by_receipt = {candidate["request_id"]: slate for slate in report["stepwise"]["slates"]
                        for candidate in slate["candidates"]}
    for entries, route_credit in adaptive_slates([[report]], spec):
        signaled = [(c, local, h) for c, local, h in entries if h != 0]
        if not signaled:
            continue
        slate = slate_by_receipt[signaled[0][0]["request_id"]]
        by_index = {c["request_id"]: idx for idx, c in enumerate(slate["candidates"])}
        selection = {c["index"]: c["request_id"] for c in slate["selections"]}
        prompts, occurrences, action_keys = set(), set(), set()
        actions = []
        for c, local, history in signaled:
            receipt = c["request_id"]
            i = by_index[receipt]
            outcome = slate["outcomes"][str(i)]
            if (not c["accepted"] or c["censored"] or c["duplicate_of"] is not None
                    or selection[i] != receipt or outcome != ("closed" if history > 0 else "failed")
                    or c["behavior_version"] != report["behavior_version"]):
                raise ValueError("Historical credit requires a genuinely attempted accepted action")
            act = semantic_action(c["raw"])
            if act is None:
                raise ValueError("Selected action is missing its native tool identity")
            prompts.add(sha(json.dumps(c["prompt_token_ids"]).encode()))
            occurrences.add(c["occurrence"])
            action_keys.add(c["action_key"])
            actions.append(dict(receipt=receipt, tool=act[1]["name"], outcome=outcome,
                                teacher_score=c["teacher_score"], local_score=c["local_score"],
                                local_advantage=local, history_advantage=history,
                                weighted_history_advantage=history*spec["history_weight"]))
        if len(prompts) != 1 or len(occurrences) != 1 or len(action_keys) != len(actions):
            raise ValueError("Candidate actions do not share one exact context with different grounded identities")
        yield dict(slate_key=slate["key"], occurrence=next(iter(occurrences)), prompt_sha256=next(iter(prompts)),
                   candidate_count=len(slate["candidates"]), differently_grounded_candidates=len(action_keys),
                   unattempted_candidates=len(slate["candidates"])-len(slate["selections"]),
                   route_or_process_credit=route_credit, actions=actions)


def census(contract, through, checkpoint_sha):
    raw = contract.read_bytes()
    cfg, contract_sha = json.loads(raw), sha(raw)
    root, code = Path(cfg["output_dir"]).resolve(), Path(cfg["code_root"]).resolve()
    from mechet import route_adaptive_credit, route_stepwise

    for module in (route_adaptive_credit, route_stepwise):
        path = Path(module.__file__).resolve()
        relative = str(path.relative_to(code))
        if sha(path.read_bytes()) != cfg["code_files_sha256"][relative]:
            raise ValueError("Loaded learner code differs from the original training contract")
    # Record each completed checkpoint. Only positive rounds need group reads.
    rounds, rows = [], []
    for step in range(1, through+1):
        complete_raw = (root / "checkpoints" / f"round-{step:06d}" / "complete.json").read_bytes()
        complete = json.loads(complete_raw)
        if complete["contract_sha256"] != contract_sha or complete["optimizer_step"] != step:
            raise ValueError("Expected original immutable learner checkpoint")
        parts = complete["round_metrics"]
        learner = parts[cfg["role_partition"]["learner"]]
        if learner["algorithm"] != "adaptive_finish_refit_po_v1" or not learner["optimizer_updated"]:
            raise ValueError("Expected completed adaptive update")
        total = learner["historical_comparison_slates"]
        found, read_groups = 0, 0
        if total:
            folder = root / f"round-{step-1:06d}"
            closed = sum(p["native_stock_solved"] for p in parts)
            for report, source_sha in scored_closure_groups(folder, contract_sha, learner["learner_roots"], closed):
                read_groups += 1
                for row in signaled_slates(report, cfg["stepwise"]):
                    evidence = report["reward_evidence"]
                    rows.append(dict(row, update=step, target_id=report["task_id"], scored_group_sha256=source_sha,
                                     train_lane=report["stepwise"]["lane"], structural_closed=evidence["structural_stock_closed"],
                                     reference_verified=evidence["verified_stock_closed"],
                                     committed_reactions=len(report["reactions"]), report_reward=report["reward"]))
                    found += 1
        if found != total:
            raise ValueError(f"Learner reported {total} historical groups at update {step}, census found {found}")
        rounds.append(dict(update=step, complete_sha256=sha(complete_raw), candidates=learner["policy_groups"],
                           historical_candidates=total, target_episodes=learner["learner_roots"],
                           fully_read_stock_closure_groups=read_groups))
    if rounds[-1]["complete_sha256"] != checkpoint_sha:
        raise ValueError("Census cutoff differs from the requested verified checkpoint")

    by_type, pairwise = Counter(), Counter()
    for row in rows:
        tools = sorted({action["tool"] for action in row["actions"]})
        by_type["+".join(tools)] += 1
        for closed in [a for a in row["actions"] if a["outcome"] == "closed"]:
            for failed in [a for a in row["actions"] if a["outcome"] == "failed"]:
                pairwise["closed_failed_pairs"] += 1
                if closed["tool"]==failed["tool"]=="apply_electron_flow":
                    pairwise["flow_flow_pairs"] += 1
                    if closed["teacher_score"] is not None and failed["teacher_score"] is not None:
                        pairwise["flow_pairs_with_two_teacher_scores"] += 1
                        pairwise["flow_pairs_teacher_favored_failed"] += failed["teacher_score"] > closed["teacher_score"]
    return dict(scope="Exact learner inputs for the completed adaptive TRAIN updates, no new executor or model calls",
                runtime_contract_sha256=contract_sha, through_update=through, cutoff_sha256=checkpoint_sha,
                target_episodes=sum(r["target_episodes"] for r in rounds),
                action_candidate_groups=sum(r["candidates"] for r in rounds), historical_comparison_groups=len(rows),
                action_type_groups=dict(by_type), pairwise=dict(pairwise),
                distinct_targets=len({r["target_id"] for r in rows}),
                lanes=dict(Counter(r["train_lane"] for r in rows)),
                compatible_reference_groups=sum(r["reference_verified"] for r in rows),
                groups_with_unattempted_candidates=sum(r["unattempted_candidates"]>0 for r in rows),
                new_model_calls=0, rounds=rounds, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--through", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.through < 1 or not output.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Choose a completed checkpoint and shared-storage evidence path")
    result = census(args.runtime_contract, args.through, args.checkpoint_sha256)
    with output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rounds","rows")}))


if __name__ == "__main__":
    main()
