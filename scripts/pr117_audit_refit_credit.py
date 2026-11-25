"""Join executor-replayed exact-state forks back to the pinned learner inputs."""
import argparse
import hashlib
import json
from pathlib import Path

from mechet import route_adaptive_credit, route_stepwise


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verified_metric(root, step, contract_sha):
    path = root / "checkpoints" / f"round-{step:06d}" / "complete.json"
    raw = path.read_bytes()
    seal = json.loads(raw)
    if (seal["contract_sha256"] != contract_sha or seal["optimizer_step"] != step
            or seal["next_round"] != step):
        raise ValueError(f"Checkpoint contract mismatch at {step}")
    learners = [m for m in seal["round_metrics"] if m.get("algorithm") == "adaptive_finish_refit_po_v1"]
    if len(learners) != 1 or learners[0]["optimizer_updated"] is not True:
        raise ValueError(f"Expected a durable update of the observed adaptive learner at {step}")
    m = learners[0]
    return dict(optimizer_step=step, complete_sha256=sha(raw), learner_roots=m["learner_roots"],
                policy_groups=m["policy_groups"], historical_comparison_slates=m["historical_comparison_slates"],
                route_credit_slates=m["route_credit_slates"], historical_pg_loss=m["historical_pg_loss"])


def audit(args):
    raw = args.contract.read_bytes()
    contract_sha, cfg = sha(raw), json.loads(raw)
    root, code = Path(cfg["output_dir"]).resolve(), Path(cfg["code_root"]).resolve()
    source_digests = {}
    for name in ("route_adaptive_credit", "route_stepwise", "route_stepwise_learner"):
        rel = f"src/mechet/{name}.py"
        module = globals().get(name)
        path = code / rel
        if module and Path(module.__file__).resolve() != path:
            raise ValueError(f"Wrong imported learner runtime: {name}")
        source_digests[rel] = sha(path.read_bytes())
        if source_digests[rel] != cfg["code_files_sha256"][rel]:
            raise ValueError(f"Learner source hash differs: {name}")

    weights = cfg["stepwise"]
    rows, evidence = [], {}
    updates = {step: verified_metric(root, step, contract_sha) for step in range(1, args.through + 1)}
    for source in args.witnesses:
        archived_raw = source.read_bytes()
        archived = json.loads(archived_raw)
        evidence[source.name] = sha(archived_raw)
        if archived["actual_source_contract_sha256"] != contract_sha or archived["new_model_calls"] != 0:
            raise ValueError("Witnesses must derive from the pinned original run")
        for w in archived["witnesses"]:
            target, step = w["target_id"], w["training_round"] + 1
            broker = root / f"round-{step-1:06d}"
            claims = (json.loads(line) for line in (broker / "claims.jsonl").read_text().splitlines())
            matches = [event for event in claims if event.get("task_id") == target and "result_path" in event]
            if len(matches) != 1:
                raise ValueError("Expected one immutable broker result for the target")
            claim = matches[0]
            result_path = (broker / claim["result_path"]).resolve()
            if not result_path.is_relative_to(broker / "groups"):
                raise ValueError("Learner result is outside its immutable broker directory")
            report_raw = result_path.read_bytes()
            if sha(report_raw) != claim["result_sha"] or sha(report_raw) != w["source_sha256"]:
                raise ValueError("Replayed witness and broker artifact hashes differ")
            group = json.loads(report_raw)
            (report,) = group
            if report["task_id"] != target or report["behavior_version"] != claim["actor_version"]:
                raise ValueError("Witness target and learner group differ")
            entries = {}
            for slate, route_credit in route_adaptive_credit.adaptive_slates([group], weights):
                for candidate, local_adv, hist_adv in slate:
                    receipt = candidate["request_id"]
                    if receipt in entries:
                        raise ValueError("Learner scored a receipt twice")
                    entries[receipt] = dict(local_score=candidate["local_score"],
                                            teacher_score=candidate["teacher_score"],
                                            local_adv=local_adv, history_adv=hist_adv,
                                            weighted_history_adv=hist_adv * weights["history_weight"],
                                            slate_route_credit=route_credit,
                                            prompt_token_count=len(candidate["prompt_token_ids"]))
            for fork in w["witnesses"]:
                closed, failed = fork["closed"]["receipt"], fork["failed"]["receipt"]
                index = {}
                for slate in report["stepwise"]["slates"]:
                    for i, candidate in enumerate(slate["candidates"]):
                        receipt = candidate["request_id"]
                        if receipt in (closed, failed):
                            index[receipt] = slate, i, candidate
                sc, ic, cc = index[closed]
                sf, i_f, cf = index[failed]
                if (sc is not sf or sc["outcomes"].get(str(ic)) != "closed" or sf["outcomes"].get(str(i_f)) != "failed"
                        or cc["prompt_token_ids"] != cf["prompt_token_ids"] or cc["action_key"] == cf["action_key"]
                        or not cc["accepted"] or not cf["accepted"] or entries[closed]["history_adv"] <= 0
                        or entries[failed]["history_adv"] >= 0):
                    raise ValueError("Replayed pair did not receive opposite archived learner credit")
                rows.append(dict(target_id=target, optimizer_step=step, result_sha256=sha(report_raw),
                                 closed_receipt=closed, failed_receipt=failed, report_scalar_reward=report["reward"],
                                 report_reward_status=report["reward_evidence"]["reward_status"],
                                 closure_supported_by_train_reference=report["reward_evidence"]["verified_stock_closed"],
                                 raw_teacher_prefers_failed=entries[failed]["teacher_score"] > entries[closed]["teacher_score"],
                                 teacher_blended_local_score_prefers_failed=entries[failed]["local_score"] > entries[closed]["local_score"],
                                 closed=entries[closed], failed=entries[failed]))

    if len({r["target_id"] for r in rows}) != len(rows):
        raise ValueError("Current analysis requires one distinct witness per case")
    return dict(scope="CPU reconstruction from authentic scored groups and durable optimizer summaries. No model generation, gradient rerun or checkpoint performance measurement.",
                contract_sha256=contract_sha, source_sha256=source_digests, witness_files_sha256=evidence,
                history_weight=weights["history_weight"], cases=len(rows), verified_optimizer_updates=len(updates),
                score_order_reversed_by_history=sum(r["teacher_blended_local_score_prefers_failed"] for r in rows),
                raw_teacher_prefers_failed=sum(r["raw_teacher_prefers_failed"] for r in rows),
                structural_closed_without_reference=sum(not r["closure_supported_by_train_reference"] for r in rows),
                zero_scalar_rewards=sum(r["report_scalar_reward"] == 0 for r in rows),
                published_totals={k:sum(update[k] for update in updates.values()) for k in (
                    "learner_roots", "policy_groups", "historical_comparison_slates", "route_credit_slates")},
                updates=list(updates.values()), witness_credit=rows)


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--contract", type=Path, required=True)
    pa.add_argument("--witnesses", type=Path, nargs="+", required=True)
    pa.add_argument("--through", type=int, required=True)
    pa.add_argument("--output", type=Path, required=True)
    args = pa.parse_args()
    evidence = audit(args)
    with args.output.open("x") as out:
        out.write(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k:evidence[k] for k in ("cases", "verified_optimizer_updates", "raw_teacher_prefers_failed",
        "score_order_reversed_by_history", "zero_scalar_rewards", "published_totals")}))


if __name__ == "__main__":
    main()
