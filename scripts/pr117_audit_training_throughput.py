"""Summarize physical dispatches and learner phases from completed ReFIT updates."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median


def hashed_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def collect(contract, through, expected_complete):
    cfg, contract_sha = hashed_json(contract)
    root = Path(cfg["output_dir"])
    recovered, recovery_sha = hashed_json(Path(cfg["recovered_initial_round"]))
    roles = cfg["role_partition"]
    if cfg["inference_backend"] != "vllm":
        raise ValueError("Expected the pinned native vLLM training backend")
    role_counts = {"learner": 1, "actor": len(roles["actors"]), "scorer": len(roles["scorers"])}
    rows, order, last_mtime = [], None, None
    for step in range(1, through + 1):
        directory = root / "checkpoints" / f"round-{step:06d}"
        complete, complete_sha = hashed_json(directory / "complete.json")
        round_dir = root / f"round-{step-1:06d}"
        broker, broker_sha = hashed_json(round_dir / "scheduler_summary.json")
        if order is None:
            order = complete["training_order_sha256"]
        if (complete["contract_sha256"] != contract_sha or complete["optimizer_step"] != step
                or complete["next_round"] != step or complete["training_order_sha256"] != order):
            raise ValueError(f"Durable optimizer contract changed at update {step}")
        parts = complete["round_metrics"]
        if Counter(p["hardware_role"] for p in parts) != role_counts:
            raise ValueError(f"The training role placement differs at update {step}")
        learner = parts[roles["learner"]]
        if (learner["algorithm"] != "adaptive_finish_refit_po_v1" or not learner["optimizer_updated"]
                or learner["learner_roots"] != broker["expected"] or broker["complete"] != broker["expected"]):
            raise ValueError(f"Incomplete training update {step}")
        counts = Counter()
        for part in parts:
            counts.update(part["adaptive_round"]["lanes"])
        actor_calls = [parts[rank]["backend_completed_requests"] for rank in roles["actors"]]
        physical = sum(actor_calls)
        model_records = sum(p["model_decision_requests"] for p in parts)
        if physical != model_records or sum(counts.values()) != learner["learner_roots"]:
            raise ValueError(f"Native calls or training lanes are inconsistent at update {step}")
        if step == 1:
            if physical != recovered["recovered_original_requests"] or learner["learner_roots"] != recovered["reused_groups"]:
                raise ValueError("Recovered initial round does not match its original call receipt")
            average_queue_seconds = None
        else:
            expected, finished = (broker["counts"][k] for k in ("policy_requests_submitted", "completed_requests"))
            if not expected == finished == physical:
                raise ValueError(f"Actual physical requests disagree with the closed broker at update {step}")
            average_queue_seconds = broker["counts"]["queue_wait_microseconds"] / 1e6 / physical
        mtime = (directory / "complete.json").stat().st_mtime
        collection, receipt_read, compute = (learner[k] for k in (
            "sampling_seconds", "learner_receipt_read_seconds", "learner_compute_seconds"))
        rows.append(dict(
            update=step, roots=learner["learner_roots"], full_lane_roots=counts["full"],
            segment_lane_roots=counts["segment"], physical_calls=physical,
            recovered_original_calls=physical if step == 1 else 0,
            policy_candidate_groups=learner["policy_groups"], historical_comparison_groups=learner["historical_comparison_slates"],
            nonzero_local_groups=learner["effective_local_slates"], route_or_process_groups=learner["route_credit_slates"],
            structural_stock_closed_rollouts=sum(p["native_stock_solved"] for p in parts),
            reference_verified_closed_rollouts=sum(p["verified_stock_solved"] for p in parts),
            collection_seconds=collection, receipt_read_seconds=receipt_read, learner_compute_seconds=compute,
            sum_of_timed_stages_seconds=collection+receipt_read+compute,
            avg_pre_dispatch_queue_seconds=average_queue_seconds, min_actor_requests=min(actor_calls),
            max_actor_requests=max(actor_calls), publication_interval_seconds=None if last_mtime is None else mtime-last_mtime,
            complete_sha256=complete_sha, scheduler_summary_sha256=broker_sha,
        ))
        last_mtime = mtime
    if rows[-1]["complete_sha256"] != expected_complete:
        raise ValueError("Final optimizer record does not match the selected durable checkpoint")
    steady = rows[1:]
    time_fields = ("collection_seconds", "receipt_read_seconds", "learner_compute_seconds")
    clock = {k: sum(row[k] for row in steady) for k in time_fields}
    totals = {k: sum(row[k] for row in rows) for k in (
        "roots", "full_lane_roots", "segment_lane_roots", "physical_calls", "recovered_original_calls",
        "policy_candidate_groups", "historical_comparison_groups", "nonzero_local_groups", "route_or_process_groups",
        "structural_stock_closed_rollouts", "reference_verified_closed_rollouts",
    )}
    return dict(optimizer_updates=through, contract_sha256=contract_sha, initial_collection_recovery_sha256=recovery_sha,
                training_order_sha256=order, backend=cfg["inference_backend"], roles=role_counts, totals=totals,
                steady_updates=[2, through], steady_wall_fractions={k: v/sum(clock.values()) for k,v in clock.items()},
                steady_median_seconds={k: median(row[k] for row in steady) for k in time_fields},
                steady_median_physical_calls=median(row["physical_calls"] for row in steady),
                steady_median_published_interval_seconds=median(row["publication_interval_seconds"] for row in steady),
                steady_median_pre_dispatch_wait_seconds=median(row["avg_pre_dispatch_queue_seconds"] for row in steady),
                latest_published_at=datetime.fromtimestamp(last_mtime).astimezone().isoformat(timespec="seconds"), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-contract", required=True, type=Path)
    parser.add_argument("--through", required=True, type=int)
    parser.add_argument("--final-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.through < 2 or not output.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Analyze at least two completed updates and use shared storage output")
    report = collect(args.runtime_contract, args.through, args.final_checkpoint_sha256)
    output.mkdir(exist_ok=False)
    with (output / "summary.json").open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    with (output / "per_update.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report["rows"][0])
        writer.writeheader()
        writer.writerows(report["rows"])
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
