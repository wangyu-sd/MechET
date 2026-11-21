"""Export the November 22 PR117 result ledger from independently verified evidence."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def index(run):
    records = {}
    used = {}

    def evidence(relative, expected=None):
        raw = (run / relative).read_bytes()
        digest = sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"Evidence digest changed for {relative}")
        used[relative] = digest
        return json.loads(raw)

    def record(name, main_result, status, finding, **sections):
        records[name] = dict(main_result=main_result, status=status, finding=finding, **sections,
                             source_files=dict(used))
        used.clear()

    development = evidence("stage_a_final_verified/complete.json")
    paired = evidence("stage_a_final_verified/paired_results.json", development["analysis_sha256"]["paired_results.json"])
    if development["fully_paired_targets"] != 24 or len(development["solved"]) != 4:
        raise ValueError("Development replay lacks the declared paired population")
    historical = evidence("results/R1.json")
    public = evidence("reference_1084_full_target_lock_20261121.json")
    public_batches = evidence("reference_public_batches_1084_20261122/evidence.json")
    if public["total_targets"] != len(public["targets"]) or public["total_targets"] != 1084 or public["model_calls"] != 0:
        raise ValueError("Expected the locked, unevaluated public results population")
    record("R1", 1, "PARTIAL", "Historical and development closure measured. Official public outcomes pending.",
           historical=historical,
           development=dict(targets=development["fully_paired_targets"], results=paired["arm_metrics"],
                            actor_sha256=paired["actor_sha256"], panel_sha256=paired["panel_sha256"],
                            all_four_arm_physical_calls=development["native_requests"],
                            reserved_a100_hours=development["GPU_hours"]),
           reference_full=dict(status="NOT_RUN", targets=public["total_targets"], independent=public["independent_count"],
                               cohorts=public["cohorts"], stock_fingerprints=public["stocks"],
                               batching_status=public_batches["state"], scope="All 1084 official targets retained"),
           secondary_paroutes=dict(status="NOT_RUN", targets=10000, condition="Measured residual GPU capacity"))

    archived = evidence("results/R2.json")
    census_path = "train_history_credit_census_through28_20261122/evidence.json"
    replay = evidence("train_history_credit_census_through28_20261122/complete_replay_evidence.json")
    census = evidence(census_path, replay["artifacts"][census_path])
    learned = evidence("reference_trained_actor_launch_1a100_20261122_update29/prepared.json")
    learned_proof = evidence("reference_trained_actor_launch_1a100_20261122_update29/cpu_admission.json",
                             learned["cpu_admission_sha256"])
    if census["history_comparisons"] != replay["all_history_comparisons"] or learned["native_calls"] != 0:
        raise ValueError("Learner comparison census or preflight scope changed")
    record("R2", 3, "PARTIAL", "Observed route-outcome credit reached actual learner inputs. Held-out policy effect is pending.",
           older_archive=archived,
           current_archive={key: census[key] for key in ("updates", "targets", "candidate_groups", "multi_candidate_groups",
               "history_comparisons", "reference_verified_groups", "cutoff_sha256")},
           replay={key: replay[key] for key in ("all_history_comparisons", "distinct_mapped_successors", "same_mapped_successor",
               "native_receipts_verified", "flow_teacher_preference_by_replay_status")},
           trained_actor_paired=dict(status="NOT_RUN", admission=learned_proof,
                                     checkpoint_scope="Update 29 used to validate launch. Select durable checkpoint before launch."))

    evidence("stage_a_final_verified/complete.json")
    evidence("stage_a_final_verified/paired_results.json", development["analysis_sha256"]["paired_results.json"])
    frontier = evidence("stage_a_final_verified/early_electron_frontier.json",
                        development["analysis_sha256"]["early_electron_frontier.json"])
    freeze = evidence("stage_a_final_method_freeze_20261122.json")
    confirm = evidence("reference_confirmation_launch_1a100_20261122/prepared.json")
    registration = evidence("reference_confirmation_frozen_20261122/confirmation_registration.json",
                            confirm["confirmation_registration_sha256"])
    record("R3", 2, "PARTIAL", "Matched development comparison completed. All four arms had zero stock-closed targets.",
           development=dict(targets=development["fully_paired_targets"], actor_sha256=paired["actor_sha256"],
                            paired_effects=paired["paired_effects"], arm_metrics=paired["arm_metrics"],
                            distinct_executable_early_alternative_targets=frontier["multiple_successor_targets"],
                            duplicate_frontier_targets=frontier["root_frontier_duplicate_targets"],
                            backjumps=frontier["observed_backjumps"], selected_primary=freeze["primary"]),
           independent_confirmation=dict(status="NOT_RUN", targets=registration["targets"],
                                         arms=registration["arms"], cap=registration["per_target_arm_request_cap"],
                                         arms_total_call_ceiling=registration["total_request_ceiling"],
                                         stock_and_targets=registration["target_lock"], actor_sha256=registration["actor_sha256"]))

    original = evidence("launch_1a100_recovery_20261122_01/independent_replay.json")
    feedback = evidence("launch_1a100_feedback_20261122_01/independent_replay.json")
    input_pair = evidence("downstream_executor_feedback_paired_input_analysis_20261122.json")
    throughput = evidence("training_throughput_through_update27_20261122/paper/evidence.json")
    if original["target_lock_sha256"] != feedback["target_lock_sha256"] or original["actor_sha256"] != feedback["actor_sha256"]:
        raise ValueError("The matched recovery policies used different evidence or actors")
    record("R4", 5, "PARTIAL", "Matched rejection-feedback and training cost evidence. Distillation has no completed condition.",
           distillation=dict(status="NOT_RUN", admission="One genuine study subject to PR117 phase and resource availability"),
           fallback_same_state_retry=dict(actor_sha256=original["actor_sha256"], original_cases=original["cases"],
               feedback_cases=feedback["cases"], original_recovered=original["recovered"], feedback_recovered=feedback["recovered"],
               original_calls=original["actual_native_calls"], feedback_calls=feedback["actual_native_calls"],
               pairs=input_pair["comparisons"]), training_throughput=throughput)

    prior_chemistry = evidence("results/R5.json")
    example = evidence("case-610053-electron-backtracking/manifest.json")
    cbyd = evidence("cbyd_native_evaluation_prepared_20261122/prepared.json")
    literature = evidence("cbyd_native_scorer_literature_calibration_20261122.json")
    if cbyd["views"]["cumulative"]["targets"] != literature["sources_scored"] or literature["native_model_calls"] != 0:
        raise ValueError("CbyD source-calibration denominator or outcome scope changed")
    record("R5", 4, "PARTIAL", "Executor-replayed five-reaction TRAIN witness and literature scorer calibration. Independent CbyD outcomes pending.",
           train_archive=prior_chemistry, witnessed_route=dict(case=example["case"], totals=example["totals"],
                                                               scope=example["scope"], source_sha256=example["source_sha256"]),
           cbyd=dict(status="NOT_RUN", actor=cbyd["model"], stock=cbyd["protocol_stock"], panels=cbyd["views"],
                     target_lock_sha256=cbyd["expanded_target_lock_sha256"],
                     source_only_calibration=dict(targets=literature["sources_scored"], exact_matches=literature["exact_literature_source_matches"],
                                                  scope=literature["scope"])))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if any(not path.is_relative_to("/aaa/fionafyang/buddy1") for path in (run, output)):
        parser.error("Use Ceph for both the evidence and generated index")
    records = index(run)
    output.mkdir()
    for name, data in records.items():
        with (output / f"{name}.json").open("x") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
