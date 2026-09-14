from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from retflow.experiment import Experiment
from retflow.runner.slurm import (SlurmConfig, make_jobarray_file_contents,
                                  make_jobarray_file_contents_ddp)
from retflow.runner.wandb_integration import (download_run_data,
                                              get_successful_ids_and_runs,
                                              get_wandb_runs_for_group)
from retflow.utils.distributed_helper import DistributedHelper


def cli_runner(
    exps: List[Experiment] | List[ExperimentEvaluator], slurm_config: SlurmConfig = None
):
    if not isinstance(exps[0], Experiment):
        from retflow.experiment_eval import ExperimentEvaluator

        if not isinstance(exps[0], ExperimentEvaluator):
            raise TypeError("cli_runner expects Experiment or ExperimentEvaluator")
        _eval_cli_runner(exps, slurm_config)
        return

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--local",
        action="store_true",
        help="run experiments on local workstation",
        default=False,
    )

    group.add_argument(
        "--slurm",
        action="store_true",
        help="run experiments on slurm cluster",
        default=False,
    )

    group.add_argument(
        "--single",
        action="store",
        type=int,
        help="Run a single experiment on a slurm cluster.",
        default=None,
    )

    group.add_argument(
        "--ddp_single",
        action="store",
        type=int,
        help="Run a single experiment with ddp on a slurm cluster.",
        default=None,
    )

    group.add_argument(
        "--download",
        action="store_true",
        help="download run data from wandb",
        default=False,
    )

    parser.add_argument(
        "--force", action="store_true", help="force experiments to run", default=False
    )

    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Run data distribution training",
        default=False,
    )

    parser.add_argument(
        "--ddp_gpu_ids",
        type=str,
        default=None,
        help="List of GPU ids for ddp training",
    )

    args = parser.parse_args()

    if args.local:
        run_experiments_locally(exps)
        return

    if args.download:
        download_exp_data(exps)
        return

    if args.slurm:
        if args.ddp:
            run_slurm(
                experiments=exps,
                slurm_config=slurm_config,
                force_rerun=args.force,
                ddp=True,
            )
        else:
            run_slurm(
                experiments=exps, slurm_config=slurm_config, force_rerun=args.force
            )
        return

    if args.single is not None:
        idx = int(args.single)
        if idx < 0 or idx >= len(exps):
            raise ValueError(
                f"Given index {idx} out of bounds for {len(exps)} experiments"
            )
        exps[args.single].run()
        return

    if args.ddp_single is not None:
        idx = int(args.ddp_single)
        if not args.ddp_gpu_ids:
            raise ValueError("GPU IDS must be provided")
        if not slurm_config.n_gpu > 1:
            raise ValueError(
                "For DDP training the number of GPUs must be greater "
                "than one. The SlurmConfig has only 1 GPU."
            )
        assert len(exps) == 1
        dist_helper = DistributedHelper(
            ddp_gpu_ids=args.ddp_gpu_ids,
            init_method="env://",
        )

        if idx < 0 or idx >= len(exps):
            raise ValueError(
                f"Given index {idx} out of bounds for {len(exps)} experiments"
            )
        exps[args.ddp_single].run(dist_helper=dist_helper)
        return


def _eval_cli_runner(
    exps_eval: List[ExperimentEvaluator], slurm_config: SlurmConfig = None
):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--local",
        action="store_true",
        help="run experiments on local workstation",
        default=False,
    )

    group.add_argument(
        "--slurm",
        action="store_true",
        help="run experiments on slurm cluster",
        default=False,
    )

    group.add_argument(
        "--single",
        action="store",
        type=int,
        help="Run a single experiment on a slurm cluster.",
        default=None,
    )

    group.add_argument(
        "--compute",
        action="store_true",
        help="Process generated samples and compute metrics.",
        default=False,
    )

    parser.add_argument(
        "--round_trip",
        action="store_true",
        help="compute round trip metrics",
        default=False,
    )

    parser.add_argument(
        "--valid", action="store_true", help="evaluate on validation set", default=False
    )

    args = parser.parse_args()

    if args.local:
        for exp_eval in exps_eval:
            exp_eval.run(on_valid=args.valid, compute_round_trip=args.round_trip)
        return

    if args.slurm:
        run_eval_slurm(
            exps_eval,
            slurm_config,
            args.valid,
            args.round_trip,
        )
        return

    if args.single is not None:
        idx = int(args.single)
        if idx < 0 or idx >= len(exps_eval):
            raise ValueError(
                f"Given index {idx} out of bounds for {len(exps_eval)} experiments"
            )
        exps_eval[args.single].run(args.valid, args.round_trip)
        return

    if args.compute:
        from retflow.utils.eval_helper import process_data_compute_metrics

        for exp_eval in exps_eval:
            process_data_compute_metrics(
                exp_eval.save_file_path(args.valid),
                exp_eval.examples_per_sample,
                args.round_trip,
            )
        return
    
    if args.compute_slurm:
        return 


def run_experiments_locally(exps: List[Experiment]):
    # each set of experiments should correspond to one dataset
    exps[0].problem.dataset.download()
    for exp in exps:
        exp.run()


def run_eval_slurm(
    experiment_evals: List[ExperimentEvaluator],
    slurm_config: SlurmConfig,
    on_valid: bool = False,
    compute_round_trip: bool = False,
):
    print("Preparing to evaluate experiments to run on Slurm")
    path_to_python_script = Path(sys.argv[0]).resolve()

    contents = make_jobarray_file_contents(
        experiment_file=path_to_python_script,
        should_run=[True for exp_eval in experiment_evals],
        slurm_config=slurm_config,
        on_valid=on_valid,
        compute_round_trip=compute_round_trip
    )

    group = experiment_evals[0].experiment.group
    tmp_filename = f"tmp_eval_{group}.sh"
    print(f"  Saving sbatch file in {tmp_filename}")
    with open(tmp_filename, "w+") as file:
        file.writelines(contents)

    experiment_evals[0].experiment.problem.dataset.download()

    print(f"  Sending experiments to Slurm - executing sbatch file")
    os.system(f"sbatch {tmp_filename}")


def run_slurm(
    experiments: List[Experiment],
    slurm_config: SlurmConfig,
    force_rerun: bool,
    ddp: bool = False,
) -> None:
    """Run experiments on Slurm."""
    print("Preparing experiments to run on Slurm")
    path_to_python_script = Path(sys.argv[0]).resolve()

    if not force_rerun:
        print("  Checking which experiments have to run")
        exps_to_run = remove_experiments_that_are_already_saved(experiments)
        should_run = [exp in exps_to_run for exp in experiments]
        print(f"    Should run {should_run.count(True)}/{len(should_run)} experiments")
    else:
        should_run = [True for _ in experiments]

    if ddp:
        contents = make_jobarray_file_contents_ddp(
            experiment_file=path_to_python_script,
            should_run=should_run,
            slurm_config=slurm_config,
        )
    else:
        contents = make_jobarray_file_contents(
            experiment_file=path_to_python_script,
            should_run=should_run,
            slurm_config=slurm_config,
        )

    group = experiments[0].group
    tmp_filename = f"tmp_{group}.sh"
    print(f"  Saving sbatch file in {tmp_filename}")
    with open(tmp_filename, "w+") as file:
        file.writelines(contents)

    experiments[0].problem.dataset.download()

    print(f"  Sending experiments to Slurm - executing sbatch file")
    os.system(f"sbatch {tmp_filename}")


def remove_experiments_that_are_already_saved(
    experiments: List[Experiment],
) -> List[Experiment]:
    """Checks a list of experiments against the experiments stored on wandb.
    Returns only the experiments that are not saved and marked as successful.

    Args:
        experiments: List of experiments to check

    Returns: List of experiments that still need to run
    """

    if len(experiments) == 0:
        return []

    group = experiments[0].group
    if not all(exp.group == group for exp in experiments):
        raise ValueError("All experiments must have the same group.")

    successful_runs = get_wandb_runs_for_group(group)
    successful_exp_ids = [run.config["exp_id"] for run in successful_runs]

    experiments_to_run = [
        exp for exp in experiments if exp.exp_id() not in successful_exp_ids
    ]

    return experiments_to_run


def download_exp_data(exps: List[Experiment]):
    # assume all experiments are part of the same group.
    _, runs = get_successful_ids_and_runs(exps[0].group)
    for run in runs:
        download_run_data(run)
