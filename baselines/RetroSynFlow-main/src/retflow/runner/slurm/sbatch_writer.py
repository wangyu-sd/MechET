"""Module to integrate with Slurm."""

import textwrap
from pathlib import Path
from typing import List

from retflow import config
from retflow.runner.slurm.slurm_config import SlurmConfig


def make_sbatch_header(
    slurm_config: SlurmConfig, n_jobs: int, ddp: bool = False
) -> str:
    """Generates the header of a sbatch file for Slurm.

    Args:
        slurm_config: Slurm configuration to use
        n_jobs: Number of jobs to run in the batch
    """

    gpu_str = ""
    if slurm_config.gpu is not None:
        if isinstance(slurm_config.gpu, str):
            if config.get_slurm_cluster() == "sockeye":
                gpu_str = f"#SBATCH --gpus-per-node={slurm_config.n_gpu}"
            else:
                gpu_str = (
                    f"#SBATCH --gpus-per-node={slurm_config.gpu}:{slurm_config.n_gpu}"
                )
        elif isinstance(slurm_config.gpu, bool):
            if slurm_config.gpu:
                gpu_str = f"#SBATCH --gpus-per-node={slurm_config.n_gpu}"
            else:
                gpu_str = ""

    extra_sockeye_str = (
        "#SBATCH --constraint=gpu_mem_32"
        if slurm_config.gpu_mem == 32 and config.get_slurm_cluster() == "sockeye"
        else ""
    )
    bangline = "#!/bin/sh\n"

    if ddp:
        formatted_header = textwrap.dedent(
            """
            #SBATCH --account={acc}
            #SBATCH --mem={mem}
            #SBATCH --time={time}
            #SBATCH --cpus-per-task={cpus}
            #SBATCH --mail-user={email}
            #SBATCH --mail-type=ALL
            #SBATCH --nodes={nodes}
            #SBATCH --array=0-{last_job_idx}
            {gpu_str}
            {extra_sockeye_str}

            export OMP_NUM_THREADS=6
            """
        ).format(
            acc=config.get_slurm_account(),
            mem=slurm_config.mem,
            time=slurm_config.time,
            cpus=slurm_config.cpus,
            email=config.get_slurm_email(),
            gpu_str=gpu_str,
            nodes=slurm_config.nodes,
            last_job_idx=n_jobs - 1,
            extra_sockeye_str=extra_sockeye_str,
        )
    else:
        formatted_header = textwrap.dedent(
            """
            #SBATCH --account={acc}
            #SBATCH --mem={mem}
            #SBATCH --time={time}
            #SBATCH --cpus-per-task={cpus}
            #SBATCH --mail-user={email}
            #SBATCH --mail-type=ALL
            #SBATCH --nodes={nodes}
            #SBATCH --array=0-{last_job_idx}
            {gpu_str}
            {extra_sockeye_str}

            """
        ).format(
            acc=config.get_slurm_account(),
            mem=slurm_config.mem,
            time=slurm_config.time,
            cpus=slurm_config.cpus,
            email=config.get_slurm_email(),
            gpu_str=gpu_str,
            nodes=slurm_config.nodes,
            last_job_idx=n_jobs - 1,
            extra_sockeye_str=extra_sockeye_str,
        )

    return bangline + formatted_header


def make_jobarray_content(
    run_exp_by_idx_command: str,
    should_run: List[bool],
):
    """Creates the content of a jobarray sbatch file for Slurm.

    Args:
        run_exp_by_idx_command: Command to call to run the i-th experiment
        should_run: Whether the matching experiment should run
    """

    bash_script_idx_to_exp_script_idx = []
    for i, _should_run in enumerate(should_run):
        if _should_run:
            bash_script_idx_to_exp_script_idx.append(i)

    commands_for_each_experiment = []
    for bash_script_idx, exp_script_idx in enumerate(bash_script_idx_to_exp_script_idx):
        commands_for_each_experiment.append(
            textwrap.dedent(
                f"""
                if [ $SLURM_ARRAY_TASK_ID -eq {bash_script_idx} ]
                then
                    {run_exp_by_idx_command} {exp_script_idx}
                fi
                """
            )
        )

    return "".join(commands_for_each_experiment)


def make_jobarray_file_contents(
    experiment_file: Path,
    should_run: List[bool],
    slurm_config: SlurmConfig,
    on_valid: bool = False,
    compute_round_trip: bool = False,
):
    """Creates a jobarray sbatch file for Slurm."""

    header = make_sbatch_header(slurm_config=slurm_config, n_jobs=sum(should_run))
    command = f"python {experiment_file} "

    if on_valid:
        command += "--valid"
    if compute_round_trip:
        command += " --round_trip"
    command += " --single"
    body = make_jobarray_content(
        run_exp_by_idx_command=command,
        should_run=should_run,
    )

    footer = textwrap.dedent(
        """
        exit
        """
    )

    return header + body + footer


def make_jobarray_file_contents_ddp(
    experiment_file: Path,
    should_run: List[bool],
    slurm_config: SlurmConfig,
):
    header = make_sbatch_header(
        slurm_config=slurm_config, n_jobs=sum(should_run), ddp=True
    )

    body = make_jobarray_content_ddp(
        run_exp_by_idx_command=f"{experiment_file} --ddp_single",
        should_run=should_run,
        gpus_per_exp=slurm_config.n_gpu,
    )

    footer = textwrap.dedent(
        """
        exit
        """
    )

    return header + body + footer


def make_jobarray_content_ddp(
    run_exp_by_idx_command: str, should_run: List[bool], gpus_per_exp: int
):
    bash_script_idx_to_exp_script_idx = []
    for i, _should_run in enumerate(should_run):
        if _should_run:
            bash_script_idx_to_exp_script_idx.append(i)

    commands_for_each_experiment = []
    for bash_script_idx, exp_script_idx in enumerate(bash_script_idx_to_exp_script_idx):
        gpu_ids_str = "--ddp_gpu_ids $CUDA_VISIBLE_DEVICES"

        commands_for_each_experiment.append(
            textwrap.dedent(
                f"""
                if [ $SLURM_ARRAY_TASK_ID -eq {bash_script_idx} ]
                then
                    torchrun --nnodes=1 --nproc_per_node={gpus_per_exp} --master_port=29400 {run_exp_by_idx_command} {exp_script_idx} {gpu_ids_str}
                fi
                """
            )
        )

    return "".join(commands_for_each_experiment)
