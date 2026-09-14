from typing import List

import pandas as pd
import wandb
import wandb.apis
from wandb.apis.public import Run

from retflow import config
from retflow.config import get_logger


class WandbAPI:
    """Static class to provide a singleton handler to the wandb api.

    When in need to call the Wandb API, use WandbAPI.get_handler()
    instead of creating a new instance of wandb.Api().
    """

    api_handler = None

    @staticmethod
    def get_handler():
        if WandbAPI.api_handler is None:
            WandbAPI.api_handler = wandb.Api(timeout=60)
        return WandbAPI.api_handler

    @staticmethod
    def get_path():
        return f"{config.get_wandb_entity()}/{config.get_wandb_project()}"


def get_wandb_runs_for_group(group: str) -> List[Run]:
    """Get the runs of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = WandbAPI.get_handler().runs(
        WandbAPI.get_path(), filters={"group": group}, per_page=1000
    )

    if any("exp_id" not in run.config for run in runs):
        get_logger().warning("Some runs do not have an exp_id attribute.")

    return [run for run in runs if run.state == "finished"]


def get_successful_ids_and_runs(group: str):
    """Get the experiment ids of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = get_wandb_runs_for_group(group)
    successful_runs = []
    successful_exp_ids = []
    for run in runs:
        if run.config["exp_id"] not in successful_exp_ids and run.state == "finished":
            successful_runs.append(run)
            successful_exp_ids.append(run.config["exp_id"])

    return successful_exp_ids, successful_runs


def download_run_data(run: Run, parquet=True):
    """Given a Wandb Run, download the full history."""
    save_dir = (
        config.get_experiment_directory()
        / run.config["exp_config"]["group"]
        / run.config["exp_id"]
    )
    if parquet:
        save_file = save_dir / "exp_data.parquet"
    else:
        save_file = save_dir / "exp_data.csv"

    if save_file.exists():
        return

    df: pd.DataFrame = run.history(pandas=True, samples=10000)  # type: ignore
    if parquet:
        df.to_parquet(save_file)
    else:
        df.to_csv(save_file)
