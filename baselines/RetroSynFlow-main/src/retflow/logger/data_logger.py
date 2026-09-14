from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import wandb

from retflow import config
from retflow.config import get_logger
from retflow.logger.rate_limited_logger import RateLimitedLogger
from retflow.logger.utils import pprint_dict


class DataLogger:
    def __init__(
        self,
        config_dict: Dict,
        exp_id: str,
        group: str,
        save_dir: Path,
        name: str | None = None,
    ) -> None:
        """Data logger for experiments.

        Delegates to a console logger to print progress.
        Saves the data to a csv and experiment configuration to a json file.
        Creates the save_dir if it does not exist.
        """
        self.config_dict = config_dict
        self.save_directory = save_dir
        self.group = group
        self.name = name
        print(self.config_dict)

        if not os.path.exists(config.get_experiment_directory()):
            os.makedirs(config.get_experiment_directory())

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        try:
            if config.get_wandb_status():
                get_logger().info("WandB is enabled.")
                self.run = wandb.init(
                    project=config.get_wandb_project(),
                    config={
                        "config_dict": self.config_dict,
                        "exp_id": exp_id,
                    },
                    mode=config.get_wandb_mode(),
                    entity=config.get_wandb_entity(),
                    dir=config.get_experiment_directory().parent,
                    group=self.group,
                    name=self.name
                )
            else:
                get_logger().warning(
                    "WandB is not enabled. Only logging to command line."
                )
        except ValueError as e:
            get_logger().warning(str(e))

        self._current_dict: Dict = {}
        self.console_logger = RateLimitedLogger()

    def log_data(self, metric_dict: dict) -> None:
        """Log a dictionary of metrics.

        Based on the wandb log function (https://docs.wandb.ai/ref/python/log)
        Uses the concept of "commit" to separate different steps/iterations.

        log_data can be called multiple times per step,
        and repeated calls update the current logging dictionary.
        If metric_dict has the same keys as a previous call to log_data,
        the keys will get overwritten.

        To move on to the next step/iteration, call commit.

        Args:
            metric_dict: Dictionary of metrics to log
        """
        self._current_dict.update(metric_dict)
        if config.get_wandb_status():
            self.run.log(metric_dict, commit=False)

    def commit(self, on_batch=False, force_console=False) -> None:
        """Commit the current logs and move on to the next step/iteration."""
        if not on_batch:
            self.console_logger.log(
                pprint_dict(self._current_dict), force=force_console
            )
        self._current_dict = {}
        if config.get_wandb_status():
            self.run.log({}, commit=True)

    def save(self, exit_code):
        config_dict_file = self.save_directory / "config_dict.json"

        json_data = json.dumps(self.config_dict, indent=4)
        with open(config_dict_file, "w") as f:
            f.write(json_data)

        if config.get_wandb_status():
            if self.run is None:
                raise ValueError("Expected a WandB run but None found.")

            get_logger().info("Finishing Wandb run")
            wandb.finish(exit_code=exit_code)

            if config.get_wandb_mode() == "offline":
                if config.check_internet_connection():
                    get_logger().info(f"Uploading wandb run in {Path(self.run.dir).parent}")
                    command = (
                        f"wandb sync "
                        f"--id {self.run.id} "
                        f"-p {config.get_wandb_project()} "
                        f"-e {config.get_wandb_entity()} "
                        f"{Path(self.run.dir).parent}"
                    )
                    get_logger().info(f"    {command}")

                    subprocess.run(
                        command,
                        shell=True,
                    )
                else:
                    get_logger().info(
                        f"Your local machine or compute node"
                        "is not connected to the internet. Check your internet connection"
                        "or manually sync from a login node."
                    )

            else:
                command = (
                    f"wandb sync "
                    f"--id {self.run.id} "
                    f"-p {config.get_wandb_project()} "
                    f"-e {config.get_wandb_entity()} "
                    f"{Path(self.run.dir).parent}"
                )
                get_logger().info(f"Wandb mode is online. Not Running:    {command}")
