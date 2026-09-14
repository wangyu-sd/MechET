import hashlib
import random
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

from retflow import config
from retflow.logger.asdict_with_classes import asdict_with_class
from retflow.logger.data_logger import DataLogger
from retflow.optimizers.optimizer import Optimizer
from retflow.problems.problem import Problem
from retflow.utils.distributed_helper import DistributedHelper


@dataclass
class Experiment:
    problem: Problem
    optim: Optimizer
    epochs: int
    sample_epoch: int
    num_samples: int
    examples_per_sample: int
    seed: int
    group: str
    name: str | None = None

    def run(self, dist_helper: DistributedHelper | None = None):
        config.get_logger().info("Initializing experiment.")
        if dist_helper:
            _DistributedExperiment(**vars(self)).run(dist_helper)
            return

        self.set_seed()
        data_logger = DataLogger(
            config_dict=asdict_with_class(self),  # type: ignore
            exp_id=self.exp_id(),
            save_dir=self.save_directory(),
            group=self.group,
            name=self.name,
        )

        self.problem.setup_problem()
        optimizer = self.problem.get_optimizer(self.optim)  # type: ignore
        lr_sched = self.optim.lr_sched.get_scheduler(optimizer)

        try:
            for epoch in range(1, self.epochs + 1):
                train_metrics = self.problem.one_epoch(
                    optimizer,
                    lr_sched,
                    dist_helper=dist_helper,
                )
                val_metrics = self.problem.validation(dist_helper)

                data_logger.log_data({"epoch": epoch})
                data_logger.log_data(train_metrics["metrics"])
                data_logger.log_data(val_metrics["metrics"])

                if epoch % self.sample_epoch == 0.0:
                    sampling_metrics = self.problem.sample_generation(
                        self.num_samples, self.examples_per_sample
                    )
                    data_logger.log_data(sampling_metrics)  # type: ignore
                    data_logger.commit(force_console=True)
                    self.problem.save_model(
                        checkpoint_dir=self.save_directory(), epoch=epoch
                    )
                else:
                    data_logger.commit()
        except KeyboardInterrupt:
            config.get_logger().error("Keyboard Interrupt. Exiting Experiment.")
            data_logger.save(exit_code=1)
            self.problem.save_model(checkpoint_dir=self.save_directory(), epoch=epoch)  # type: ignore
            return
        except RuntimeError as e:
            config.get_logger().error(
                f"Encountered runtime error: {str(e)}.  Exiting Experiment."
            )
            data_logger.save(exit_code=1)
            tb = traceback.format_exc()
            config.get_logger().error(tb)
            self.problem.save_model(checkpoint_dir=self.save_directory(), epoch=epoch)  # type: ignore
            return
        config.get_logger().info("Experiment done.")
        data_logger.save(exit_code=0)
        self.problem.save_model(checkpoint_dir=self.save_directory(), ddp=False)  # type: ignore

    def save_directory(self) -> Path:
        base_dir = config.get_experiment_directory()
        if self.name:
            return base_dir / self.group / self.name
        return base_dir / self.group / self.exp_id()

    def load_data(self):
        raise NotImplementedError()

    def exp_id(self):
        return hashlib.sha1(str.encode(str(self))).hexdigest()

    def set_seed(self):
        self._set_seed(self.seed)

    def _set_seed(self, seed: int | None = None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


@dataclass
class _DistributedExperiment(Experiment):
    def run(self, dist_helper: DistributedHelper):
        self.set_seed()
        rank = dist_helper.get_rank()

        dist_helper.ddp_sync()

        if rank == 0:
            data_logger = DataLogger(
                config_dict=asdict_with_class(self),  # type: ignore
                exp_id=self.exp_id(),
                save_dir=self.save_directory(),
                group=self.group,
                name=self.name,
            )

        dist_helper.ddp_sync()

        self.problem.setup_problem(dist_helper)
        optimizer = self.problem.get_optimizer(self.optim)
        lr_sched = self.optim.lr_sched.get_scheduler(optimizer)

        try:
            for epoch in range(1, self.epochs + 1):
                self.problem.train_loader.sampler.set_epoch(epoch)

                train_metrics = self.problem.one_epoch(
                    optimizer,
                    lr_sched,
                    dist_helper=dist_helper,
                )

                val_metrics = self.problem.validation(dist_helper)
                dist_helper.ddp_sync()
                num_train_points = train_metrics["num_points"]
                dist.all_reduce(num_train_points)
                for metric in train_metrics["metrics"]:
                    metric_val = train_metrics["metrics"][metric]
                    dist.all_reduce(metric_val)
                    train_metrics["metrics"][metric] = (
                        metric_val.item() / num_train_points.item()
                    )
                dist_helper.ddp_sync()

                num_val_points = val_metrics["num_points"]
                for metric in val_metrics["metrics"]:
                    metric_val = val_metrics["metrics"][metric]
                    dist.all_reduce(metric_val)
                    val_metrics["metrics"][metric] = (
                        metric_val.item() / num_val_points.item()
                    )

                dist_helper.ddp_sync()

                if rank == 0:
                    data_logger.log_data({"epoch": epoch})
                    data_logger.log_data(train_metrics["metrics"])
                    data_logger.log_data(val_metrics["metrics"])

                dist_helper.ddp_sync()

                if epoch % self.sample_epoch == 0.0:
                    valid_metrics = self.problem.sample_generation(
                        self.num_samples, self.examples_per_sample, dist_helper
                    )
                    dist_helper.ddp_sync()
                    if rank == 0:
                        data_logger.log_data(valid_metrics)  # type: ignore
                        data_logger.commit(force_console=True)
                        self.problem.save_model(
                            checkpoint_dir=self.save_directory(), epoch=epoch, ddp=True
                        )
                else:
                    dist_helper.ddp_sync()
                    if rank == 0:
                        data_logger.commit()
                dist_helper.ddp_sync()
        except KeyboardInterrupt:
            config.get_logger().error("Keyboard Interrupt. Exiting Experiment.")
            dist_helper.ddp_sync()
            if rank == 0:
                data_logger.save(exit_code=1)
                self.problem.save_model(
                    checkpoint_dir=self.save_directory(), epoch=epoch, ddp=True
                )
            dist_helper.clean_up()
            return
        except RuntimeError as e:
            config.get_logger().error(
                f"Encountered runtime error: {str(e)}.  Exiting Experiment."
            )

            if rank == 0:
                data_logger.save(exit_code=1)
                tb = traceback.format_exc()
                config.get_logger().error(tb)
                self.problem.save_model(
                    checkpoint_dir=self.save_directory(), epoch=epoch, ddp=True
                )
            dist_helper.clean_up()
            return

        config.get_logger().info("Experiment done.")
        if rank == 0:
            data_logger.save(exit_code=0)
            self.problem.save_model(checkpoint_dir=self.save_directory(), ddp=True)
        dist_helper.clean_up()
