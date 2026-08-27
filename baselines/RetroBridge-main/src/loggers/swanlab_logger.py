"""SwanLab logger adapter for the project's ``pytorch_lightning`` runtime.

SwanLab's current built-in Lightning integration targets the newer
``lightning`` package. RetroBridge uses ``pytorch_lightning`` directly, so this
small adapter implements that package's logger interface while delegating run
creation and metric storage to SwanLab's public API.
"""

from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Optional

import swanlab
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def _config_value(value):
    """Convert arbitrary Lightning hyperparameters to compact SwanLab values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_config_value(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


class SwanLabLogger(Logger):
    """Log PyTorch Lightning hyperparameters and metrics to SwanLab."""

    def __init__(
        self,
        *,
        project: str,
        experiment_name: str,
        save_dir: str,
        log_dir: str,
        workspace: Optional[str] = None,
        mode: str = "online",
        run_id: Optional[str] = None,
        resume: str = "allow",
        **init_kwargs: Any,
    ):
        super().__init__()
        self._project = project
        self._experiment_name = experiment_name
        self._save_dir = save_dir
        self._run_id = run_id
        self._experiment = None
        self._init_kwargs = {
            "project": project,
            "name": experiment_name,
            "log_dir": log_dir,
            "mode": mode,
            "id": run_id,
            "resume": resume,
            **init_kwargs,
        }
        if workspace is not None:
            self._init_kwargs["workspace"] = workspace

    @property
    def name(self):
        return self._project

    @property
    def version(self):
        if self._experiment is not None:
            return getattr(self._experiment, "id", self._run_id)
        return self._run_id

    @property
    def save_dir(self):
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            try:
                self._experiment = swanlab.get_run()
            except RuntimeError:
                self._experiment = None
            if self._experiment is None:
                self._experiment = swanlab.init(**self._init_kwargs)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        if isinstance(params, Namespace):
            params = vars(params)
        elif isinstance(params, Mapping):
            params = dict(params)
        elif hasattr(params, "__dict__"):
            params = vars(params)
        else:
            params = {}
        self.experiment.config.update(_config_value(params))

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self.experiment.log(dict(metrics), step=step)

    @rank_zero_only
    def save(self):
        return None

    @rank_zero_only
    def finalize(self, status):
        if self._experiment is None:
            return
        state = "success" if status == "success" else "crashed"
        error = None if state == "success" else f"PyTorch Lightning status: {status}"
        self._experiment.finish(state=state, error=error)
        self._experiment = None

