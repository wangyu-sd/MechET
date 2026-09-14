import logging
import os
import socket
from logging import Logger
from pathlib import Path
from typing import Optional

import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import torch

ENV_VAR_WORKSPACE = "RETRO_WORKSPACE"
ENV_VAR_LOGGING = "RETRO_CONSOLE_LOGGING_LEVEL"
LOG_FMT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

WANDB_STATUS = "RETRO_WANDB_ENABLE"
WANDB_API_KEY = "WANDB_API_KEY"
RETRO_WANDB_MODE = "RETRO_WANDB_MODE"
RETRO_WANDB_PROJECT = "RETRO_WANDB_PROJECT"
RETRO_WANDB_ENTITY = "RETRO_WANDB_ENTITY"
RETRO_SLURM_ACCOUNT = "RETRO_SLURM_ACCOUNT"
RETRO_SLURM_EMAIL = "RETRO_SLURM_NOTIF_EMAIL"
RETRO_SLURM_CLUSTER = "RETRO_SLURM_CLUSTER"

# disable rdkit logging
rkl.logger().setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")


def get_workspace_directory() -> Path:
    workspace = os.environ.get(ENV_VAR_WORKSPACE, None)
    if workspace is None:
        raise ValueError(
            "Workspace not set. "
            f"Define the {ENV_VAR_WORKSPACE} environment variable"
            "To define where to save datasets and experiment results."
        )
    return Path(workspace)


def get_dataset_directory() -> Path:
    return get_workspace_directory() / "datasets"


def get_experiment_directory() -> Path:
    return get_workspace_directory() / "experiments"


def get_models_directory() -> Path:
    return get_workspace_directory() / "models"  # path to pretrained models


def get_plots_directory() -> Path:
    return get_workspace_directory() / "plots"


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        get_logger("GPU not available, running experiments on CPU.", logging.WARNING)
    return device


def get_wandb_status() -> bool:
    status = os.environ.get(WANDB_STATUS, None)
    if status is None:
        raise ValueError("wandb mode is not set")
    return status.lower() == "true"


def get_wandb_mode() -> str:
    mode = os.environ.get(RETRO_WANDB_MODE, None)
    if mode is None:
        raise ValueError("wandb mode is not set")
    return mode


def get_wandb_project() -> str:
    proj = os.environ.get(RETRO_WANDB_PROJECT, None)
    if proj is None:
        raise ValueError("wandb project is not set")
    return proj


def get_wandb_entity() -> str:
    ent = os.environ.get(RETRO_WANDB_ENTITY, None)
    if ent is None:
        raise ValueError("wandb entity is not set")
    return ent


def get_slurm_account() -> str:
    acc = os.environ.get(RETRO_SLURM_ACCOUNT, None)
    if acc is None:
        raise ValueError("slurm account not set")
    return acc


def get_slurm_email() -> str:
    email = os.environ.get(RETRO_SLURM_EMAIL, None)
    if email is None:
        raise ValueError("slurm email not set")
    return email


def get_slurm_cluster() -> str:
    types = ["cc", "sockeye"]
    cluster_type = os.environ.get(RETRO_SLURM_CLUSTER, None)
    if cluster_type.lower() not in types:
        raise ValueError(
            "If you are using Slurm you must set a cluster type. Either cc or sockeye"
        )
    return cluster_type


def get_console_logging_level() -> str:
    return os.environ.get(ENV_VAR_LOGGING, "DEBUG")


def get_logger(name: Optional[str] = None, level: Optional[str | int] = None) -> Logger:
    """Get a logger with a console handler.

    Args:
        name: Name of the logger.
        level: Logging level.
            Defaults to the value of the env variable OPTEXP_CONSOLE_LOGGING_LEVEL.
    """
    logger = logging.getLogger(__name__ if name is None else name)

    if not any(isinstance(x, logging.StreamHandler) for x in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level=get_console_logging_level() if level is None else level)
        formatter = logging.Formatter(LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.setLevel(level=get_console_logging_level() if level is None else level)
    return logger


def set_logfile(path: Path, name: Optional[str] = None):
    handler = logging.FileHandler(path)
    handler.formatter = logging.Formatter(LOG_FMT)
    get_logger(name=name).addHandler(handler)
    return handler


def remove_loghandler(handler: logging.FileHandler, name: Optional[str] = None):
    get_logger(name=name).removeHandler(handler)


def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Tries to connect to a public DNS server (Google's 8.8.8.8) to check for internet access.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        return False
