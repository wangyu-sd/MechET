"""Runner API with lazy imports to avoid experiment/problem import cycles."""

from importlib import import_module


_EXPORTS = {
    "cli_runner": ("retflow.runner.cli", "cli_runner"),
    "DistributedHelper": (
        "retflow.utils.distributed_helper",
        "DistributedHelper",
    ),
}

__all__ = ["cli_runner", "slurm_config", "DistributedHelper"]


def __getattr__(name):
    if name == "slurm_config":
        value = import_module("retflow.runner.slurm.slurm_config")
    elif name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(module_name), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
