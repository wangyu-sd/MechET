"""Public RetroSynFlow API with lazy imports.

Dataset preprocessing does not require the Molecular Transformer forward model
or experiment stack.  Lazy imports keep those optional runtime dependencies
out of preprocessing-only jobs while preserving the published public API.
"""

from importlib import import_module


_EXPORTS = {
    "RetroDataset": ("retflow.datasets", "RetroDataset"),
    "SynthonDataset": ("retflow.datasets", "SynthonDataset"),
    "TorchDrugRetroDataset": ("retflow.datasets", "TorchDrugRetroDataset"),
    "Experiment": ("retflow.experiment", "Experiment"),
    "ExperimentEvaluator": ("retflow.experiment_eval", "ExperimentEvaluator"),
    "GraphDiscreteFM": ("retflow.methods", "GraphDiscreteFM"),
    "GraphMarkovBridge": ("retflow.methods", "GraphMarkovBridge"),
    "GraphTransformer": ("retflow.models", "GraphTransformer"),
    "ADAMW": ("retflow.optimizers", "ADAMW"),
    "Retrosynthesis": ("retflow.problems", "Retrosynthesis"),
    "SynthonCompletion": ("retflow.problems", "SynthonCompletion"),
    "SynthonRetrosynthesis": ("retflow.problems", "SynthonRetrosynthesis"),
    "DistributedHelper": (
        "retflow.utils.distributed_helper",
        "DistributedHelper",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
