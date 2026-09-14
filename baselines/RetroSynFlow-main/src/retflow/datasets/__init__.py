"""Dataset API with lazy imports for preprocessing-only environments."""

from importlib import import_module


_EXPORTS = {
    "Dataset": ("retflow.datasets.dataset", "Dataset"),
    "RetroDataset": ("retflow.datasets.retro", "RetroDataset"),
    "SmallRetroDataset": ("retflow.datasets.retro", "SmallRetroDataset"),
    "ToyRetroDataset": ("retflow.datasets.retro", "ToyRetroDataset"),
    "TorchDrugRetroDataset": (
        "retflow.datasets.retro_drug",
        "TorchDrugRetroDataset",
    ),
    "SynthonDataset": ("retflow.datasets.synthon", "SynthonDataset"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
