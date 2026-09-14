"""Problem API with lazy imports to avoid runner/experiment import cycles."""

from importlib import import_module


_EXPORTS = {
    "Problem": ("retflow.problems.problem", "Problem"),
    "Retrosynthesis": (
        "retflow.problems.retrosynthesis",
        "Retrosynthesis",
    ),
    "SynthonCompletion": (
        "retflow.problems.synthon_completion",
        "SynthonCompletion",
    ),
    "SynthonRetrosynthesis": (
        "retflow.problems.synthon_retro",
        "SynthonRetrosynthesis",
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
