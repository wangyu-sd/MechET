"""Version gates for executor-sensitive chemical runtime behavior."""
from __future__ import annotations

import re


MIN_ENDPOINT_PROCESS_RDKIT = (2026, 3, 4)


def rdkit_version_tuple(version: str) -> tuple[int, int, int]:
    values = [int(value) for value in re.findall(r"\d+", str(version))[:3]]
    if not values:
        raise ValueError(f"cannot parse RDKit version: {version!r}")
    return tuple((values + [0, 0, 0])[:3])  # type: ignore[return-value]


def require_endpoint_process_rdkit() -> str:
    """Reject runtimes known to alter frozen aromatic coupled moves."""

    import rdkit

    current = rdkit_version_tuple(rdkit.__version__)
    if current < MIN_ENDPOINT_PROCESS_RDKIT:
        required = ".".join(str(value) for value in MIN_ENDPOINT_PROCESS_RDKIT)
        raise RuntimeError(
            f"Endpoint-process executor requires RDKit >= {required}; "
            f"found {rdkit.__version__}. Older releases change frozen "
            "aromatic/kekule coupled-move outcomes."
        )
    return str(rdkit.__version__)
