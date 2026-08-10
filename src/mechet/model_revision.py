"""Immutable model/tokenizer revision helpers for reproducible training and inference."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_MUTABLE_REVISION_NAMES = {"", "main", "master", "latest", "head"}


def is_immutable_revision(value: object) -> bool:
    """Return True only for a full Git/Hugging Face commit SHA."""

    return bool(_COMMIT_RE.fullmatch(str(value or "").strip()))


def tokenizer_commit_revision(tokenizer: Any) -> str:
    """Extract the immutable snapshot revision recorded by Transformers/HF Hub."""

    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    candidates = (
        getattr(tokenizer, "_commit_hash", None),
        init_kwargs.get("_commit_hash"),
        init_kwargs.get("commit_hash"),
        init_kwargs.get("revision"),
    )
    for value in candidates:
        text = str(value or "").strip()
        if is_immutable_revision(text):
            return text.lower()
    return ""


def hub_commit_revision(
    model_name_or_path: str,
    requested_revision: str | None,
) -> str:
    """Resolve a remote Hugging Face model revision through Hub metadata."""

    if Path(str(model_name_or_path)).exists():
        return ""
    try:
        from huggingface_hub import model_info
    except ImportError:
        return ""
    info = model_info(
        str(model_name_or_path),
        revision=str(requested_revision or "").strip() or None,
    )
    sha = str(getattr(info, "sha", "") or "").strip()
    return sha.lower() if is_immutable_revision(sha) else ""


def resolve_loaded_model_revision(
    *,
    model_name_or_path: str,
    requested_revision: str | None,
    tokenizer: Any,
) -> dict[str, str | None]:
    """Resolve a mutable request such as ``main`` to an immutable revision.

    Remote model runs must end with a full 40-hex commit SHA. Transformers normally
    records the resolved snapshot on the tokenizer. If that metadata is absent, the
    Hub model record is queried as a fallback. Local model paths are deliberately
    marked as local rather than pretending a Git revision exists; callers should
    separately hash local model artifacts for final experiments.
    """

    requested = str(requested_revision or "").strip()
    tokenizer_revision = tokenizer_commit_revision(tokenizer)
    hub_revision = ""
    if tokenizer_revision:
        resolved = tokenizer_revision
        source = "tokenizer_commit_hash"
    elif is_immutable_revision(requested):
        resolved = requested.lower()
        source = "requested_immutable_revision"
    elif Path(str(model_name_or_path)).exists():
        resolved = None
        source = "local_model_path"
    else:
        hub_revision = hub_commit_revision(model_name_or_path, requested_revision)
        if not hub_revision:
            raise ValueError(
                "MUTABLE_MODEL_REVISION_UNRESOLVED: requested revision "
                f"{requested or '<default>'!r} did not resolve to an immutable "
                "40-hex commit SHA through either the loaded tokenizer or Hugging "
                "Face Hub metadata"
            )
        resolved = hub_revision
        source = "huggingface_hub_model_info"
    return {
        "requested_model_revision": requested or None,
        "resolved_model_revision": resolved,
        "tokenizer_revision": tokenizer_revision or resolved,
        "model_revision_resolution_source": source,
        "hub_resolved_model_revision": hub_revision or None,
    }


def resolve_lineage_revision(
    configured_revision: object,
    adapter_revision: object,
) -> str:
    """Choose the immutable revision that should govern downstream lineage.

    A mutable config request (for example ``main``) never overrides an adapter
    manifest that already records the immutable base-model commit used for SFT.
    """

    configured = str(configured_revision or "").strip()
    adapter = str(adapter_revision or "").strip()
    if adapter and not is_immutable_revision(adapter):
        raise ValueError(
            "ADAPTER_MODEL_REVISION_NOT_IMMUTABLE: "
            f"{adapter!r}"
        )
    if is_immutable_revision(configured):
        configured = configured.lower()
        if adapter and adapter.lower() != configured:
            raise ValueError(
                "Tool-SFT base model revision mismatch: "
                f"{adapter} != {configured}"
            )
        return configured
    if adapter:
        return adapter.lower()
    return ""


def revision_contract(
    *,
    configured_revision: object,
    adapter_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(adapter_manifest or {})
    adapter_revision = str(
        manifest.get("base_model_revision")
        or manifest.get("model_revision")
        or ""
    ).strip()
    resolved = resolve_lineage_revision(configured_revision, adapter_revision)
    requested = str(configured_revision or "").strip()
    return {
        "requested_model_revision": requested or None,
        "adapter_base_model_revision": adapter_revision or None,
        "resolved_model_revision": resolved or None,
        "requested_revision_is_immutable": is_immutable_revision(requested),
        "resolved_revision_is_immutable": is_immutable_revision(resolved),
        "requested_revision_is_mutable_alias": requested.casefold()
        in _MUTABLE_REVISION_NAMES,
    }
