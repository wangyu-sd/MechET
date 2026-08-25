"""Memory-efficient causal attention adapter for long, unpadded Tool-SFT rows."""
from __future__ import annotations

from typing import Any


ATTENTION_NAME = "mechet_xformers_causal"


def xformers_causal_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    **_: Any,
) -> tuple[Any, None]:
    """Run causal xFormers attention without constructing an S-by-S mask.

    The Tool-SFT runtime uses per-device batch size one and packing=false, so
    every batch contains one unpadded sequence.  Refuse any explicit mask
    instead of silently changing padding semantics.
    """

    if attention_mask is not None:
        raise ValueError(
            "mechet_xformers_causal expects an unpadded batch and no expanded mask"
        )
    from xformers.ops import LowerTriangularMask, memory_efficient_attention

    groups = int(getattr(module, "num_key_value_groups", 1) or 1)
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    # Transformers uses [batch, heads, sequence, dim]; xFormers uses
    # [batch, sequence, heads, dim].
    output = memory_efficient_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_bias=LowerTriangularMask(),
        p=float(dropout),
        scale=scaling,
    )
    return output, None


def register_xformers_attention() -> str:
    from transformers import AttentionInterface

    AttentionInterface.register(
        ATTENTION_NAME, xformers_causal_attention_forward
    )
    return ATTENTION_NAME
