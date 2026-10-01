"""Gold-free source/sink pointer scoring for executed-action search."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .electron_pointer import (
    action_pointer_targets,
    candidate_keys,
    locate_marker_tokens,
    parse_pointer_observation,
)
from .electron_pointer_model import CoupledPointerHead, PointerHead, pairs_for_observation


def action_pointer_log_likelihood_ratio(
    source_logits: Any,
    sink_logits: Any,
    observation: Any,
    action_name: str,
    arguments: Mapping[str, Any],
    pair_logits: Any = None,
) -> float:
    """Return mean log-likelihood gain over a uniform structural-site baseline.

    Import and finish actions are neutral. Empty or malformed electron-flow
    actions and generated handles outside the current inventory are rejected.
    """
    if action_name != "apply_electron_flow":
        return 0.0
    import torch

    try:
        sources, sinks = action_pointer_targets(dict(arguments))
    except (KeyError, TypeError, ValueError):
        return float("-inf")
    if not sources:
        return float("-inf")
    n = len(observation.atom_names)
    source_index = {key: i for i, key in enumerate(candidate_keys(n, observation.bonds, source=True))}
    sink_index = {key: i for i, key in enumerate(candidate_keys(n, observation.bonds, source=False))}
    if any(key not in source_index for key in sources) or any(key not in sink_index for key in sinks):
        return float("-inf")
    if pair_logits is not None:
        joint = torch.log_softmax(pair_logits.float().flatten(), 0)
        values = [
            float(joint[source_index[src] * len(sink_index) + sink_index[sink]])
            + math.log(len(source_index) * len(sink_index))
            for src, sink in zip(sources, sinks, strict=True)
        ]
    else:
        src_logp = torch.log_softmax(source_logits.float(), 0)
        sink_logp = torch.log_softmax(sink_logits.float(), 0)
        values = [
            0.5 * (
                float(src_logp[source_index[src]]) + math.log(len(source_index))
                + float(sink_logp[sink_index[sink]]) + math.log(len(sink_index))
            )
            for src, sink in zip(sources, sinks, strict=True)
        ]
    return sum(values) / len(values)


class FrozenPointerScorer:
    """A Qwen-prefix scorer using the independently trained pointer head.

    The Qwen actor and adapter are shared with the proposal runtime and stay
    frozen. This first integration adds one prefix forward per expanded state;
    it is a measured inference ablation, not a zero-cost deployment path.
    """

    def __init__(self, checkpoint: Path, *, model: Any, tokenizer: Any, adapter: Path):
        import torch

        manifest_path = checkpoint.parent / "pointer_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("artifact_type") not in {
            "pr71_p0_stage2_frozen_policy_source_sink_pointer",
            "pr71_p1_stage2_joint_lm_source_sink_pointer",
        }:
            raise ValueError("unrecognized pointer artifact")
        actual_hash = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        if actual_hash != manifest.get("adapter_sha256"):
            raise ValueError("pointer checkpoint and policy adapter do not match")
        if manifest.get("base_revision") != getattr(tokenizer, "_commit_hash", None):
            # Local staged model caches do not always expose _commit_hash on the
            # tokenizer object; the caller also checks the pinned model revision.
            if getattr(tokenizer, "_commit_hash", None) is not None:
                raise ValueError("pointer checkpoint and tokenizer revision mismatch")
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        hidden = int(model.config.hidden_size)
        self.conditional_pair = bool(manifest.get("conditional_pair", False))
        self.head = (
            CoupledPointerHead(hidden) if self.conditional_pair else PointerHead(hidden)
        ).to(self.device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.head.load_state_dict(state, strict=True)
        self.head.requires_grad_(False)

    def logits(self, prefix: str, user_content: str):
        import torch

        observation = parse_pointer_observation(user_content)
        marker_positions = locate_marker_tokens(self.tokenizer, prefix, observation)
        ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            states = self.model(input_ids=inputs, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
            source_pairs, sink_pairs = pairs_for_observation(observation, self.device)
            output = self.head(
                states[-1], states[marker_positions], source_pairs, sink_pairs
            )
        return observation, *output
