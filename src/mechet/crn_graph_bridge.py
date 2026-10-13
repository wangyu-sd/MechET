"""Optional CRN-JEPA graph electron moves + frozen Qwen3 control/fragment calls.

This module does not replace or modify the historical MechET executor. Import
and finish decisions remain Qwen Tool-SFT decisions; ONLY apply_electron_flow
arguments are provided by the frozen/pretrained CRN-JEPA graph policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

GRAPH_PRETRAIN_SHA256 = "bb2f7fe53e87fc0e2438c27065ceb588dcd38e604959c9d69fb3984885a376ee"
TOOLS = ("import_fragments", "apply_electron_flow", "finish_trace")
CALL_BEGIN = '<tool_call>\n{"name": "'
TOOL_NAME_PATTERN = re.compile(r'<tool_call>\s*\{\s*"name"\s*:\s*"([a-z_]+)"')


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_prefix(name: str) -> str:
    if name not in TOOLS:
        raise ValueError(f"not a MechET tool: {name}")
    return CALL_BEGIN + name + '", "arguments": '


def _shared_prefix_length(sequences: Sequence[Sequence[int]]) -> int:
    if len(sequences) < 2 or any(not row for row in sequences):
        raise ValueError("missing tokenized tool choices")
    count = 0
    for column in zip(*sequences):
        if len(set(column)) != 1:
            break
        count += 1
    if count == 0 or count >= min(map(len, sequences)):
        raise ValueError("tool choices lack a discriminative suffix")
    return count


def route_tool_name(model: Any, tokenizer: Any, device: Any, prompt: str) -> tuple[str, dict[str, float]]:
    """Score Qwen's tool-name continuation, without decoding its FLOW arguments.

    This performs one batched forward call and requests only the last K logits
    from Qwen3 (logits_to_keep). A tokenizer-level longest common prefix handles
    different BPE boundaries for import/apply/finish. No labels are used.
    """
    import torch
    import torch.nn.functional as F

    options = [prompt + CALL_BEGIN + name + '"' for name in TOOLS]
    token_ids = [list(tokenizer.encode(text, add_special_tokens=False)) for text in options]
    common = _shared_prefix_length(token_ids)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if common < len(prompt_ids) - 4:
        raise ValueError("Qwen tool-choice prefix diverged from frozen prompt")
    if tokenizer.pad_token_id is None:
        raise ValueError("Qwen pad_token_id is missing")

    max_len = max(map(len, token_ids))
    keep = max_len - common + 1
    rows = []
    masks = []
    for ids in token_ids:
        pad = max_len - len(ids)
        rows.append([tokenizer.pad_token_id] * pad + ids)
        masks.append([0] * pad + [1] * len(ids))

    input_ids = torch.tensor(rows, device=device, dtype=torch.long)
    attention_mask = torch.tensor(masks, device=device, dtype=torch.long)
    with torch.inference_mode():
        # Qwen3 supports logits_to_keep: never materialize full-context vocab logits.
        output = model(
            input_ids=input_ids, attention_mask=attention_mask,
            logits_to_keep=keep, use_cache=False,
        )
        logits = output.logits.float()
        if logits.shape[1] != keep:
            raise RuntimeError("Qwen backend did not honor logits_to_keep")
        scores = {}
        for i, (name, ids) in enumerate(zip(TOOLS, token_ids, strict=True)):
            pad = max_len - len(ids)
            token_indices = list(range(common, len(ids)))
            positions = torch.tensor(
                [pad + j - 1 - (max_len - keep) for j in token_indices],
                device=device, dtype=torch.long,
            )
            targets = torch.tensor(
                [ids[j] for j in token_indices], device=device, dtype=torch.long,
            )
            nll = F.cross_entropy(logits[i, positions], targets, reduction="mean")
            scores[name] = -float(nll.item())
    return max(TOOLS, key=lambda name: scores[name]), scores


@dataclass(frozen=True)
class GreedyToolRoute:
    name: str
    token_ids: tuple[int, ...]
    logprob: float
    import_arguments: dict[str, Any] | None = None
    raw: str = ""
    name_tokens: int = 0


def route_tool_name_greedy(
    model: Any, tokenizer: Any, device: Any, prompt: str,
    *, max_route_tokens: int = 64,
) -> GreedyToolRoute:
    """Read the actual greedy tool name; finish IMPORT in the same generation.

    Unlike independently scored tool-name suffixes, this has the same prompt
    and greedy token trajectory as the historical K=1 Qwen full-call decoder.
    FLOW/FINISH stop immediately after the name; IMPORT continues to EOS so
    no second prefill can alter fragment generation. No reference is inserted.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    class StopAfterToolName(StoppingCriteria):
        def __init__(self):
            self.name = ""
            self.name_tokens = 0

        def __call__(self, input_ids, scores, **kwargs):
            if self.name == "import_fragments":
                return False
            continuation = tokenizer.decode(
                input_ids[0, prompt_length:].tolist(), skip_special_tokens=False,
            )
            match = TOOL_NAME_PATTERN.search(continuation)
            if match:
                self.name = match.group(1)
                self.name_tokens = input_ids.shape[1] - prompt_length
                return self.name != "import_fragments"
            return False

    stopper = StopAfterToolName()
    with torch.inference_mode():
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=max_route_tokens,
            return_dict_in_generate=True, output_scores=True,
            stopping_criteria=StoppingCriteriaList([stopper]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if stopper.name not in TOOLS:
        raise ValueError(f"QWEN_TOOL_ROUTE_INVALID:{stopper.name or 'no tool name'}")
    continuation = tuple(generated.sequences[0, prompt_length:].tolist())
    scores = model.compute_transition_scores(
        generated.sequences, generated.scores, normalize_logits=True,
    )
    raw = tokenizer.decode(continuation, skip_special_tokens=False)
    import_arguments = None
    if stopper.name == "import_fragments":
        from scripts.eval_natural_language_event_local import prediction_call

        name, import_arguments, error = prediction_call(raw, tokenizer)
        if error or name != "import_fragments":
            raise ValueError(f"QWEN_IMPORT_INVALID:{error or name}")
    return GreedyToolRoute(
        name=stopper.name, token_ids=continuation,
        logprob=float(scores[0, :stopper.name_tokens].sum().item()),
        import_arguments=import_arguments, raw=raw,
        name_tokens=stopper.name_tokens,
    )


def graph_event_coordinates(actions: Sequence[Mapping[str, Any]], initial_budget: int) -> tuple[int, int]:
    """Derive graph observation depth from accepted actions, never from GT."""
    if not 1 <= initial_budget <= 32:
        raise ValueError("graph initial budget must be 1..32")
    depth = sum(str(action.get("name")) == "apply_electron_flow" for action in actions)
    return depth, max(1, initial_budget - depth)


@dataclass(frozen=True)
class GraphFlowCandidate:
    arguments: dict[str, Any]
    moves: tuple[dict[str, Any], ...]
    logprob: float


class FrozenCRNGraphPolicy:
    """Reuse CRN-JEPA PR18 weights, including its already trained FLOW head."""

    def __init__(
        self, *, root: Path, checkpoint: Path, device: Any,
        expected_sha256: str = GRAPH_PRETRAIN_SHA256,
        policy_head_checkpoint: Path | None = None,
    ) -> None:
        import torch

        root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
        if not (root / "src/crn_jepa/models/pr18_full_corpus_world.py").is_file():
            raise FileNotFoundError("CRN-JEPA checkout must contain PR18 model code")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CRN-JEPA weights not found: {checkpoint}")
        actual = file_sha256(checkpoint)
        if actual != expected_sha256:
            raise ValueError(f"CRN-JEPA checkpoint SHA mismatch: {actual}")
        sys.path.insert(0, str(root / "src"))
        from crn_jepa.models.pr18_full_corpus_world import DirectionalCRNPretrainFull
        from crn_jepa.models.grounded_graph import build_grounded_graph
        from crn_jepa.models.grounded_event_policy import decode_events
        from crn_jepa.grounded_actions import MechETGroundedInventory
        from mechet.natural_language_electron_flow import render_event_arguments

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        config = payload["run_config"]
        model = DirectionalCRNPretrainFull(
            hidden_dim=int(config["hidden_dim"]),
            graph_layers=int(config["graph_layers"]),
            num_heads=int(config["heads"]),
        )
        model.load_state_dict(payload["model"], strict=True)
        if policy_head_checkpoint:
            update = torch.load(
                Path(policy_head_checkpoint), map_location="cpu",
                weights_only=True,
            )
            if update.get("base_checkpoint_sha256") != actual:
                raise ValueError("RL policy head belongs to a different CRN-JEPA checkpoint")
            head = dict(update["policy_head"])
            original = model.policy.state_dict()
            if not head or any(
                key not in original or key.startswith("graph_encoder.")
                or original[key].shape != value.shape
                for key, value in head.items()
            ):
                raise ValueError("RL policy head contains unknown/incompatible tensors")
            original.update(head)
            model.policy.load_state_dict(original, strict=True)
        self.model = model.to(device).eval()
        self.device = device
        self.sha256 = actual
        self._build_graph = build_grounded_graph
        self._decode = decode_events
        self._inventory = MechETGroundedInventory
        self._render = render_event_arguments

    def exploration_candidates(self, current_mapped_smiles: str, *, count: int, rng,
                               event_depth: int = 0, remaining_budget: int = 1):
        """Native full-head draws expand support BEFORE conditional IID sampling."""
        import torch
        from mechet.graph_endpoint_rl import sample_grounded_event
        if not 0 <= count <= 32:
            raise ValueError("invalid exploration proposal count")
        if not count:
            return ()
        with torch.inference_mode():
            graph = self._build_graph(current_mapped_smiles, None)
            inventory = self._inventory(current_mapped_smiles)
            nodes, context, prefix = self.model.policy.encode_directional_observation(
                graph, event_depth=event_depth, remaining_budget=remaining_budget,
                direction="retrosynthetic_inverse")
            proposals = []
            for _ in range(count):
                sampled = sample_grounded_event(self.model.policy, graph, inventory,
                                                nodes, context, prefix, rng)
                if sampled is None:
                    continue
                moves, logprob = sampled
                raw_moves = tuple(move.to_raw() for move in moves)
                try:
                    arguments = self._render(current_mapped_smiles, raw_moves)
                except ValueError:
                    # A proposal not representable by this schema cannot enter
                    # the actor support. No gold move or weaker executor is used.
                    continue
                proposals.append(GraphFlowCandidate(arguments, raw_moves, logprob))
        return tuple(proposals)

    def candidates(
        self, current_mapped_smiles: str, *, limit: int = 1,
        event_depth: int = 0, remaining_budget: int = 1,
        beam_size: int = 16, source_width: int = 8,
        sink_width: int = 8, move_width: int = 16,
    ) -> tuple[GraphFlowCandidate, ...]:
        import torch
        if not 1 <= limit <= 32:
            raise ValueError("graph top-k must be between 1 and 32")
        if not 0 <= event_depth <= 32 or not 1 <= remaining_budget <= 32:
            raise ValueError("invalid graph event depth or remaining budget")
        state = current_mapped_smiles
        with torch.inference_mode():
            graph = self._build_graph(state, None)  # matches PR18 first-event view
            inventory = self._inventory(state)
            choices, _ = self._decode(
                self.model.policy, graph, inventory,
                event_depth=event_depth, remaining_budget=remaining_budget,
                direction="retrosynthetic_inverse",
                max_results=limit, beam_size=beam_size,
                source_width=source_width, sink_width=sink_width,
                move_width=move_width,
            )
        return tuple(
            GraphFlowCandidate(
                arguments=self._render(state, tuple(move.to_raw() for move in item.moves)),
                moves=tuple(move.to_raw() for move in item.moves),
                logprob=float(item.logprob),
            )
            for item in choices
        )


def generate_qwen_import(
    model: Any, tokenizer: Any, device: Any, prompt: str,
    *, max_new_tokens: int,
) -> tuple[dict[str, Any], str, int]:
    """Generate only the IMPORT arguments, preserving the frozen tool-call schema."""
    import torch
    from scripts.eval_natural_language_event_local import prediction_call

    if max_new_tokens < 8:
        raise ValueError("too few tokens to generate a fragment import")
    full_prompt = prompt + tool_prefix("import_fragments")
    ids = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    ids = {key: val.to(device) for key, val in ids.items()}
    with torch.inference_mode():
        generated = model.generate(
            **ids, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, ids["input_ids"].shape[1]:].tolist()
    raw = tool_prefix("import_fragments") + tokenizer.decode(
        continuation, skip_special_tokens=False
    )
    name, arguments, error = prediction_call(raw, tokenizer)
    if error or name != "import_fragments":
        raise ValueError(f"QWEN_IMPORT_INVALID:{error or name}")
    return arguments, raw, len(continuation)
