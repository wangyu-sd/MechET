"""Typed Jev-style decision encoding and readout for chemical actions.

This implementation follows the public decision-model pattern used by Kev:
one shared state, typed question branches, explicit option spans, <decide>
readouts, and option-isolated block-causal attention. It does not claim to
reproduce TypeSafe AI's private Jev internals.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

CONTROL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
OPT_NONE = -1
OPT_DECIDE = -2
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


@dataclass(frozen=True)
class TypedQuestion:
    name: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class JevEncoding:
    input_ids: tuple[int, ...]
    segment_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    option_owner: tuple[int, ...]
    decide_indices: tuple[int, ...]
    option_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TypedPairOutput:
    source_logits: torch.Tensor
    sink_logits: torch.Tensor
    pair_logits: torch.Tensor


def _user_tokens(tokenizer: Any, text: str) -> list[int]:
    safe = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    return list(tokenizer(safe, add_special_tokens=False)["input_ids"])


def resolve_control_ids(tokenizer: Any) -> tuple[int, int, int, int, int]:
    ids = []
    unk = getattr(tokenizer, "unk_token_id", None)
    for token in CONTROL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = list(tokenizer(token, add_special_tokens=False)["input_ids"])
        if token_id is None or token_id == unk or encoded != [token_id]:
            raise ValueError(
                f"tokenizer does not expose {token!r} as one stable control token"
            )
        ids.append(int(token_id))
    if len(set(ids)) != len(ids):
        raise ValueError("Jev control tokens must be distinct")
    return tuple(ids)  # type: ignore[return-value]


def encode_typed_record(
    tokenizer: Any,
    *,
    state: str,
    questions: Sequence[TypedQuestion],
    option_isolation: bool = True,
) -> JevEncoding:
    """Pack one shared state and independent typed question branches."""
    if not state.strip():
        raise ValueError("state cannot be empty")
    if not questions:
        raise ValueError("at least one typed question is required")
    state_id, q_id, opt_start, opt_end, decide_id = resolve_control_ids(tokenizer)

    state_tokens = [state_id] + _user_tokens(tokenizer, state)
    ids = list(state_tokens)
    seg = [0] * len(state_tokens)
    pos = list(range(len(state_tokens)))
    owner = [OPT_NONE] * len(state_tokens)
    decide_indices: list[int] = []
    option_indices: list[tuple[int, ...]] = []
    state_len = len(state_tokens)

    for q_index, question in enumerate(questions, start=1):
        if not question.options:
            raise ValueError(f"{question.name}: typed question has no options")
        if len(set(question.options)) != len(question.options):
            raise ValueError(f"{question.name}: duplicate typed options")
        instruction = [q_id] + _user_tokens(tokenizer, question.name)
        spans = [
            [opt_start] + _user_tokens(tokenizer, option) + [opt_end]
            for option in question.options
        ]
        branch = instruction + [tok for span in spans for tok in span] + [decide_id]
        base = len(ids)

        if option_isolation:
            p0 = state_len
            branch_pos = list(range(p0, p0 + len(instruction)))
            span_start = p0 + len(instruction)
            for span in spans:
                branch_pos.extend(range(span_start, span_start + len(span)))
            branch_pos.append(span_start + max(len(span) for span in spans))
        else:
            branch_pos = list(range(state_len, state_len + len(branch)))

        branch_owner = [OPT_NONE] * len(instruction)
        ends = []
        cursor = len(instruction)
        for option_index, span in enumerate(spans):
            branch_owner.extend([option_index] * len(span))
            cursor += len(span)
            ends.append(base + cursor - 1)
        branch_owner.append(OPT_DECIDE)

        ids.extend(branch)
        seg.extend([q_index] * len(branch))
        pos.extend(branch_pos)
        owner.extend(branch_owner)
        decide_indices.append(base + len(branch) - 1)
        option_indices.append(tuple(ends))

    return JevEncoding(
        input_ids=tuple(ids),
        segment_ids=tuple(seg),
        position_ids=tuple(pos),
        option_owner=tuple(owner),
        decide_indices=tuple(decide_indices),
        option_indices=tuple(option_indices),
    )


def block_causal_option_mask(
    encoding: JevEncoding,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return additive [1,1,L,L] branch- and option-isolated attention."""
    seg = torch.tensor(encoding.segment_ids, dtype=torch.long, device=device)
    own = torch.tensor(encoding.option_owner, dtype=torch.long, device=device)
    length = seg.numel()
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same_branch = (seg[None, :] == seg[:, None]) | (seg[None, :] == 0)
    allow = causal & same_branch

    key_is_option = own[None, :] >= 0
    query_is_decide = own[:, None] == OPT_DECIDE
    same_option = own[None, :] == own[:, None]
    allow = allow & (~key_is_option | query_is_decide | same_option)

    allow |= torch.eye(length, dtype=torch.bool, device=device)
    mask = torch.zeros((length, length), dtype=dtype, device=device)
    mask = mask.masked_fill(~allow, torch.finfo(dtype).min)
    return mask[None, None]


class OptionPointerHead(nn.Module):
    """Typed <decide> query to explicit <option> representation readout."""

    def __init__(self, hidden_size: int, pointer_dim: int = 256):
        super().__init__()
        self.query = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.key = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.scale = 1.0 / math.sqrt(pointer_dim)
        self.temperature = 1.0

    def forward(self, decide_state: torch.Tensor, option_states: torch.Tensor) -> torch.Tensor:
        logits = self.key(option_states.float()) @ self.query(decide_state.float())
        logits = logits * self.scale
        if not self.training and self.temperature != 1.0:
            logits = logits / self.temperature
        return logits


class TypedElectronFlowHead(nn.Module):
    """SOURCE and SINK typed questions plus a coupled move score."""

    def __init__(self, hidden_size: int, pointer_dim: int = 256):
        super().__init__()
        self.source = OptionPointerHead(hidden_size, pointer_dim)
        self.sink = OptionPointerHead(hidden_size, pointer_dim)
        self.pair_source = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.pair_sink = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.scale = 1.0 / math.sqrt(pointer_dim)

    def forward(
        self,
        source_decide: torch.Tensor,
        source_options: torch.Tensor,
        sink_decide: torch.Tensor,
        sink_options: torch.Tensor,
    ) -> TypedPairOutput:
        source_logits = self.source(source_decide, source_options)
        sink_logits = self.sink(sink_decide, sink_options)
        pair_logits = (
            self.pair_source(source_options.float())
            @ self.pair_sink(sink_options.float()).T
        ) * self.scale
        pair_logits = pair_logits + source_logits[:, None] + sink_logits[None, :]
        return TypedPairOutput(source_logits, sink_logits, pair_logits)


class FactorizedTypedElectronFlowHead(nn.Module):
    """Compose unordered sink-pair options from O(n) explicit atom options.

    The numeric score space still contains every atom and every unordered atom
    pair. Only the *textual* option list is factorized, avoiding O(n²) tokens
    and O(n⁴) dense-attention memory for large molecular states.
    """

    def __init__(self, hidden_size: int, pointer_dim: int = 256):
        super().__init__()
        self.source = OptionPointerHead(hidden_size, pointer_dim)
        self.sink = OptionPointerHead(hidden_size, pointer_dim)
        self.pair_compose = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.pair_source = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.pair_sink = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.scale = 1.0 / math.sqrt(pointer_dim)

    def forward(
        self,
        source_decide: torch.Tensor,
        source_options: torch.Tensor,
        sink_decide: torch.Tensor,
        sink_atom_options: torch.Tensor,
    ) -> TypedPairOutput:
        atom_states = sink_atom_options.float()
        n_atoms = atom_states.shape[0]
        pair_indices = torch.triu_indices(n_atoms, n_atoms, offset=1,
                                          device=atom_states.device)
        left = atom_states[pair_indices[0]]
        right = atom_states[pair_indices[1]]
        pair_states = self.pair_compose(
            torch.cat((left + right, torch.abs(left - right), left * right), dim=-1)
        )
        sink_options = torch.cat((atom_states, pair_states), dim=0)
        source_states = source_options.float()
        source_logits = self.source(source_decide, source_states)
        sink_logits = self.sink(sink_decide, sink_options)
        pair_logits = (
            self.pair_source(source_states) @ self.pair_sink(sink_options).T
        ) * self.scale
        pair_logits = pair_logits + source_logits[:, None] + sink_logits[None, :]
        return TypedPairOutput(source_logits, sink_logits, pair_logits)


def required_target_nll(logits: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    """Mean NLL over every required target in a multi-flow event."""
    unique = sorted(set(int(index) for index in indices))
    if not unique:
        raise ValueError("at least one target index is required")
    logp = torch.log_softmax(logits.reshape(-1).float(), dim=0)
    target = torch.tensor(unique, dtype=torch.long, device=logits.device)
    return -logp[target].mean()
