#!/usr/bin/env python3
"""Product-only executable beam search with a learned state-value critic.

The policy sees only the target, executor-owned current state, and temporary
atom aliases.  The reference precursor is used after search for evaluation and
for selecting successful training trajectories; it is never in a model prompt
or search score.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.assistant_masking import render_chat, render_qwen_sft_tool_prefix
from mechet.endpoints import (
    reference_structural_precursor,
    split_precursor_endpoints,
    structural_exact,
)
from mechet.electron_pointer_runtime import (
    FrozenPointerScorer,
    action_pointer_log_likelihood_ratio,
)
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    map_unmapped_fragment,
    mapped_atom_numbers,
)
from mechet.natural_language_electron_flow import compile_event_arguments
from mechet.natural_language_anchor_branch_rl import contains_unchanged_target
from mechet.successor_value import (
    REACHABILITY_VALUE_SYSTEM, SUCCESSOR_VALUE_SYSTEM,
    reachability_value_prompt, successor_value_prompt,
)
from mechet.trajectory_history import TrajectoryHistory
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS, _decision_row, _prompt
from scripts.build_natural_language_state_value import VALUE_SYSTEM, value_prompt
from scripts.eval_natural_language_event_local import MODEL_REVISION, prediction_call


def read_selected(
    path: Path, size: int, seed: int, *, unique_source: bool = False
) -> list[dict[str, Any]]:
    """Streaming deterministic bottom-k selection."""
    import heapq

    heap: list[tuple[int, int, dict[str, Any]]] = []
    seen_sources: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row.get("source_id") or row["id"]) if unique_source else str(row["id"])
            if unique_source:
                if identifier in seen_sources:
                    continue
                if int((row.get("metadata") or {}).get("decision_index", 0)) != 0:
                    raise ValueError(f"first row for {identifier} is not the product-start decision")
                seen_sources.add(identifier)
            digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
            key = int.from_bytes(digest, "big")
            item = (-key, -line_number, row)
            if len(heap) < size:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda value: (-value[0], -value[1]))]


def visible(state: str) -> str:
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(state or ""), params)
    if mol is None:
        raise ValueError("invalid molecular state")
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if all(value == 0 for value in maps):
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if all(value > 0 for value in maps) and len(set(maps)) == len(maps):
        return deterministic_unmapped_state(state).text
    raise ValueError("partially or multiply mapped molecular state")


def private_product_state(product: str) -> str:
    """Map a product-only input deterministically, without reference reactants."""
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(product or ""), params)
    if mol is None:
        raise ValueError("invalid product SMILES")
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if all(value == 0 for value in maps):
        return map_unmapped_fragment(product, first_map=1)[0]
    if any(value <= 0 for value in maps) or len(set(maps)) != len(maps):
        raise ValueError("partially or multiply mapped product")
    return product


def product_only_private_state(product: str) -> str:
    """Assign fresh private maps from the unmapped product, never source maps."""

    return private_product_state(normal_smiles(product))


def validate_matched_v2_args(args: argparse.Namespace) -> None:
    """Fail closed unless inference matches the frozen protocol-v2 policy."""
    if not bool(getattr(args, "matched_v2", False)):
        return
    if bool(getattr(args, "legacy_dual_prompt", False)):
        raise ValueError("matched v2 forbids the legacy dual prompt")
    if int(getattr(args, "max_decisions", 0)) != 40:
        raise ValueError("matched v2 requires the frozen 40 decisions budget")
    if int(getattr(args, "max_imports", 0)) != 32:
        raise ValueError("matched v2 requires the frozen 32 imports budget")
    if int(getattr(args, "branching", 0)) != 1:
        raise ValueError("matched v2 pure-policy evaluation requires branching=1")
    if int(getattr(args, "early_beam", 0)) != 1 or int(
        getattr(args, "late_beam", 0)
    ) != 1:
        raise ValueError("matched v2 pure-policy evaluation requires beam width 1")
    if str(getattr(args, "value_adapter", "") or "") or abs(
        float(getattr(args, "value_weight", 0.0) or 0.0)
    ) > 1e-12:
        raise ValueError("matched v2 pure-policy evaluation forbids a value critic")


def validate_v2_adapter_manifest(
    adapter: Path, *, compact_history: bool, vnext: bool = False,
    expected_model: str | None = None, expected_revision: str | None = None,
) -> dict[str, Any]:
    """Require an adapter produced by the clean protocol-v2 SFT lineage."""
    manifest_path = adapter / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"adapter manifest is missing: {manifest_path}")
    manifest = dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    expected_environment = (
        "natural_language_electron_event_history_v2"
        if compact_history
        else "natural_language_electron_event_v2"
    )
    if manifest.get("environment_revision") != expected_environment:
        raise ValueError(
            "matched protocol-v2 evaluation refuses this adapter: "
            f"expected environment_revision={expected_environment}, "
            f"observed={manifest.get('environment_revision')}"
        )
    allowed_executors = {"MECH_PROOF_v1_full_coverage_v4"}
    if vnext:
        allowed_executors.add("mech_uspto31k_current_compiler_20260824")
    if manifest.get("executor_revision") not in allowed_executors:
        raise ValueError("protocol-v2 adapter executor lineage is not recognized")
    revision = str(manifest.get("base_model_revision") or "")
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("matched protocol-v2 adapter must pin an immutable base revision")
    if expected_model is not None and manifest.get("base_model") != expected_model:
        raise ValueError("protocol-v2 adapter base model differs from inference model")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("protocol-v2 adapter base revision differs from inference revision")
    return manifest


def normal_smiles(value: str) -> str:
    from rdkit import Chem

    # Explicit hydrogen atoms may be electron-flow participants.  The default
    # RDKit parser removes them, which used to make an apparently successful
    # import diverge from the supervised executor state and invalidate the next
    # event.  Preserve graph atoms here; map_unmapped_fragment does the same.
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(value or ""), params)
    if mol is None:
        raise ValueError("invalid SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@dataclass
class Action:
    name: str
    arguments: dict[str, Any]
    raw: str
    logprob: float
    tokens: int
    pointer_score: float = 0.0


@dataclass
class Node:
    target: str
    state: str
    next_map: int
    actions: list[dict[str, Any]] = field(default_factory=list)
    visited: set[str] = field(default_factory=set)
    imported: dict[str, int] = field(default_factory=dict)
    logprob: float = 0.0
    tokens: int = 0
    value: float = 0.0
    pointer_score: float = 0.0
    terminal: bool = False

    @property
    def policy_score(self) -> float:
        return self.logprob / max(self.tokens, 1)

    def score(self, weight: float, pointer_weight: float = 0.0) -> float:
        pointer_mean = self.pointer_score / max(len(self.actions), 1)
        return self.policy_score + weight * self.value + pointer_weight * pointer_mean


class Runtime:
    def __init__(self, args: argparse.Namespace, local_rank: int):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.args = args
        torch.cuda.set_device(local_rank)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.model_revision, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_kwargs: dict[str, Any] = {
            "revision": args.model_revision,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": {"": local_rank},
            "attn_implementation": "sdpa",
        }
        if not args.no_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        base = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        self.model = PeftModel.from_pretrained(
            base, args.policy_adapter, adapter_name="policy", is_trainable=False
        )
        self.has_value = bool(str(args.value_adapter or "").strip())
        if self.has_value:
            self.model.load_adapter(
                args.value_adapter, adapter_name="value", is_trainable=False
            )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.pointer_invalid_handles = 0
        self.pointer = (
            FrozenPointerScorer(
                Path(args.pointer_head), model=self.model,
                tokenizer=self.tokenizer, adapter=Path(args.policy_adapter),
            )
            if str(args.pointer_head or "").strip() else None
        )
        self.value_kind = str(args.value_kind)
        self.label_ids: dict[str, list[int]] = {}
        if self.has_value:
            labels = "PN" if self.value_kind in {"successor_pn", "reachability_pn"} else "ABC"
            self.label_ids = {
                label: self.tokenizer(label, add_special_tokens=False)["input_ids"]
                for label in labels
            }
            if any(len(ids) != 1 for ids in self.label_ids.values()):
                raise RuntimeError(
                    f"value labels must each be one token: {self.label_ids}"
                )

    def proposals(
        self,
        node: Node,
        *,
        candidates: int,
        max_new_tokens: int,
        compact_history: bool = False,
        planning_sample: bool = False,
    ) -> list[Action]:
        torch = self.torch
        self.last_proposal_error = ""
        self.model.set_adapter("policy")
        target = node.target
        state = node.state
        inventory_modes = [False, True] if self.args.legacy_dual_prompt else [True]
        message_sets = [
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": policy_prompt(
                        target,
                        state,
                        include_inventory=include_inventory,
                        actions=node.actions,
                        compact_history=compact_history,
                    ),
                },
            ]
            for include_inventory in inventory_modes
        ]
        prompts = [
            (
                render_qwen_sft_tool_prefix(
                    self.tokenizer, messages, tools=TOOLS
                )
                if self.args.matched_v2 or self.args.vnext_v2_prefix
                else render_chat(
                    self.tokenizer,
                    messages,
                    tools=TOOLS,
                    add_generation_prompt=True,
                )
            )
            for messages in message_sets
        ]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        width = int(encoded["input_ids"].shape[1])
        if (self.args.matched_v2 or getattr(self.args, "product_only_remap", False)) and (
            width + max_new_tokens > self.args.max_context
        ):
            self.last_proposal_error = "CONTEXT_BUDGET_EXCEEDED"
            return []
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        pointer_logits = (
            [
                self.pointer.logits(prompt, messages[-1]["content"])
                for prompt, messages in zip(prompts, message_sets, strict=True)
            ]
            if self.pointer is not None else []
        )
        with torch.inference_mode():
            generation = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": candidates,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generation.update(generation_sampling_policy(
                matched_v2=bool(self.args.matched_v2),
                vnext_v2_prefix=bool(self.args.vnext_v2_prefix),
                candidates=candidates,
                planning_sample=planning_sample,
            ))
            output = self.model.generate(**encoded, **generation)
        scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        actions: list[Action] = []
        seen: set[str] = set()
        for index, sequence in enumerate(output.sequences):
            ids = sequence[width:].tolist()
            stop = len(ids)
            if self.tokenizer.eos_token_id in ids:
                stop = ids.index(self.tokenizer.eos_token_id) + 1
            ids = ids[:stop]
            raw = self.tokenizer.decode(ids, skip_special_tokens=False)
            name, arguments, error = prediction_call(raw, self.tokenizer)
            prompt_kind = index // candidates
            if error:
                continue
            if self.args.legacy_dual_prompt:
                if prompt_kind == 0 and name not in {"import_fragments", "finish_trace"}:
                    continue
                if prompt_kind == 1 and name != "apply_electron_flow":
                    continue
            pointer_score = 0.0
            if self.pointer is not None:
                pointer_output = pointer_logits[prompt_kind]
                observation, src_logits, sink_logits = pointer_output[:3]
                pointer_score = action_pointer_log_likelihood_ratio(
                    src_logits, sink_logits, observation, name, arguments,
                    pair_logits=pointer_output[3] if len(pointer_output) > 3 else None,
                )
                if not math.isfinite(pointer_score):
                    self.pointer_invalid_handles += 1
                    continue
            signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
            if signature in seen:
                continue
            seen.add(signature)
            token_count = max(stop, 1)
            actions.append(
                Action(
                    name=name,
                    arguments=arguments,
                    raw=raw,
                    logprob=float(scores[index, :stop].sum().item()),
                    tokens=token_count,
                    pointer_score=pointer_score,
                )
            )
        if not actions:
            self.last_proposal_error = "NO_PARSEABLE_TOOL_CALL"
        return actions

    def values(
        self,
        target: str,
        states: list[str],
        *,
        current_states: list[str] | None = None,
        terminal: bool = False,
        remaining_decisions: int | None = None,
        batch_size: int = 16,
    ) -> list[float]:
        if not self.has_value:
            return [0.0] * len(states)
        torch = self.torch
        self.model.set_adapter("value")
        output_values: list[float] = []
        labels = "PN" if self.value_kind in {"successor_pn", "reachability_pn"} else "ABC"
        label_tokens = [self.label_ids[label][0] for label in labels]
        if self.value_kind in {"successor_pn", "reachability_pn"} and (
            current_states is None or len(current_states) != len(states)
        ):
            raise ValueError("successor critic requires one parent state per successor")
        if self.value_kind == "reachability_pn" and remaining_decisions is None:
            raise ValueError("reachability critic requires remaining decision budget")
        for start in range(0, len(states), batch_size):
            batch_states = states[start : start + batch_size]
            if self.value_kind in {"successor_pn", "reachability_pn"}:
                batch_parents = current_states[start : start + batch_size]
                prompts = [
                    render_chat(
                        self.tokenizer,
                        [
                            {"role": "system", "content": (
                                REACHABILITY_VALUE_SYSTEM if self.value_kind == "reachability_pn"
                                else SUCCESSOR_VALUE_SYSTEM
                            )},
                            {
                                "role": "user",
                                "content": (
                                    reachability_value_prompt(
                                        target, parent, state,
                                        terminal=terminal,
                                        remaining_decisions=int(remaining_decisions),
                                    )
                                    if self.value_kind == "reachability_pn"
                                    else successor_value_prompt(
                                        target, parent, state, terminal=terminal
                                    )
                                ),
                            },
                        ],
                        tools=[],
                        add_generation_prompt=True,
                    )
                    for parent, state in zip(batch_parents, batch_states, strict=True)
                ]
            else:
                prompts = [
                    render_chat(
                        self.tokenizer,
                        [
                            {"role": "system", "content": VALUE_SYSTEM},
                            {"role": "user", "content": value_prompt(target, state)},
                        ],
                        tools=[],
                        add_generation_prompt=True,
                    )
                    for state in batch_states
                ]
            encoded = self.tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, label_tokens].float()
                logp = torch.log_softmax(logits, dim=-1)
                if self.value_kind in {"successor_pn", "reachability_pn"}:
                    useful = logp[:, 0] - logp[:, 1]
                else:
                    useful = (
                        logp[:, 1] - torch.logsumexp(logp[:, [0, 2]], dim=-1)
                        if terminal
                        else torch.logsumexp(logp[:, :2], dim=-1) - logp[:, 2]
                    )
            output_values.extend(float(value) for value in useful.cpu().tolist())
        if len(output_values) != len(states):
            raise RuntimeError(
                f"value/state cardinality mismatch: {len(output_values)} != {len(states)}"
            )
        return output_values


def execute(
    node: Node,
    action: Action,
    *,
    max_imports: int,
    reject_target_retained_finish: bool = False,
) -> tuple[Node | None, str]:
    state_before = node.state
    state = node.state
    next_map = node.next_map
    imported = dict(node.imported)
    result: dict[str, Any]
    terminal = False
    try:
        if action.name == "import_fragments":
            fragments = action.arguments.get("fragments")
            if not isinstance(fragments, list) or not fragments:
                raise ValueError("IMPORT_SCHEMA_INVALID")
            expanded: list[str] = []
            for item in fragments:
                if not isinstance(item, Mapping):
                    raise ValueError("IMPORT_SCHEMA_INVALID")
                count = int(item.get("count", 0))
                if count < 1:
                    raise ValueError("IMPORT_COUNT_INVALID")
                expanded.extend([normal_smiles(str(item.get("smiles") or ""))] * count)
            if sum(imported.values()) + len(expanded) > max_imports:
                raise ValueError("IMPORT_BUDGET_EXCEEDED")
            mapped: list[str] = []
            for fragment in expanded:
                # Repeated stoichiometric imports are allowed only when explicitly counted.
                mapped_fragment, next_map = map_unmapped_fragment(fragment, first_map=next_map)
                mapped.append(mapped_fragment)
                imported[fragment] = imported.get(fragment, 0) + 1
            # Preserve the executor's exact Kekule representation.  A mapped
            # canonical round-trip can re-aromatize a symmetric component;
            # the next verify_electron_step then chooses a different Kekule
            # form and applies an otherwise identical arrow to the wrong bond.
            state = append_mapped_fragments_verbatim(state, mapped)
            result = {"ok": True, "code": "PASS", "current_state": visible(state)}
        elif action.name == "apply_electron_flow":
            moves = compile_event_arguments(state, action.arguments)
            replay = verify_electron_step(state, moves)
            if not replay.get("ok"):
                raise ValueError(str(replay.get("code") or "EXECUTION_FAILED"))
            state = str(replay["state_smiles"])
            if visible(state) in node.visited:
                raise ValueError("STATE_CYCLE")
            result = {"ok": True, "code": "PASS", "current_state": visible(state)}
        elif action.name == "finish_trace":
            if action.arguments:
                raise ValueError("FINISH_ARGUMENTS_NOT_EMPTY")
            transformed = any(
                record.get("name") == "apply_electron_flow" for record in node.actions
            )
            if (
                reject_target_retained_finish
                and not transformed
                and contains_unchanged_target(state, node.target)
            ):
                raise ValueError("TARGET_RETAINED_NO_TRANSFORM")
            terminal = True
            result = {"ok": True, "code": "PASS", "derived_precursor": visible(state)}
        else:
            raise ValueError("UNKNOWN_TOOL")
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"
    record = {
        "state_before": state_before,
        "name": action.name,
        "arguments": action.arguments,
        "result": result,
    }
    return Node(
        target=node.target,
        state=state,
        next_map=next_map,
        actions=node.actions + [record],
        visited=set(node.visited) | {visible(state)},
        imported=imported,
        logprob=node.logprob + action.logprob,
        tokens=node.tokens + action.tokens,
        pointer_score=node.pointer_score + action.pointer_score,
        terminal=terminal,
    ), ""


PROMPT_SUFFIX = "\nChoose the single next retrosynthetic action."


def compact_history_from_actions(
    actions: Sequence[Mapping[str, Any]],
) -> TrajectoryHistory:
    """Reconstruct exactly the Stage-II capsule from accepted runtime actions."""

    history = TrajectoryHistory()
    for record in actions:
        history = history.accept(
            str(record["name"]),
            dict(record["arguments"]),
            dict(record["result"]),
        )
    return history


def policy_prompt(
    target: str,
    state: str,
    *,
    include_inventory: bool,
    actions: Sequence[Mapping[str, Any]],
    compact_history: bool,
) -> str:
    """Render either the Stage-I or Stage-II policy observation contract."""

    prompt = _prompt(target, state, include_inventory=include_inventory)
    if not compact_history:
        return prompt
    if not prompt.endswith(PROMPT_SUFFIX):
        raise ValueError("natural-language policy prompt suffix changed")
    return (
        prompt[: -len(PROMPT_SUFFIX)]
        + "\n\n"
        + compact_history_from_actions(actions).render()
        + PROMPT_SUFFIX
    )


def select_successful_terminals(terminals: list[Node], structural_match, expected_full: str):
    """Keep full-endpoint teachers separate from structural-only hits."""
    structural = next((node for node in terminals if structural_match(node)), None)
    full = next(
        (node for node in terminals if node.terminal and visible(node.state) == expected_full),
        None,
    )
    return structural, full


def generation_sampling_policy(
    *, matched_v2: bool, vnext_v2_prefix: bool, candidates: int,
    planning_sample: bool = False,
) -> dict:
    """K=1 vNext is genuinely greedy; K>1 remains stochastic expansion."""
    if candidates < 1:
        raise ValueError("candidate count must be positive")
    greedy = not planning_sample and (
        matched_v2 or (vnext_v2_prefix and candidates == 1)
    )
    return {"do_sample": False} if greedy else {
        "do_sample": True, "temperature": 0.7, "top_p": 0.95,
    }


def search_unlabeled(
    runtime: Runtime, target_smiles: str, args: argparse.Namespace,
) -> dict[str, Any]:
    """Search from a product without reading any reference precursor.

    This is the shared policy/executor path for held-out evaluation and
    on-demand planning of arbitrary intermediate target molecules.
    """
    pointer_rejected_before = runtime.pointer_invalid_handles
    if getattr(args, "vnext_v2_prefix", False) and not mapped_atom_numbers(str(target_smiles)):
        raise ValueError(
            "vNext matched search requires the frozen privately mapped trace source; "
            "unmapped decision rows can reassign stereochemical graph addresses"
        )
    target_mapped = (
        product_only_private_state(str(target_smiles))
        if getattr(args, "product_only_remap", False)
        else private_product_state(str(target_smiles))
    )
    target = visible(target_mapped)
    root = Node(
        target=target,
        state=target_mapped,
        next_map=max(mapped_atom_numbers(target_mapped), default=0) + 1,
        visited={target},
    )
    beam = [root]
    terminals: list[Node] = []
    rejected: dict[str, int] = {}
    attempts: list[dict[str, Any]] = []
    for depth in range(args.max_decisions):
        children: list[Node] = []
        new_terminals: list[Node] = []
        for node in beam:
            proposals = runtime.proposals(
                node,
                candidates=args.branching,
                max_new_tokens=args.max_new_tokens,
                compact_history=args.compact_history,
                planning_sample=bool(getattr(args, "planning_sample", False)),
            )
            if not proposals:
                code = str(getattr(runtime, "last_proposal_error", "") or "NO_VALID_TOOL_CALL")
                rejected[code] = rejected.get(code, 0) + 1
                attempts.append({
                    "depth": depth, "state_before": visible(node.state),
                    "name": "", "arguments": {}, "accepted": False,
                    "error": code, "state_after": "", "terminal": False,
                    "action_logprob": None, "action_tokens": 0,
                    "action_policy_score": None,
                })
            for action in proposals:
                child, error = execute(
                    node,
                    action,
                    max_imports=args.max_imports,
                    reject_target_retained_finish=args.reject_target_retained_finish,
                )
                attempts.append({
                    "depth": depth, "state_before": visible(node.state),
                    "name": action.name, "arguments": action.arguments,
                    "accepted": child is not None, "error": error,
                    "action_logprob": action.logprob,
                    "action_tokens": action.tokens,
                    "action_policy_score": action.logprob / max(action.tokens, 1),
                    "state_after": (
                        str(child.actions[-1]["result"].get("current_state") or
                            child.actions[-1]["result"].get("derived_precursor") or "")
                        if child is not None else ""
                    ),
                    "terminal": bool(child is not None and child.terminal),
                })
                if child is None:
                    rejected[error] = rejected.get(error, 0) + 1
                elif child.terminal:
                    new_terminals.append(child)
                else:
                    children.append(child)
        if new_terminals:
            terminal_values = runtime.values(
                target,
                [child.state for child in new_terminals],
            current_states=[child.actions[-1]["state_before"] for child in new_terminals],
            terminal=True,
            remaining_decisions=args.max_decisions - depth - 1,
            )
            for child, value in zip(new_terminals, terminal_values, strict=True):
                child.value = value
            terminals.extend(new_terminals)
        if not children:
            break
        # Collapse graph-equivalent visible states before paying for critic scores.
        unique: dict[str, Node] = {}
        for child in children:
            key = visible(child.state)
            incumbent = unique.get(key)
            if incumbent is None or child.policy_score > incumbent.policy_score:
                unique[key] = child
        children = list(unique.values())
        values = runtime.values(
            target,
            [child.state for child in children],
            current_states=[child.actions[-1]["state_before"] for child in children],
            remaining_decisions=args.max_decisions - depth - 1,
        )
        for child, value in zip(children, values, strict=True):
            child.value = value
        width = args.early_beam if depth < args.early_depth else args.late_beam
        beam = sorted(
            children,
            key=lambda child: child.score(args.value_weight, args.pointer_weight),
            reverse=True,
        )[:width]
    terminals.sort(key=lambda node: node.score(args.value_weight, args.pointer_weight), reverse=True)
    top = terminals[0] if terminals else (beam[0] if beam else root)
    return {
        "target": target,
        "target_mapped": target_mapped,
        "top": top,
        "terminals": terminals,
        "rejected": rejected,
        "attempts": attempts,
        "pointer_invalid_handles": runtime.pointer_invalid_handles - pointer_rejected_before,
    }


def rollout(runtime: Runtime, row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Score a product-only search only after all model decisions are complete."""
    searched = search_unlabeled(runtime, str(row["target_smiles"]), args)
    target = searched["target"]
    target_mapped = searched["target_mapped"]
    top = searched["top"]
    terminals = searched["terminals"]
    rejected = searched["rejected"]
    attempts = searched["attempts"]
    expected_full_mapped = str(
        row.get("full_precursor_state") or row["expected_precursor"]
    )
    expected_full = visible(expected_full_mapped)
    expected_structural = reference_structural_precursor(dict(row))
    has_structural_reference = bool(expected_structural)

    def is_structural_match(node: Node) -> bool:
        if not node.terminal:
            return False
        if not has_structural_reference:
            return visible(node.state) == expected_full
        predicted = split_precursor_endpoints(node.state, target_mapped).structural
        return structural_exact(predicted, expected_structural)

    any_exact = any(is_structural_match(node) for node in terminals)
    top_exact = is_structural_match(top)
    top_full_exact = bool(top.terminal and visible(top.state) == expected_full)
    successful, successful_full = select_successful_terminals(
        terminals, is_structural_match, expected_full
    )
    successful_full_exact = successful_full is not None
    top_structural = (
        visible(split_precursor_endpoints(top.state, target_mapped).structural)
        if top.terminal and has_structural_reference
        else ""
    )
    return {
        "id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "target": target,
        "expected_precursor": expected_full,
        "expected_structural_precursor": visible(expected_structural) if has_structural_reference else "",
        "endpoint_metric": "structural" if has_structural_reference else "full_unmapped",
        "top_precursor": visible(top.state),
        "top_structural_precursor": top_structural,
        "top_terminal": top.terminal,
        "top1_exact": top_exact,
        "top1_structural_exact": top_exact,
        "top1_full_exact": top_full_exact,
        "pass_at_beam": any_exact,
        "n_terminals": len(terminals),
        "top_policy_score": top.policy_score,
        "top_value": top.value,
        "top_pointer_score": top.pointer_score,
        "pointer_invalid_handles": searched["pointer_invalid_handles"],
        "n_actions": len(top.actions),
        "rejected": rejected,
        "attempts": attempts,
        "top_actions": top.actions,
        "successful_actions": (
            successful_full.actions if successful_full is not None
            else successful.actions if successful is not None else []
        ),
        "successful_full_exact": successful_full_exact,
    }


def distill_rows(result: Mapping[str, Any], source: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = list(result.get("successful_actions") or [])
    rows: list[dict[str, Any]] = []
    public_source = dict(
        source,
        target_smiles=str(result["target"]),
        expected_precursor=str(result["expected_precursor"]),
    )
    for index, record in enumerate(actions):
        rows.append(
            _decision_row(
                row=public_source,
                sequence_index=index,
                decision_type=(
                    "import" if record["name"] == "import_fragments" else
                    "finish" if record["name"] == "finish_trace" else "event"
                ),
                mapped_state=str(record["state_before"]),
                name=str(record["name"]),
                arguments=dict(record["arguments"]),
                result=dict(record["result"]),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--policy-adapter", required=True)
    parser.add_argument("--value-adapter", default="")
    parser.add_argument(
        "--value-kind", choices=["state_abc", "successor_pn", "reachability_pn"], default="state_abc"
    )
    parser.add_argument("--sample-reactions", type=int, default=128)
    parser.add_argument("--reaction-level", action="store_true",
                        help="sample one product-start row per source reaction")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--branching", type=int, default=4)
    parser.add_argument("--early-beam", type=int, default=4)
    parser.add_argument("--late-beam", type=int, default=2)
    parser.add_argument("--early-depth", type=int, default=2)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=40,
        help="maximum accepted tool decisions; covers the frozen SFT contract",
    )
    parser.add_argument(
        "--max-imports",
        type=int,
        default=32,
        help="maximum imported fragment copies; frozen SFT maximum is 24",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--value-weight", type=float, default=0.20)
    parser.add_argument("--pointer-head", default="")
    parser.add_argument("--pointer-weight", type=float, default=0.0)
    parser.add_argument("--vnext-v2-prefix", action="store_true")
    parser.add_argument("--search-no-value", action="store_true")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="load the base model in BF16 (for A100/H20 images without bitsandbytes)",
    )
    parser.add_argument("--reject-target-retained-finish", action="store_true")
    parser.add_argument(
        "--matched-v2",
        action="store_true",
        help=(
            "fail closed on protocol-v2 parity and run a greedy pure-policy "
            "evaluation with the SFT-aligned Qwen tool-call prefix"
        ),
    )
    parser.add_argument(
        "--legacy-dual-prompt",
        action="store_true",
        help=(
            "reproduce the historical action-conditioned two-prompt evaluator; "
            "new unified-observation checkpoints must not use this"
        ),
    )
    parser.add_argument(
        "--compact-history",
        action="store_true",
        help="append executor-reconstructible accepted-action history to policy prompts",
    )
    parser.add_argument(
        "--product-only-remap", action="store_true",
        help="strip source atom maps and deterministically remap only the product before rollout",
    )
    parser.add_argument("--write-distill", action="store_true")
    args = parser.parse_args()
    validate_matched_v2_args(args)
    if args.pointer_head and not args.no_4bit:
        parser.error("the BF16-trained pointer requires --no-4bit")
    if args.pointer_head and (args.legacy_dual_prompt or not args.vnext_v2_prefix):
        parser.error("the pointer requires a unified --vnext-v2-prefix prompt")
    if args.vnext_v2_prefix and (args.legacy_dual_prompt or not args.compact_history):
        parser.error("vNext v2 prefix requires unified compact-history prompts")
    if args.search_no_value and (args.value_adapter or abs(args.value_weight) > 1e-12):
        parser.error("--search-no-value requires no critic and --value-weight 0")
    if not args.matched_v2 and not args.search_no_value and not str(args.value_adapter or "").strip():
        parser.error("--value-adapter is required unless --matched-v2 or --search-no-value")
    if args.matched_v2:
        validate_v2_adapter_manifest(
            Path(args.policy_adapter), compact_history=args.compact_history,
            expected_model=args.model, expected_revision=args.model_revision,
        )
    elif args.vnext_v2_prefix:
        validate_v2_adapter_manifest(
            Path(args.policy_adapter), compact_history=True, vnext=True,
            expected_model=args.model, expected_revision=args.model_revision,
        )
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if args.reaction_level and not args.vnext_v2_prefix:
        parser.error("reaction-level vNext search requires --vnext-v2-prefix")
    rows = read_selected(
        args.data, args.sample_reactions, args.seed,
        unique_source=args.reaction_level,
    )
    selected = [row for index, row in enumerate(rows) if index % world == rank]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"results.shard-{rank:02d}-of-{world:02d}.jsonl"
    distill = args.output / f"distill.shard-{rank:02d}-of-{world:02d}.jsonl"
    runtime = Runtime(args, local_rank)
    print(
        f"[meteor-value-search] rank={rank}/{world} reactions={len(selected)} "
        f"beam={args.early_beam}->{args.late_beam} branching={args.branching}",
        flush=True,
    )
    started = time.time()
    with output.open("w", encoding="utf-8") as sink, distill.open("w", encoding="utf-8") as dsink:
        for number, row in enumerate(selected, 1):
            result = rollout(runtime, row, args)
            sink.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            if args.write_distill and result["pass_at_beam"]:
                for item in distill_rows(result, row):
                    dsink.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            sink.flush()
            dsink.flush()
            print(
                json.dumps(
                    {
                        "stage": "value-search",
                        "rank": rank,
                        "done": number,
                        "total": len(selected),
                        "top1_exact": result["top1_exact"],
                        "pass_at_beam": result["pass_at_beam"],
                        "elapsed_s": round(time.time() - started),
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
