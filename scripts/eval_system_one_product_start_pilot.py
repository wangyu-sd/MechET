#!/usr/bin/env python3
"""Oracle-free final-mixture-start pilot: route + electrons + train retrieval.

This is a deliberately labelled hybrid diagnostic on the strict 31k trace
view, not the final typed-v2 unified policy or a full 3,120-reaction benchmark.
Its target is the executor's final molecular mixture (often with byproducts),
NOT the principal product molecule in the full-endpoint benchmark. The policy
receives that mixture, its own current state/history, and training IMPORT
examples. Held-out answers are loaded solely by the terminal scorer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.electron_pointer import PointerObservation, candidate_keys, parse_pointer_observation
from mechet.in_place_grounded_flow import deterministic_unmapped_state
from mechet.jev_style_decision import (
    FactorizedTypedElectronFlowHead, TypedQuestion, block_causal_option_mask,
    encode_typed_record,
)
from mechet.system_one_action_family import ACTION_NAMES, ActionFamilyHead
from mechet.system_one_decision import append_option_anchors
from mechet.system_one_replay import execute_pair_indices, pair_indices_to_arguments
from mechet.trajectory_history import TrajectoryHistory
from scripts.audit_system_one_observation_parity import mapped_from_visible, runtime_prompt
from scripts.eval_jev_style_successor import verify_checkpoint_contract
from scripts.eval_system_one_import_retrieval import (
    ImportExample, append_import_batch, fingerprint, load_imports,
    rank_unique_batches,
)
from scripts.train_jev_style_electron_flow import option_label
from scripts.train_system_one_electron_flow import file_sha256, verify_source


@dataclass(frozen=True)
class ProductInput:
    """Policy-visible strict-trace input; ``target`` is a final mixture."""

    reaction_id: str
    target: str
    system: str
    tools: list[dict[str, Any]]


@dataclass(frozen=True)
class ReactionTask:
    policy_input: ProductInput
    expected_precursor: str
    reference_decisions: int


def canonical_visible(smiles: str) -> str:
    """Canonicalize all components while retaining explicit hydrogen atoms."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError("invalid predicted/reference endpoint SMILES")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def load_reactions(path: Path) -> tuple[list[ReactionTask], dict]:
    source = verify_source(path)
    groups: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank source row in {path}")
            row = json.loads(line)
            reaction_id = str(row["metadata"]["reaction_id"])
            groups.setdefault(reaction_id, []).append(row)
    if len(groups) != source["reaction_denominator"]:
        raise ValueError("reaction denominator differs from frozen source")
    tasks = []
    for reaction_id, rows in groups.items():
        if [int(row["metadata"]["decision_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"{reaction_id}: nonconsecutive reference decisions")
        first, last = rows[0], rows[-1]
        if last["metadata"]["decision_type"] != "finish":
            raise ValueError(f"{reaction_id}: no final reference FINISH")
        system = [msg for msg in first["messages"] if msg.get("role") == "system"]
        user = [msg for msg in first["messages"] if msg.get("role") == "user"]
        if len(system) != 1 or len(user) != 1:
            raise ValueError(f"{reaction_id}: missing static system or initial observation")
        target = str(first["target_smiles"])
        if any(str(row["target_smiles"]) != target for row in rows):
            raise ValueError(f"{reaction_id}: target changed within reaction")
        if runtime_prompt(target, target, TrajectoryHistory()) != user[0]["content"]:
            raise ValueError(f"{reaction_id}: product-start prompt differs from SFT")
        tools = [msg for msg in last["messages"] if msg.get("role") == "tool"]
        if len(tools) != 1 or tools[0].get("name") != "finish_trace":
            raise ValueError(f"{reaction_id}: no authoritative terminal result")
        result = json.loads(str(tools[0]["content"]))
        if result.get("ok") is not True or not result.get("derived_precursor"):
            raise ValueError(f"{reaction_id}: reference endpoint did not execute")
        if canonical_visible(str(result["derived_precursor"])) != canonical_visible(
            str(last["expected_precursor"])
        ):
            raise ValueError(f"{reaction_id}: reference endpoint differs from frozen answer")
        tasks.append(ReactionTask(
            ProductInput(reaction_id, target, system[0]["content"], first["tools"]),
            str(result["derived_precursor"]), len(rows),
        ))
    return tasks, source


def typed_runtime_encoding(tokenizer, messages: list[dict], observation: PointerObservation):
    """Use precisely the trained typed-v2 text/options, without answer fields."""
    chunks = [f"{message['role'].upper()}:\n{message['content']}"
              for message in messages if message.get("role") in {"system", "user"}
              and isinstance(message.get("content"), str)]
    if not chunks:
        raise ValueError("runtime typed observation has no state text")
    source = candidate_keys(len(observation.atom_names), observation.bonds, source=True)
    return encode_typed_record(
        tokenizer,
        state="\n\n".join(chunks),
        questions=(
            TypedQuestion("SELECT ELECTRON SOURCE", tuple(option_label(key) for key in source)),
            TypedQuestion("SELECT ELECTRON SINK ATOM OR PAIR", tuple(
                option_label(("atom", index, index))
                for index in range(len(observation.atom_names))
            )),
        ),
        option_isolation=True,
    )


def select_tasks(tasks: list[ReactionTask], *, seed: int, limit: int) -> list[ReactionTask]:
    """Hash-select reaction IDs independently of endpoint/trajectory labels."""
    ordered = sorted(tasks, key=lambda item: (
        hashlib.sha256(f"{seed}:{item.policy_input.reaction_id}".encode()).hexdigest(),
        item.policy_input.reaction_id,
    ))
    return ordered[:limit] if limit else ordered


def verify_model_lineage(route_checkpoint: Path, route_run: Path,
                         typed_checkpoint: Path, source: dict, train_source: dict) -> dict:
    route_manifest = json.loads((route_checkpoint / "run_manifest_epoch1.json").read_text())
    route_report = json.loads((route_run / "report.json").read_text())
    typed_manifest = json.loads((typed_checkpoint / "run_manifest_epoch1.json").read_text())
    typed_preflight = verify_checkpoint_contract(typed_checkpoint, typed_manifest)
    if (route_manifest.get("artifact_type") != "system_one_phase0_decision_policy"
            or route_report.get("artifact_type") != "system_one_phase1a_action_family_result"
            or typed_manifest.get("artifact_type") != "system_one_jev_typed_v2_factorized_decision_policy"):
        raise ValueError("one or more frozen policy artifacts have the wrong type")
    if (route_manifest["model"] != typed_manifest["model"]
            or route_manifest["model_revision"] != typed_manifest["model_revision"]):
        raise ValueError("router and typed policy use different pinned backbones")
    if (route_manifest["train_source"]["sha256"] != train_source["sha256"]
            or typed_manifest["train_source"]["sha256"] != train_source["sha256"]
            or route_report["source_sha256"][Path(source["path"]).stem] != source["sha256"]):
        raise ValueError("frozen model source differs from train/evaluation trace view")
    checks = (
        (route_checkpoint / "adapter_epoch1/adapter_model.safetensors",
         route_manifest["adapter_model_sha256"]),
        (route_run / "action_family_head.pt", route_report["head_sha256"]),
        (typed_checkpoint / "adapter_epoch1/adapter_model.safetensors",
         typed_manifest["adapter_model_sha256"]),
        (typed_checkpoint / "typed_head_epoch1.pt", typed_manifest["decision_head_sha256"]),
    )
    for path, digest in checks:
        if file_sha256(path) != digest:
            raise ValueError(f"frozen model weight hash mismatch: {path}")
    if route_report["phase0_adapter_sha256"] != route_manifest["adapter_model_sha256"]:
        raise ValueError("router head is not bound to its base adapter")
    return {
        "route_manifest": route_manifest,
        "route_report": route_report,
        "typed_manifest": typed_manifest,
        "route_max_length": int(json.loads((route_run / "preflight.json").read_text())["max_length"]),
        "typed_max_length": int(typed_preflight["max_length"]),
    }


class HybridPolicy:
    def __init__(self, *, route_checkpoint: Path, route_run: Path,
                 typed_checkpoint: Path, lineage: dict):
        import torch
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                      else torch.float16 if self.device.type == "cuda" else torch.float32)
        manifest = lineage["typed_manifest"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            manifest["model"], revision=manifest["model_revision"], trust_remote_code=True
        )

        def load_adapter(path: Path):
            base = AutoModel.from_pretrained(
                manifest["model"], revision=manifest["model_revision"], trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation="sdpa" if self.device.type == "cuda" else "eager",
            ).to(self.device)
            return PeftModel.from_pretrained(base, path, is_trainable=False).eval()

        self.route_policy = load_adapter(route_checkpoint / "adapter_epoch1")
        self.typed_policy = load_adapter(typed_checkpoint / "adapter_epoch1")
        hidden = int(self.typed_policy.config.hidden_size)
        if hidden != int(self.route_policy.config.hidden_size):
            raise ValueError("route and typed backbones have different hidden sizes")
        self.route_head = ActionFamilyHead(hidden).to(self.device)
        self.route_head.load_state_dict(torch.load(
            route_run / "action_family_head.pt", map_location="cpu", weights_only=True
        ), strict=True)
        self.route_head.eval()
        self.typed_head = FactorizedTypedElectronFlowHead(
            hidden, int(manifest["pointer_dim"])
        ).to(self.device)
        self.typed_head.load_state_dict(torch.load(
            typed_checkpoint / "typed_head_epoch1.pt", map_location="cpu", weights_only=True
        ), strict=True)
        self.typed_head.eval()
        self.route_cap = lineage["route_max_length"]
        self.typed_cap = lineage["typed_max_length"]

    def route(self, messages: list[dict], tools: list[dict], observation: PointerObservation) -> tuple[str, list[float], int]:
        torch = self.torch
        augmented = append_option_anchors(messages, observation.atom_names)
        prefix = render_qwen_sft_tool_prefix(self.tokenizer, augmented, tools=tools)
        ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if len(ids) > self.route_cap:
            raise ValueError("ROUTE_INPUT_LENGTH_CAP")
        with torch.inference_mode():
            states = self.route_policy(
                input_ids=torch.tensor([ids], dtype=torch.long, device=self.device),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
                use_cache=False,
            ).last_hidden_state[0]
            logits = self.route_head(states[-1])
        return ACTION_NAMES[int(logits.argmax())], logits.cpu().tolist(), len(ids)

    def electrons(self, messages: list[dict], observation: PointerObservation) -> tuple[list[int], int]:
        torch = self.torch
        encoding = typed_runtime_encoding(self.tokenizer, messages, observation)
        if len(encoding.input_ids) > self.typed_cap:
            raise ValueError("TYPED_INPUT_LENGTH_CAP")
        with torch.inference_mode():
            states = self.typed_policy(
                input_ids=torch.tensor([encoding.input_ids], dtype=torch.long, device=self.device),
                attention_mask=block_causal_option_mask(
                    encoding, device=self.device, dtype=self.dtype
                ),
                position_ids=torch.tensor([encoding.position_ids], dtype=torch.long, device=self.device),
                use_cache=False,
            ).last_hidden_state[0]
            decision = self.typed_head(
                states[encoding.decide_indices[0]],
                states[list(encoding.option_indices[0])],
                states[encoding.decide_indices[1]],
                states[list(encoding.option_indices[1])],
            )
            ranked = decision.pair_logits.flatten().topk(
                min(8, decision.pair_logits.numel())
            ).indices.tolist()
        return ranked, len(encoding.input_ids)


class TrainImportRetriever:
    def __init__(self, train: list[ImportExample]):
        self.train = train
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.target_fps = [fingerprint(row.target, self.generator) for row in train]
        self.current_fps = [fingerprint(row.current, self.generator) for row in train]

    def propose(self, target: str, current: str) -> tuple[tuple[str, int], ...]:
        return rank_unique_batches(
            fingerprint(target, self.generator), fingerprint(current, self.generator),
            self.train, self.target_fps, self.current_fps, limit=1,
        )[0]


def execute_ranked_electron_action(
    mapped: str, observation: PointerObservation, ranked: list[int],
    *, legality_backoff: bool = False, executor=execute_pair_indices,
) -> tuple[list[int], dict, int | None, int]:
    """Try the frozen top-two/top-one rule, optionally then legal Top-8 singles.

    The fallback conditions only on executor validity at the policy's own
    state. It never receives a reference move, successor, or endpoint score.
    """
    if not ranked:
        raise ValueError("electron ranking must contain at least one candidate")
    selected = ranked[:2]
    result = executor(mapped, observation, selected)
    attempts = 1
    if result.get("ok"):
        return selected, result, None, attempts
    selected = ranked[:1]
    result = executor(mapped, observation, selected)
    attempts += 1
    if result.get("ok") or not legality_backoff:
        return selected, result, 1, attempts
    original_result = result
    for rank, pair in enumerate(ranked[1:], 2):
        selected = [pair]
        result = executor(mapped, observation, selected)
        attempts += 1
        if result.get("ok"):
            return selected, result, rank, attempts
    return ranked[:1], original_result, 1, attempts


def principal_component_atom_indices(current: str, principal_product: str) -> set[int]:
    """Locate the still-intact input product in a pre-first-event state.

    Stereo is ignored only for component localization: the frozen endpoint
    mapper can omit stereotags that are present in the executable trace view.
    If symmetry yields multiple identical components, any one is acceptable.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    state = Chem.MolFromSmiles(current, params)
    product = Chem.MolFromSmiles(principal_product, params)
    if state is None or product is None:
        raise ValueError("invalid current/principal-product SMILES")
    key = Chem.MolToSmiles(product, canonical=True, isomericSmiles=False)
    matches = [set(indices) for indices in Chem.GetMolFrags(state) if
               Chem.MolFragmentToSmiles(state, atomsToUse=list(indices),
                                        canonical=True, isomericSmiles=False) == key]
    if not matches:
        raise ValueError("principal product is absent before the first electron event")
    return set().union(*matches)


def pair_touches_atoms(observation: PointerObservation, flat: int,
                       atom_indices: set[int]) -> bool:
    source = candidate_keys(len(observation.atom_names), observation.bonds, source=True)
    sink = candidate_keys(len(observation.atom_names), observation.bonds, source=False)
    if not 0 <= int(flat) < len(source) * len(sink):
        raise ValueError("electron pair outside current candidate universe")
    src, dst = source[int(flat) // len(sink)], sink[int(flat) % len(sink)]
    return bool(atom_indices & {src[1], src[2], dst[1], dst[2]})


def execute_first_event_target_focus(
    mapped: str, observation: PointerObservation, ranked: list[int],
    principal_atoms: set[int], *, executor=execute_pair_indices,
) -> tuple[list[int], dict, int | None, int, dict]:
    """Replace a legal context-only first event only if target-localized replay succeeds.

    The original Top-8 legality policy is computed first and retained whenever
    it already touches the input product. No reference move or endpoint is read.
    """
    selected, result, single_rank, attempts = execute_ranked_electron_action(
        mapped, observation, ranked, legality_backoff=True, executor=executor,
    )
    metadata = {"enabled": True, "overrode_baseline": False,
                "baseline_selected_pairs": selected, "baseline_execute_ok": bool(result.get("ok"))}
    if not result.get("ok") or any(pair_touches_atoms(observation, pair, principal_atoms)
                                   for pair in selected):
        metadata["reason"] = "baseline_failed_or_already_target_localized"
        return selected, result, single_rank, attempts, metadata
    target_ranked = [(rank, pair) for rank, pair in enumerate(ranked, 1)
                     if pair_touches_atoms(observation, pair, principal_atoms)]
    if not target_ranked:
        metadata["reason"] = "no_target_localized_pair_in_top8"
        return selected, result, single_rank, attempts, metadata
    best_rank, best_pair = target_ranked[0]
    if best_pair != ranked[0]:
        mixed = [best_pair, ranked[0]]
        alternative = executor(mapped, observation, mixed)
        attempts += 1
        if alternative.get("ok"):
            metadata.update({"overrode_baseline": True, "reason": "legal_mixed_event",
                             "target_pair_rank": best_rank})
            return mixed, alternative, None, attempts, metadata
    for rank, pair in target_ranked:
        alternative = executor(mapped, observation, [pair])
        attempts += 1
        if alternative.get("ok"):
            metadata.update({"overrode_baseline": True, "reason": "legal_target_single",
                             "target_pair_rank": rank})
            return [pair], alternative, rank, attempts, metadata
    metadata["reason"] = "target_localized_candidates_failed_execution"
    return selected, result, single_rank, attempts, metadata


def rollout(policy_input: ProductInput, policy: HybridPolicy, retriever: TrainImportRetriever,
            *, max_actions: int, legality_backoff: bool = False,
            principal_product: str | None = None,
            first_event_target_focus: bool = False,
            principal_target_prompt: bool = False) -> dict:
    if first_event_target_focus and (not legality_backoff or not principal_product):
        raise ValueError("first-event target focus requires Top-8 legality backoff and input product")
    if principal_target_prompt and not principal_product:
        raise ValueError("principal-target prompt requires the input product")
    current = policy_input.target
    target_in_prompt = principal_product if principal_target_prompt else policy_input.target
    history = TrajectoryHistory()
    actions = []
    pending_import = False
    terminal = "ACTION_BUDGET"
    max_route_tokens = max_typed_tokens = 0
    for step in range(max_actions):
        prompt = runtime_prompt(target_in_prompt, current, history)
        messages = [
            {"role": "system", "content": policy_input.system},
            {"role": "user", "content": prompt},
        ]
        observation = parse_pointer_observation(prompt, row_id=f"{policy_input.reaction_id}::runtime_{step}")
        try:
            action, route_logits, route_tokens = policy.route(messages, policy_input.tools, observation)
        except ValueError as exc:
            if str(exc) != "ROUTE_INPUT_LENGTH_CAP":
                raise
            terminal = str(exc)
            break
        max_route_tokens = max(max_route_tokens, route_tokens)
        record = {"step": step, "action": action, "route_logits": route_logits,
                  "route_input_tokens": route_tokens, "state_before": current}
        if action == "finish_trace":
            if history.electron_events < 1 or pending_import:
                terminal = "PREMATURE_OR_PENDING_FINISH"
            else:
                terminal = "FINISHED"
                history = history.accept(action, {}, {"ok": True, "code": "PASS"})
            record["accepted"] = terminal == "FINISHED"
            actions.append(record)
            break
        if action == "import_fragments":
            batch = retriever.propose(policy_input.target, current)
            successor = append_import_batch(current, batch)
            arguments = {"fragments": [
                {"smiles": smiles, "count": count, "purpose": "electron_participant"}
                for smiles, count in batch
            ]}
            result = {"ok": True, "code": "PASS", "current_state": successor,
                      "imported_fragments": sum(count for _, count in batch)}
            history = history.accept(action, arguments, result)
            current = successor
            pending_import = True
            record.update({"accepted": True, "batch": batch, "state_after": current})
            actions.append(record)
            continue
        try:
            ranked, typed_tokens = policy.electrons(messages, observation)
        except ValueError as exc:
            if str(exc) != "TYPED_INPUT_LENGTH_CAP":
                raise
            terminal = str(exc)
            actions.append(record)
            break
        max_typed_tokens = max(max_typed_tokens, typed_tokens)
        mapped = mapped_from_visible(current)
        if first_event_target_focus and history.electron_events == 0:
            selected, result, single_rank, executor_attempts, focus = (
                execute_first_event_target_focus(
                    mapped, observation, ranked,
                    principal_component_atom_indices(current, principal_product),
                )
            )
            record["first_event_target_focus"] = focus
        else:
            selected, result, single_rank, executor_attempts = execute_ranked_electron_action(
                mapped, observation, ranked, legality_backoff=legality_backoff
            )
        record.update({"ranked_top8": ranked, "selected_pairs": selected,
                       "typed_input_tokens": typed_tokens,
                       "execute_ok": bool(result.get("ok")), "code": result.get("code"),
                       "selected_single_rank": single_rank,
                       "executor_attempts": executor_attempts})
        if not result.get("ok"):
            terminal = "ELECTRON_EXECUTION_FAILED"
            actions.append(record)
            break
        successor = deterministic_unmapped_state(result["state_smiles"]).text
        arguments = pair_indices_to_arguments(observation, selected)
        history = history.accept(action, arguments, {
            "ok": True, "code": result.get("code") or "PASS", "current_state": successor,
        })
        current = successor
        pending_import = False
        record.update({"accepted": True, "state_after": current})
        actions.append(record)
    return {
        "id": policy_input.reaction_id,
        "target": policy_input.target,
        "terminal": terminal,
        "completed": terminal == "FINISHED",
        "predicted_precursor": current if terminal == "FINISHED" else None,
        "accepted_actions": len(history.accepted_action_types),
        "electron_events": history.electron_events,
        "import_batches": history.import_batches,
        "max_route_input_tokens": max_route_tokens,
        "max_typed_input_tokens": max_typed_tokens,
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--route-checkpoint", type=Path, required=True)
    parser.add_argument("--route-run", type=Path, required=True)
    parser.add_argument("--typed-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--legality-backoff", action="store_true",
                        help="after frozen top-two/top-one failure, try ranked Top-8 singles")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit < 0 or args.max_actions < 1 or args.log_every < 1:
        raise ValueError("invalid pilot limit or action/log budget")
    status = json.loads((args.data_dir / "ARTIFACT_STATUS.json").read_text())
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if (status.get("artifact_id") != manifest.get("artifact_type")
            or not status.get("training_allowed") or not manifest.get("training_allowed")):
        raise ValueError("product-start pilot requires the validated current-compiler trace view")
    train_imports, train_source = load_imports(args.data_dir / "train.jsonl")
    tasks, source = load_reactions(args.data_dir / f"{args.split}.jsonl")
    selected = select_tasks(tasks, seed=args.seed, limit=args.limit)
    lineage = verify_model_lineage(
        args.route_checkpoint, args.route_run, args.typed_checkpoint, source, train_source
    )
    print(json.dumps({"phase": "preflight", "split": args.split,
                      "input_contract": "strict_trace_view_final_molecular_mixture",
                      "all_reactions": len(tasks), "selected": len(selected),
                      "train_imports": len(train_imports),
                      "train_distinct_import_batches": len({row.batch for row in train_imports}),
                      "source_sha256": source["sha256"]}), flush=True)
    if args.preflight_only:
        return
    policy = HybridPolicy(route_checkpoint=args.route_checkpoint, route_run=args.route_run,
                          typed_checkpoint=args.typed_checkpoint, lineage=lineage)
    retriever = TrainImportRetriever(train_imports)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    counts: dict[str, int] = {}
    with (args.output / "cases.jsonl").open("w") as handle:
        for index, task in enumerate(selected, 1):
            result = rollout(task.policy_input, policy, retriever,
                             max_actions=args.max_actions,
                             legality_backoff=args.legality_backoff)
            exact = (result["completed"] and canonical_visible(result["predicted_precursor"])
                     == canonical_visible(task.expected_precursor))
            result["endpoint_exact"] = bool(exact)
            result["expected_precursor"] = task.expected_precursor
            result["reference_decisions"] = task.reference_decisions
            counts[result["terminal"]] = counts.get(result["terminal"], 0) + 1
            counts["endpoint_exact"] = counts.get("endpoint_exact", 0) + int(exact)
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            handle.flush()
            if index % args.log_every == 0 or index == len(selected):
                print(json.dumps({"phase": "product_start_rollout", "reactions": index,
                                  "total": len(selected), "endpoint_exact": counts["endpoint_exact"],
                                  "terminals": {key: value for key, value in counts.items()
                                                if key != "endpoint_exact"},
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    report = {
        "artifact_type": "system_one_pr81_hybrid_final_mixture_start_retrieval_pilot",
        "scope": "final_mixture_start_autonomous_decisions_strict_step_executor_no_formal_proof_compilation",
        "input_contract": "strict_trace_view_final_molecular_mixture_not_principal_product",
        "split": args.split,
        "source": source,
        "train_import_source": train_source,
        "reaction_denominator": len(tasks),
        "evaluated_reactions": len(selected),
        "selection": {"method": "sha256_seed_reaction_id", "seed": args.seed,
                      "limit": args.limit},
        "max_actions": args.max_actions,
        "legality_backoff": args.legality_backoff,
        "evaluator_sha256": file_sha256(Path(__file__)),
        "import_retriever_sha256": file_sha256(
            ROOT / "scripts/eval_system_one_import_retrieval.py"
        ),
        "pointer_parser_sha256": file_sha256(ROOT / "src/mechet/electron_pointer.py"),
        "policy": (
            "frozen_v1_route_plus_typed_v2_electrons_plus_train_only_import_retrieval"
            + ("_top8_executor_legality_backoff" if args.legality_backoff else "")
        ),
        "endpoint_exact": counts["endpoint_exact"],
        "endpoint_exact_rate": counts["endpoint_exact"] / len(selected),
        "terminal_counts": {key: value for key, value in counts.items() if key != "endpoint_exact"},
        "elapsed_s": time.perf_counter() - started,
        "weights": {
            "route_adapter_sha256": lineage["route_manifest"]["adapter_model_sha256"],
            "route_head_sha256": lineage["route_report"]["head_sha256"],
            "typed_adapter_sha256": lineage["typed_manifest"]["adapter_model_sha256"],
            "typed_head_sha256": lineage["typed_manifest"]["decision_head_sha256"],
        },
        "limitations": [
            "input is the strict trace's final molecular mixture, often with byproducts; not the principal product in the full 3120-row benchmark",
            "strict per-event execution only; whole-trajectory formal proof not compiled",
            "train-retrieval import support contains only eight batches on this trace view",
            "not the full 3120-reaction mech-USPTO endpoint benchmark",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "report": report}), flush=True)


if __name__ == "__main__":
    main()
