#!/usr/bin/env python3
"""Product-only K-episode MechET evaluation with gold-independent NLL ranking.

K=1 is the existing greedy policy. K>1 samples K separately seeded, complete
executor trajectories from the same product; it never injects a reference
action, precursor, or state. The aggregate step reads the reference only after
all episodes have been written and keeps missing reactions in the denominator.
"""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.a7_rescue import stratified_sample
from mechet.endpoints import reference_structural_precursor, split_precursor_endpoints, structural_exact
from mechet.in_place_grounded_flow import mapped_atom_numbers
from mechet.jevretro_endpoint import canonical_unmapped
from scripts.eval_natural_language_event_local import validate_adapter_lineage
from scripts.run_natural_language_value_search import (
    Action, Node, Runtime, execute, product_only_private_state, read_selected,
    search_unlabeled, validate_matched_v2_args, validate_v2_adapter_manifest,
    validate_compact_flow_v3_adapter_manifest, visible,
)


MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EVALUATION_DEPENDENCIES = (
    "scripts/build_natural_language_event_sft.py",
    "scripts/eval_natural_language_event_local.py",
    "src/mechet/assistant_masking.py",
    "src/mechet/endpoints.py",
    "src/mechet/forward_expert.py",
    "src/mechet/in_place_grounded_flow.py",
    "src/mechet/jevretro_endpoint.py",
    "src/mechet/natural_language_anchor_branch_rl.py",
    "src/mechet/natural_language_electron_flow.py",
    "src/mechet/proof_program.py",
    "src/mechet/trajectory_history.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    """Freeze versions that can change sampling or formal chemistry replay."""

    import rdkit
    import torch

    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": metadata.version("transformers"),
        "peft": metadata.version("peft"),
    }


def evaluation_dependency_hashes(*, compact_flow_v3: bool = False) -> dict[str, str]:
    """Bind resumed shards to the exact action grammar and evaluator code."""

    files = list(EVALUATION_DEPENDENCIES)
    if compact_flow_v3:
        files.extend([
            "src/mechet/compact_electron_flow.py",
            "scripts/build_compact_electron_flow_sft.py",
        ])
    return {relative: sha256(ROOT / relative) for relative in files}


def episode_seed(base_seed: int, target: str, index: int) -> int:
    """Match the on-demand planning provider's per-product seed schedule."""

    digest = hashlib.sha256(
        f"{base_seed}:{canonical_unmapped(target)}:{index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def policy_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=MODEL, model_revision=REVISION,
        policy_adapter=str(args.adapter), value_adapter="",
        value_kind="state_abc", pointer_head="", pointer_weight=0.0,
        value_weight=0.0, no_4bit=args.no_4bit, matched_v2=True,
        dtype=getattr(args, "dtype", "bfloat16"),
        raw_model_nll=True,
        vnext_v2_prefix=False, legacy_dual_prompt=False,
        product_only_remap=True, reject_target_retained_finish=True,
        compact_history=args.stage == "trajectory",
        compact_flow_v3=bool(getattr(args, "compact_flow_v3", False)),
        branching=1,
        early_beam=1, late_beam=1, early_depth=2,
        max_decisions=40, max_imports=32,
        max_new_tokens=args.max_new_tokens, max_context=args.max_context,
        record_attempts=getattr(args, "record_attempts", False),
    )


def validate_benchmark_source(args: argparse.Namespace, source_sha256: str) -> str | None:
    """Do not label a filtered or sampled source as an official test view."""

    view = getattr(args, "benchmark_view", "diagnostic")
    manifest_path = getattr(args, "source_manifest", None)
    if view == "diagnostic":
        if manifest_path is not None:
            raise ValueError("diagnostic selection must not masquerade as a frozen benchmark view")
        return None
    if view not in {"strict_test", "full_endpoint_test"} or manifest_path is None:
        raise ValueError("formal benchmark view requires a frozen source manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = dict((manifest.get("splits") or {}).get("test") or {})
    if view == "strict_test":
        expected_rows = 28967
        relative = split.get("file")
        declared_sha256 = split.get("sha256")
        if (
            manifest.get("strict_trace_universe_complete") is not True
            or manifest.get("artifact_type") != "flower_strict_action_delta_trace_owned_tool_sft"
            or int(split.get("unique_ids", -1)) != expected_rows
            or int((manifest.get("official_reaction_denominators") or {}).get("test", -1)) != 28971
            or int((manifest.get("named_upstream_corrupt_rows_excluded") or {}).get("test", -1)) != 4
        ):
            raise ValueError("strict test manifest does not declare the frozen executable universe")
    else:
        expected_rows = 28971
        relative = split.get("output")
        declared_sha256 = split.get("output_sha256")
        if int(split.get("expected_rows", -1)) != expected_rows or split.get("coverage") != 1:
            raise ValueError("full endpoint manifest does not declare complete coverage")
    if int(split.get("rows", -1)) != expected_rows or args.sample_reactions != expected_rows:
        raise ValueError(f"{view} must score its complete {expected_rows}-reaction denominator")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("benchmark manifest test source path is not repository-relative")
    declared_path = (manifest_path.resolve().parents[2] / relative).resolve()
    if args.data.resolve() != declared_path or source_sha256 != declared_sha256:
        raise ValueError("benchmark test source path/SHA disagrees with frozen manifest")
    return sha256(manifest_path)


def run_fingerprint(args: argparse.Namespace) -> str:
    provisional = getattr(args, "provisional_training_config", None)
    contract = {
        "artifact_type": "reliable_mechet_independent_episodes_v1",
        "source_sha256": sha256(args.data),
        "benchmark_view": getattr(args, "benchmark_view", "diagnostic"),
        "source_manifest_sha256": (
            sha256(args.source_manifest) if getattr(args, "source_manifest", None) else None
        ),
        "adapter_model_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "adapter_manifest_sha256": (
            None if provisional else sha256(args.adapter / "adapter_manifest.json")
        ),
        "provisional_training_config_sha256": sha256(provisional) if provisional else None,
        "provisional_trainer_state_sha256": (
            sha256(args.adapter / "trainer_state.json") if provisional else None
        ),
        "provisional_adapter_config_sha256": (
            sha256(args.adapter / "adapter_config.json") if provisional else None
        ),
        "stage": args.stage, "model": MODEL, "revision": REVISION,
        "episodes": args.episodes, "sample_reactions": args.sample_reactions,
        "diagnostic_selection": getattr(args, "diagnostic_selection", "hash"),
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "max_context": args.max_context, "no_4bit": args.no_4bit,
        "dtype": getattr(args, "dtype", "bfloat16"),
        "record_attempts": getattr(args, "record_attempts", False),
        "ranking_score": "raw_model_mean_generated_token_logprob",
        "runtime_versions": runtime_versions(),
        "runtime_sha256": sha256(ROOT / "scripts/run_natural_language_value_search.py"),
        "evaluator_sha256": sha256(Path(__file__)),
        "evaluation_dependency_sha256": evaluation_dependency_hashes(
            compact_flow_v3=bool(getattr(args, "compact_flow_v3", False))
        ),
    }
    if getattr(args, "compact_flow_v3", False):
        contract["action_contract"] = "compact_electron_flow_v3"
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_evaluation_reactions(
    path: Path, size: int, seed: int, *, mode: str, view: str,
) -> list[dict[str, Any]]:
    if mode == "hash":
        return read_selected(path, size, seed)
    if mode != "stratified" or view != "diagnostic":
        raise ValueError("stratified selection is diagnostic-only")
    with path.open(encoding="utf-8") as handle:
        return stratified_sample(
            (json.loads(line) for line in handle if line.strip()),
            size=size, seed=seed,
        )


def checked_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if args.episodes not in (1, 5, 10) or args.sample_reactions < 1:
        raise ValueError("episodes must be 1, 5, or 10 and sample size positive")
    if args.max_new_tokens < 1 or args.max_context < 1:
        raise ValueError("invalid token budget")
    source_sha256 = sha256(args.data)
    if source_sha256 != args.expected_source_sha256:
        raise ValueError("evaluation source SHA-256 mismatch")
    validate_benchmark_source(args, source_sha256)
    if sha256(args.adapter / "adapter_model.safetensors") != args.expected_adapter_sha256:
        raise ValueError("evaluation adapter SHA-256 mismatch")
    actor = policy_args(args)
    validate_matched_v2_args(actor)
    provisional = getattr(args, "provisional_training_config", None)
    if provisional:
        if getattr(args, "benchmark_view", "diagnostic") != "diagnostic":
            raise ValueError("an unfinished checkpoint cannot be a formal benchmark result")
        if args.sample_reactions > 16:
            raise ValueError("provisional checkpoint evaluation is limited to 16 reactions")
        lineage = validate_adapter_lineage(
            args.adapter, MODEL, REVISION, provisional_training_config=provisional,
        )
        if lineage["training_stage"] != ("trajectory_sft" if actor.compact_history else "state_sft"):
            raise ValueError("provisional checkpoint stage differs from evaluation stage")
    else:
        if getattr(args, "compact_flow_v3", False):
            if actor.compact_history:
                raise ValueError("compact v3 supports Stage-I state policy only")
            validate_compact_flow_v3_adapter_manifest(
                args.adapter, expected_model=MODEL, expected_revision=REVISION,
            )
        else:
            validate_v2_adapter_manifest(
                args.adapter, compact_history=actor.compact_history,
                expected_model=MODEL, expected_revision=REVISION,
            )
    rows = select_evaluation_reactions(
        args.data, args.sample_reactions, args.seed,
        mode=getattr(args, "diagnostic_selection", "hash"),
        view=getattr(args, "benchmark_view", "diagnostic"),
    )
    if len(rows) != args.sample_reactions or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("evaluation reaction selection is incomplete or duplicated")
    if any(not reference_structural_precursor(row) for row in rows):
        raise ValueError("evaluation row lacks a structural precursor label")
    return rows, run_fingerprint(args)


def sample_episode(
    runtime: Runtime, target_smiles: str, actor_args: argparse.Namespace,
    *, index: int, seed: int, stochastic: bool,
) -> dict[str, Any]:
    args = copy(actor_args)
    args.planning_sample = stochastic
    runtime.torch.manual_seed(seed)
    searched = search_unlabeled(runtime, target_smiles, args)
    top = searched["top"]
    structural = ""
    if top.terminal:
        projected = split_precursor_endpoints(
            top.state, searched["target_mapped"]
        ).structural
        structural = visible(projected) if projected else ""
    result = {
        "index": index, "seed": seed, "terminal": bool(top.terminal),
        "precursor": structural,
        "full_executor_precursor": visible(top.state) if top.terminal else "",
        "mean_token_logprob": float(top.policy_score) if top.terminal else None,
        "has_electron_event": any(
            item["name"] == "apply_electron_flow" for item in top.actions
        ),
        "actions": [
            {"name": item["name"], "arguments": item["arguments"]}
            for item in top.actions
        ],
        "proposal_outcomes": [
            {
                "name": str(item.get("name") or ""),
                "accepted": bool(item["accepted"]),
                "error": str(item.get("error") or ""),
                "mean_token_logprob": item.get("action_policy_score"),
            }
            for item in searched["attempts"]
        ],
        "rejected": searched["rejected"],
    }
    if getattr(actor_args, "record_attempts", False):
        result["attempts"] = searched["attempts"]
    return result


def sample_reaction(
    runtime: Runtime, row: Mapping[str, Any], actor_args: argparse.Namespace,
    *, episodes: int, seed: int,
) -> dict[str, Any]:
    target = str(row["target_smiles"])
    samples = [
        sample_episode(
            runtime, target, actor_args, index=index,
            seed=episode_seed(seed, target, index), stochastic=episodes > 1,
        )
        for index in range(episodes)
    ]
    return {
        "id": str(row["id"]), "source_id": str(row["source_id"]),
        "target": canonical_unmapped(target), "episodes": samples,
    }


def verify_episode_trace(target: str, episode: Mapping[str, Any]) -> None:
    """Independently replay saved accepted actions before endpoint scoring."""

    mapped = product_only_private_state(target)
    root_visible = visible(mapped)
    node = Node(
        target=root_visible, state=mapped,
        next_map=max(mapped_atom_numbers(mapped), default=0) + 1,
        visited={root_visible},
    )
    actions = episode.get("actions")
    if not isinstance(actions, list) or len(actions) > 40:
        raise ValueError("episode accepted-action list is invalid")
    for index, record in enumerate(actions):
        if node.terminal or not isinstance(record, Mapping):
            raise ValueError("episode contains an action after finish")
        name, arguments = record.get("name"), record.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise ValueError("episode accepted action has invalid schema")
        child, error = execute(
            node, Action(name, dict(arguments), "", 0.0, 1),
            max_imports=32, reject_target_retained_finish=True,
        )
        if child is None:
            raise ValueError(f"episode accepted action {index} failed independent replay: {error}")
        node = child
    if node.terminal != bool(episode.get("terminal")):
        raise ValueError("episode terminal flag disagrees with independent replay")
    has_event = any(record["name"] == "apply_electron_flow" for record in node.actions)
    if has_event != bool(episode.get("has_electron_event")):
        raise ValueError("episode electron-event flag disagrees with independent replay")
    if node.terminal:
        full = str(episode.get("full_executor_precursor") or "")
        if not full or visible(node.state) != visible(full):
            raise ValueError("episode full precursor disagrees with independent replay")
        structural = split_precursor_endpoints(node.state, mapped).structural
        projected = visible(structural) if structural else ""
        precursor = str(episode.get("precursor") or "")
        if projected != (visible(precursor) if precursor else ""):
            raise ValueError("episode structural precursor disagrees with independent replay")
    elif episode.get("full_executor_precursor") or episode.get("precursor"):
        raise ValueError("nonterminal episode declares a precursor")


def score_reaction(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Score only after generation; rank unique terminal candidates by mean NLL."""

    gold = reference_structural_precursor(row)
    episodes = list(prediction["episodes"])
    if str(prediction["source_id"]) != str(row["source_id"]):
        raise ValueError(f"{row['id']}: prediction source ID mismatch")
    if canonical_unmapped(str(prediction["target"])) != canonical_unmapped(str(row["target_smiles"])):
        raise ValueError(f"{row['id']}: prediction target mismatch")
    expected_count = len(episodes)
    if [int(item["index"]) for item in episodes] != list(range(expected_count)):
        raise ValueError(f"{row['id']}: episode indices are incomplete")
    candidates = []
    for item in episodes:
        index = int(item["index"])
        if int(item["seed"]) != episode_seed(int(prediction["seed"]), str(row["target_smiles"]), index):
            raise ValueError(f"{row['id']}: episode seed mismatch")
        terminal = bool(item["terminal"])
        precursor = str(item.get("precursor") or "")
        score = item.get("mean_token_logprob")
        if not terminal:
            if precursor or score is not None:
                raise ValueError(f"{row['id']}: nonterminal episode declared a candidate")
            continue
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f"{row['id']}: terminal candidate lacks finite NLL score")
        if not precursor:
            continue
        candidates.append({
            "index": index, "precursor": canonical_unmapped(precursor),
            "score": float(score), "has_electron_event": bool(item["has_electron_event"]),
            "hit": structural_exact(precursor, gold),
        })
    ranked = []
    seen = set()
    for item in sorted(candidates, key=lambda value: (-value["score"], value["index"])):
        if item["precursor"] not in seen:
            seen.add(item["precursor"])
            ranked.append(item)
    result = {
        "id": str(row["id"]), "sampled_episodes": expected_count,
        "terminal_episodes": sum(bool(item["terminal"]) for item in episodes),
        "nonterminal_episodes": sum(not bool(item["terminal"]) for item in episodes),
        "terminal_reference_mismatch_episodes": sum(
            bool(item["terminal"]) for item in episodes
        ) - sum(item["hit"] for item in candidates),
        "unique_terminal_candidates": len(ranked),
        "nll_ranked_candidates": ranked[:10],
    }
    for k in (1, 5, 10):
        result[f"generation_pass_at_{k}"] = (
            any(item["hit"] and item["index"] < k for item in candidates)
            if k <= expected_count else None
        )
        result[f"nll_ranked_top_{k}"] = (
            any(item["hit"] for item in ranked[:k])
            if k <= expected_count else None
        )
        result[f"process_reliable_pass_at_{k}"] = (
            any(item["hit"] and item["has_electron_event"] and item["index"] < k
                for item in candidates)
            if k <= expected_count else None
        )
    return result


def endpoint_risk_coverage(
    cases: list[Mapping[str, Any]], denominator: int,
) -> dict[str, Any]:
    """Selective endpoint error among NLL-ranked top-1 terminal predictions."""

    eligible = [
        (float(case["nll_ranked_candidates"][0]["score"]),
         bool(case["nll_ranked_candidates"][0]["hit"]), str(case["id"]))
        for case in cases if case["nll_ranked_candidates"]
    ]
    eligible.sort(key=lambda item: (-item[0], item[2]))
    curve = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, int(len(eligible) * fraction)) if eligible else 0
        selected = eligible[:count]
        errors = sum(not hit for _, hit, _ in selected)
        curve.append({
            "fraction_of_scored_terminals": fraction,
            "coverage_of_all_reactions": count / denominator,
            "n": count, "endpoint_misses": errors,
            "endpoint_miss_rate": errors / count if count else None,
            "min_mean_token_logprob": selected[-1][0] if selected else None,
        })
    return {
        "confidence_definition": "highest unwarped-policy mean token log-probability among unique terminal precursors",
        "scored_terminal_reactions": len(eligible),
        "abstaining_or_missing_reactions": denominator - len(eligible),
        "curve": curve,
        "interpretation": (
            "Endpoint miss means disagreement with the recorded structural precursor, "
            "not proof that an alternative executable reaction is chemically impossible."
        ),
    }


def proposal_rejection_risk_coverage(
    proposals: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """High-confidence executor-rejection proxy, not chemical hallucination."""

    scored = []
    unparseable = unscorable = 0
    for item in proposals:
        if not item.get("name"):
            unparseable += 1
            continue
        score = item.get("mean_token_logprob")
        if score is None or not math.isfinite(float(score)):
            unscorable += 1
            continue
        scored.append((float(score), not bool(item["accepted"])))
    scored.sort(key=lambda item: item[0], reverse=True)
    curve = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, int(len(scored) * fraction)) if scored else 0
        selected = scored[:count]
        rejected = sum(item[1] for item in selected)
        curve.append({
            "fraction_of_scored_parsed_proposals": fraction,
            "n": count, "executor_rejected": rejected,
            "executor_rejection_rate": rejected / count if count else None,
            "min_mean_token_logprob": selected[-1][0] if selected else None,
        })
    return {
        "confidence_definition": "unwarped-policy mean generated-token log-probability per parsed proposal",
        "observed_proposals": len(proposals),
        "scored_parsed_proposals": len(scored),
        "unparseable_proposals": unparseable,
        "unscorable_parsed_proposals": unscorable,
        "executor_rejected_scored_proposals": sum(item[1] for item in scored),
        "top_decile_executor_rejection": curve[0],
        "curve": curve,
        "interpretation": (
            "An executor-rejected proposal is an observed formal failure. "
            "This proxy does not label accepted alternatives as chemically valid "
            "or reference-different alternatives as hallucinations."
        ),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    selected, fingerprint = checked_inputs(args)
    expected = {str(row["id"]): row for row in selected}
    scored: dict[str, dict[str, Any]] = {}
    proposal_outcomes: list[dict[str, Any]] = []
    shards = sorted(args.output.glob("episodes.shard-*-of-*.jsonl"))
    if not shards:
        raise ValueError("no episode prediction shards")
    for shard in shards:
        with shard.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                prediction = json.loads(line)
                identifier = str(prediction["id"])
                if identifier not in expected or identifier in scored:
                    raise ValueError(f"unknown or duplicate prediction: {identifier}")
                if prediction.get("run_fingerprint") != fingerprint:
                    raise ValueError(f"{identifier}: episode run lineage mismatch")
                if int(prediction.get("seed", -1)) != args.seed or len(prediction["episodes"]) != args.episodes:
                    raise ValueError(f"{identifier}: episode budget mismatch")
                for episode in prediction["episodes"]:
                    verify_episode_trace(str(expected[identifier]["target_smiles"]), episode)
                    proposal_outcomes.extend(episode.get("proposal_outcomes") or [])
                scored[identifier] = score_reaction(expected[identifier], prediction)
    cases = [scored[identifier] for identifier in expected if identifier in scored]
    process_metrics_available = getattr(args, "benchmark_view", "diagnostic") != "full_endpoint_test"
    if not process_metrics_available:
        for case in cases:
            for k in (1, 5, 10):
                case[f"process_reliable_pass_at_{k}"] = None
    counts: dict[str, Any] = {}
    for k in (1, 5, 10):
        for prefix in ("generation_pass", "nll_ranked_top", "process_reliable_pass"):
            name = f"{prefix}_at_{k}" if prefix != "nll_ranked_top" else f"{prefix}_{k}"
            counts[name] = (
                sum(bool(case[name]) for case in cases)
                if k <= args.episodes and (prefix != "process_reliable_pass" or process_metrics_available)
                else None
            )
    return {
        "artifact_type": (
            "reliable_mechet_independent_episodes_provisional_checkpoint_audit_v1"
            if getattr(args, "provisional_training_config", None)
            else f"reliable_mechet_independent_episodes_{getattr(args, 'benchmark_view', 'diagnostic')}_audit_v1"
        ),
        "claim_boundary": (
            "Product-only independent executor episodes; structural reference used "
            "only after generation. Execution is not laboratory feasibility. "
            + (
                "Unfinished-checkpoint diagnostic on at most 16 reactions; "
                "not final validation or test accuracy."
                if getattr(args, "provisional_training_config", None) else ""
            )
        ),
        "process_reliable_definition": (
            "Structural endpoint hit by a terminal independently replayed trajectory "
            "with at least one accepted electron event; laboratory feasibility "
            "is a separate test."
        ) if process_metrics_available else None,
        "process_metric_status": (
            "available_on_strict_or_diagnostic_view"
            if process_metrics_available
            else "unavailable_on_full_endpoint_view_use_28967_strict_test"
        ),
        "model": MODEL, "model_revision": REVISION,
        "runtime_versions": runtime_versions(),
        "compute_dtype": getattr(args, "dtype", "bfloat16"),
        "adapter_lineage": (
            validate_adapter_lineage(
                args.adapter, MODEL, REVISION,
                provisional_training_config=args.provisional_training_config,
            ) if getattr(args, "provisional_training_config", None)
            else {"kind": "completed_adapter"}
        ),
        "stage": args.stage,
        "action_representation": (
            "compact_flow_v3" if getattr(args, "compact_flow_v3", False)
            else "natural_language_v2"
        ),
        "episodes_per_reaction": args.episodes,
        "benchmark_view": getattr(args, "benchmark_view", "diagnostic"),
        "diagnostic_selection": getattr(args, "diagnostic_selection", "hash"),
        "source_manifest_sha256": (
            sha256(args.source_manifest) if getattr(args, "source_manifest", None) else None
        ),
        "ranking": "descending mean generated-token log-probability, canonical unique terminal precursors",
        "ranking_score_source": "unwarped_adapter_policy_logits_before_temperature_top_p",
        "source_sha256": args.expected_source_sha256,
        "adapter_model_sha256": args.expected_adapter_sha256,
        "run_fingerprint": fingerprint,
        "denominator": args.sample_reactions,
        "observed_reactions": len(cases),
        "missing_reactions": args.sample_reactions - len(cases),
        "hits": counts,
        "rates": {
            key: value / args.sample_reactions if value is not None else None
            for key, value in counts.items()
        },
        "terminal_episodes": sum(case["terminal_episodes"] for case in cases),
        "nonterminal_episodes": sum(case["nonterminal_episodes"] for case in cases),
        "terminal_reference_mismatch_episodes": sum(
            case["terminal_reference_mismatch_episodes"] for case in cases
        ),
        "unique_terminal_candidates": sum(case["unique_terminal_candidates"] for case in cases),
        "endpoint_risk_coverage": endpoint_risk_coverage(cases, args.sample_reactions),
        "proposal_executor_rejection_risk_coverage": (
            proposal_rejection_risk_coverage(proposal_outcomes)
        ),
        "shards": [{"path": str(path), "sha256": sha256(path)} for path in shards],
        "cases": cases,
    }


def run(args: argparse.Namespace) -> None:
    rows, fingerprint = checked_inputs(args)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not 0 <= rank < world:
        raise ValueError("invalid rank/world size")
    selected = [row for index, row in enumerate(rows) if index % world == rank]
    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.output / f"episodes.shard-{rank:02d}-of-{world:02d}.jsonl"
    previous = []
    if shard.exists():
        with shard.open(encoding="utf-8") as stream:
            previous = [json.loads(line) for line in stream if line.strip()]
    if any(row.get("run_fingerprint") != fingerprint for row in previous):
        raise ValueError("refusing to resume an episode shard from another run")
    if any(int(row.get("seed", -1)) != args.seed or len(row.get("episodes") or []) != args.episodes
           for row in previous):
        raise ValueError("refusing to resume an incomplete episode record")
    completed = {str(row["id"]) for row in previous}
    if len(completed) != len(previous) or not completed <= {str(row["id"]) for row in selected}:
        raise ValueError("episode shard contains duplicate or foreign reactions")
    remaining = [row for row in selected if str(row["id"]) not in completed]
    if not remaining:
        print(f"[meteor-reliable-k] rank={rank} already complete", flush=True)
        return
    actor = policy_args(args)
    runtime = Runtime(actor, local_rank=local_rank)
    with shard.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(remaining, start=1):
            prediction = sample_reaction(
                runtime, row, actor, episodes=args.episodes, seed=args.seed,
            )
            prediction.update(seed=args.seed, run_fingerprint=fingerprint)
            sink.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[meteor-reliable-k] rank={rank} done={index}/{len(remaining)} "
                f"id={row['id']} episodes={args.episodes}", flush=True,
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("run", "aggregate"))
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--expected-source-sha256", required=True)
    result.add_argument("--benchmark-view", choices=(
        "diagnostic", "strict_test", "full_endpoint_test",
    ), default="diagnostic")
    result.add_argument("--source-manifest", type=Path,
                        help="Mandatory for formal strict/full test views")
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--expected-adapter-sha256", required=True)
    result.add_argument("--stage", choices=("state", "trajectory"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--episodes", type=int, choices=(1, 5, 10), default=10)
    result.add_argument("--sample-reactions", type=int, required=True)
    result.add_argument(
        "--diagnostic-selection", choices=("hash", "stratified"), default="hash",
        help="Use the local evaluator's length-stratified IDs only in diagnostic view",
    )
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--max-new-tokens", type=int, default=512)
    result.add_argument("--max-context", type=int, default=4096)
    result.add_argument("--no-4bit", action="store_true")
    result.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    result.add_argument("--provisional-training-config", type=Path,
                        help="Allow an unfinished trainer checkpoint for <=16 diagnostic reactions")
    result.add_argument("--record-attempts", action="store_true",
                        help="Store verbose failed-action/state attempts (large on full test)")
    result.add_argument("--compact-flow-v3", action="store_true",
                        help="Use compact event schema and separately validated v3 Stage-I adapter")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        run(args)
        return 0
    report = aggregate(args)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "independent_episodes_audit.json"
    cases_path = args.output / "independent_episodes_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as sink:
        for case in report["cases"]:
            sink.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary = {key: value for key, value in report.items() if key != "cases"}
    summary["cases_path"] = str(cases_path)
    summary["cases_sha256"] = sha256(cases_path)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
