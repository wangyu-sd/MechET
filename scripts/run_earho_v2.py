#!/usr/bin/env python3
"""Stage-III EARHO for a frozen v2 compressed-history trajectory policy.

The collector performs product-start first-divergence discovery, executor reset,
same-state branching and bounded continuation.  This driver updates the actor
and a separate successor-value adapter without loading the test split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.earho_v2_protocol import promote_productive_horizon, replay_reference
from scripts.run_anchor_branch_rl import log, read_rows, run_train, write_json, write_rows
from scripts.run_natural_language_anchor_branch_rl import run_workers, _sha256
from scripts.run_natural_language_value_search import read_selected
from scripts.train_python_template_rlvr import _load_yaml


PROTOCOL = "trajectory_history_v2"
RELIABLE_PROTOCOL = "reliable_mechet_three_stage_v1"
STATE_DIRECT_PROTOCOL = "state_direct_v2"


def resolve_reliable_paths(cfg: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    """Keep the PR code worktree separate from frozen Ceph data and outputs."""

    resolved = dict(cfg)
    for key in (
        "initial_adapter_path", "train_file", "validation_file",
        "stable_id_manifest", "history_file", "history_validation_file",
        "natural_language_manifest", "output_dir",
    ):
        value = str(resolved.get(key) or "")
        if not value:
            raise ValueError(f"reliable EARHO path is missing: {key}")
        path = Path(value)
        resolved[key] = str(path if path.is_absolute() else artifact_root / path)
    return resolved


def validate_reliable_contract(cfg: dict[str, Any]) -> None:
    """Bind lightweight FlowER EARHO to frozen strict data and Stage-II weights."""

    if cfg.get("protocol_version") != RELIABLE_PROTOCOL:
        raise ValueError("wrong reliable MechET protocol")
    if cfg.get("legacy_dual_prompt") or cfg.get("test_file"):
        raise ValueError("reliable EARHO forbids dual prompts and test data")
    if not (cfg.get("optimization") or {}).get("success_gated_advantages"):
        raise ValueError("reliable EARHO requires success-gated advantages")
    if (cfg.get("reward") or {}).get("endpoint_metric") != "structural":
        raise ValueError("reliable EARHO requires structural endpoint reward")
    rollout = dict(cfg.get("rollout") or {})
    if int(rollout.get("max_decisions", 0)) != 40 or int(rollout.get("max_imports", 0)) != 32:
        raise ValueError("reliable EARHO requires the frozen 40-decision/32-import budget")
    if cfg.get("value_adapter_path") or cfg.get("value_kind") != "successor_pn":
        raise ValueError("reliable EARHO must learn a new successor P/N critic")
    if cfg.get("model_name_or_path") != "Qwen/Qwen3-0.6B":
        raise ValueError("reliable EARHO base model differs from frozen 0.6B policy")
    revision = str(cfg.get("model_revision") or "")
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("reliable EARHO requires a pinned immutable model revision")
    if Path(str(cfg.get("model_snapshot") or "")).name != revision:
        raise ValueError("reliable EARHO model snapshot does not match pinned revision")
    pinned_sha = str(cfg.get("initial_adapter_model_sha256") or "")
    if len(pinned_sha) != 64 or any(ch not in "0123456789abcdef" for ch in pinned_sha):
        raise ValueError("freeze the Stage-II adapter SHA-256 before EARHO")
    expected = {"train": 257167, "valid": 2890, "test": 28967}
    if dict(cfg.get("reaction_denominator") or {}) != expected:
        raise ValueError("reliable EARHO strict reaction denominator changed")
    if dict(cfg.get("full_reaction_denominator") or {}) != {
        "train": 257171, "valid": 2890, "test": 28971,
    }:
        raise ValueError("reliable EARHO full endpoint denominator changed")

    source_dir = Path(cfg["train_file"]).parent
    source_manifest_path = Path(cfg["stable_id_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text())
    source_status = json.loads((source_dir / "ARTIFACT_STATUS.json").read_text())
    if source_status.get("training_allowed") is not True:
        raise ValueError("strict FlowER source is not training-enabled")
    if source_manifest.get("artifact_type") != "flower_strict_action_delta_trace_owned_tool_sft":
        raise ValueError("unexpected strict FlowER source artifact")
    for split, key in (("train", "train_file"), ("valid", "validation_file")):
        item = source_manifest["splits"][split]
        if int(item["rows"]) != expected[split]:
            raise ValueError(f"{split} strict FlowER row count changed")
        if _sha256(Path(cfg[key])) != item["sha256"]:
            raise ValueError(f"{split} strict FlowER SHA-256 mismatch")
    if int(source_manifest["splits"]["test"]["rows"]) != expected["test"]:
        raise ValueError("strict FlowER test denominator changed")

    history_dir = Path(cfg["history_file"]).parent
    history_manifest = json.loads(Path(cfg["natural_language_manifest"]).read_text())
    if not history_manifest.get("training_allowed") or history_manifest.get("status") != "validated_complete":
        raise ValueError("FlowER compact-history supervision is not validated")
    if history_manifest.get("reaction_denominator") != expected:
        raise ValueError("FlowER history reaction denominator changed")
    if history_manifest.get("decision_contract") != "unified_inventory_compressed_history_tool_decision_v2":
        raise ValueError("FlowER compact-history decision contract changed")
    if history_manifest.get("history_contract") != "executor_compact_accepted_actions_v1":
        raise ValueError("FlowER accepted-history contract changed")
    event_dir = Path(str(history_manifest["source_artifact"]))
    event_manifest_path = event_dir / "manifest.json"
    event_manifest = json.loads(event_manifest_path.read_text())
    if event_manifest.get("source_manifest_sha256") != _sha256(source_manifest_path):
        raise ValueError("FlowER event/source manifest lineage mismatch")
    if Path(str(event_manifest.get("source_artifact") or "")).resolve() != source_dir.resolve():
        raise ValueError("FlowER event source directory changed")
    if event_manifest.get("reaction_denominator") != expected:
        raise ValueError("FlowER event reaction denominator changed")
    for split in ("train", "valid"):
        source_item = source_manifest["splits"][split]
        event_item = event_manifest["splits"][split]
        history_item = history_manifest["splits"][split]
        if event_item["source_sha256"] != source_item["sha256"]:
            raise ValueError(f"{split} FlowER event/source file lineage mismatch")
        if history_item["source_sha256"] != event_item["output_sha256"]:
            raise ValueError(f"{split} FlowER history/event file lineage mismatch")
        if int(history_item["reactions"]) != expected[split]:
            raise ValueError(f"{split} FlowER history reaction count changed")
        if _sha256(history_dir / f"{split}.jsonl") != history_item["output_sha256"]:
            raise ValueError(f"{split} FlowER history SHA-256 mismatch")

    adapter = Path(cfg["initial_adapter_path"])
    if _sha256(adapter / "adapter_model.safetensors") != pinned_sha:
        raise ValueError("FlowER Stage-II adapter SHA-256 mismatch")
    adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text())
    if adapter_manifest.get("base_model") != cfg["model_name_or_path"]:
        raise ValueError("FlowER Stage-II adapter base model mismatch")
    if adapter_manifest.get("base_model_revision") != revision:
        raise ValueError("FlowER Stage-II adapter revision mismatch")
    if adapter_manifest.get("environment_revision") != "natural_language_electron_event_history_v2":
        raise ValueError("FlowER Stage-II adapter observation contract mismatch")
    if adapter_manifest.get("train_file_sha256") != history_manifest["splits"]["train"]["output_sha256"]:
        raise ValueError("FlowER Stage-II adapter was trained on different history data")


def validate_contract(cfg: dict[str, Any]) -> None:
    if cfg.get("protocol_version") == RELIABLE_PROTOCOL:
        validate_reliable_contract(cfg)
        return
    state_direct = cfg.get("protocol_version") == STATE_DIRECT_PROTOCOL
    if cfg.get("protocol_version") not in {PROTOCOL, STATE_DIRECT_PROTOCOL}:
        raise ValueError("EARHO v2 requires the compressed-history v2 protocol")
    if cfg.get("legacy_dual_prompt") or cfg.get("test_file"):
        raise ValueError("v1 dual prompts and test data are forbidden")
    if not (cfg.get("optimization") or {}).get("success_gated_advantages"):
        raise ValueError("EARHO requires success-gated policy advantages")
    if str(cfg.get("value_kind") or "successor_pn") != "successor_pn":
        raise ValueError("EARHO requires a transition-level P/N successor critic")
    source_dir = Path(cfg["train_file"]).parent
    source_manifest = json.loads(Path(cfg["stable_id_manifest"]).read_text())
    source_status = json.loads((source_dir / "ARTIFACT_STATUS.json").read_text())
    if source_status.get("training_allowed") is not True:
        raise ValueError("source executable trace view is not training-enabled")
    expected = dict(cfg["reaction_denominator"])
    full = dict(cfg["full_reaction_denominator"])
    if expected != {"train": 10152, "valid": 1319, "test": 1253}:
        raise ValueError("unexpected current-compiler 31k executable denominator")
    if full != {"train": 24959, "valid": 3120, "test": 3120}:
        raise ValueError("unexpected complete 31k reaction denominator")
    for split, key in (("train", "train_file"), ("valid", "validation_file")):
        item = source_manifest["splits"][split]
        if int(item["rows"]) != expected[split]:
            raise ValueError(f"{split} source row count changed")
        if _sha256(Path(cfg[key])) != item["sha256"]:
            raise ValueError(f"{split} source SHA-256 mismatch")
    history_dir = Path(cfg["history_file"]).parent
    history_manifest = json.loads(Path(cfg["natural_language_manifest"]).read_text())
    if not history_manifest.get("training_allowed") or history_manifest.get("status") != "validated_trace_view":
        raise ValueError("v2 history supervision is not validated")
    expected_decision_contract = (
        "unified_inventory_tool_decision_v2" if state_direct
        else "unified_inventory_compressed_history_tool_decision_v2"
    )
    if history_manifest.get("decision_contract") != expected_decision_contract:
        raise ValueError("v2 observation decision contract mismatch")
    if history_manifest.get("reaction_denominator") != expected:
        raise ValueError("history executable denominator changed")
    if history_manifest.get("full_reaction_denominator") != full:
        raise ValueError("history complete denominator changed")
    repository_root = source_dir.parents[1]
    event_manifest_path = (
        Path(cfg["natural_language_manifest"])
        if state_direct else repository_root / str(history_manifest["source_artifact"]) / "manifest.json"
    )
    event_manifest = json.loads(event_manifest_path.read_text())
    if event_manifest.get("source_artifact") != str(source_dir.relative_to(repository_root)):
        raise ValueError("event supervision does not descend from selected source")
    if event_manifest.get("source_manifest_sha256") != _sha256(Path(cfg["stable_id_manifest"])):
        raise ValueError("event/source manifest lineage mismatch")
    if not state_direct and history_manifest.get("source_manifest_sha256") != _sha256(event_manifest_path):
        raise ValueError("history/event manifest lineage mismatch")
    for split in ("train", "valid"):
        if _sha256(history_dir / f"{split}.jsonl") != history_manifest["splits"][split]["output_sha256"]:
            raise ValueError(f"{split} history SHA-256 mismatch")
        if event_manifest["splits"][split]["source_sha256"] != source_manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} event/source file lineage mismatch")
        if not state_direct and history_manifest["splits"][split]["source_sha256"] != event_manifest["splits"][split]["output_sha256"]:
            raise ValueError(f"{split} history/event file lineage mismatch")
    adapter = Path(cfg["initial_adapter_path"])
    if _sha256(adapter / "adapter_model.safetensors") != cfg["initial_adapter_model_sha256"]:
        raise ValueError("Stage-II parent adapter SHA-256 mismatch")
    adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text())
    expected_environment_revision = (
        "natural_language_electron_event_v2" if state_direct
        else "natural_language_electron_event_history_v2"
    )
    if adapter_manifest.get("environment_revision") != expected_environment_revision:
        raise ValueError("parent observation protocol mismatch")
    if adapter_manifest.get("base_model_revision") != cfg["model_revision"]:
        raise ValueError("parent/base model revision mismatch")
    if adapter_manifest.get("train_file_sha256") != history_manifest["splits"]["train"]["output_sha256"]:
        raise ValueError("parent was not trained on the selected 31k observation data")
    if cfg.get("value_adapter_path"):
        raise ValueError("v2 successor value must start untrained and be learned from actor rollouts")


def _attach_decisions(
    rows: list[dict[str, Any]], history_file: Path, *, max_imports: int,
    compact_history: bool = True,
) -> list[dict[str, Any]]:
    wanted = {str(row["source_id"]) for row in rows}
    if len(wanted) != len(rows):
        raise ValueError("source reaction IDs are not unique")
    found: dict[str, list[dict[str, Any]]] = {key: [] for key in wanted}
    with history_file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                decision = json.loads(line)
                key = str(decision.get("source_id") or "")
                if key in found:
                    found[key].append(decision)
    output = []
    for row in rows:
        key = str(row["source_id"])
        if not found[key]:
            raise ValueError(f"missing v2 decisions: {key}")
        packed = dict(
            row, earho_v2_reference_decisions=found[key],
            earho_v2_compact_history=compact_history,
        )
        replay_reference(
            packed, found[key], max_imports=max_imports,
            compact_history=compact_history,
        )
        output.append(packed)
    return output


def _prepare_full_round(
    index: int, staging_name: str, output_name: str, max_imports: int,
) -> tuple[int, str, list[str], int]:
    """Replay one disjoint source/history partition in a worker process."""

    staging = Path(staging_name)
    output = Path(output_name)
    base = read_rows(staging / f"source_{index:03d}.jsonl")
    selected = _attach_decisions(
        base, staging / f"history_{index:03d}.jsonl",
        max_imports=max_imports, compact_history=True,
    )
    target = output / f"round{index:02d}/source.jsonl"
    write_rows(target, selected)
    return index, _sha256(target), [str(row["source_id"]) for row in selected], len(selected)


def _prepare_full_round_from_args(
    args: tuple[int, str, str, int],
) -> tuple[int, str, list[str], int]:
    return _prepare_full_round(*args)


def prepare(cfg: dict[str, Any], output: Path) -> None:
    config_sha256 = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (output / "plan.json").exists():
        plan = json.loads((output / "plan.json").read_text())
        if plan.get("config_sha256") != config_sha256:
            raise ValueError("existing EARHO plan was prepared with a different config")
        if plan.get("initial_adapter_model_sha256") != cfg["initial_adapter_model_sha256"]:
            raise ValueError("existing EARHO plan has a different parent adapter")
        if plan.get("protocol_version") != cfg["protocol_version"]:
            raise ValueError("existing EARHO plan has a different protocol")
        if cfg["protocol_version"] == RELIABLE_PROTOCOL and (
            plan.get("private_product_mapping_basis") != "source_original_mapped_product"
            or plan.get("product_only_private_remap") is not False
        ):
            raise ValueError("existing reliable EARHO plan lacks private-map provenance")
        for name, digest in (plan.get("prepared_files") or {}).items():
            if _sha256(output / name) != digest:
                raise ValueError(f"prepared EARHO source changed: {name}")
        if len(plan.get("prepared_files") or {}) != int(cfg["rounds"]) + 1:
            raise ValueError("EARHO preparation manifest is incomplete")
        return
    if cfg.get("coverage_mode") == "all_executable":
        _prepare_all_executable(cfg, output, config_sha256)
        return
    count = int(cfg["rounds"]) * int(cfg["products_per_round"])
    if count > int(cfg["reaction_denominator"]["train"]):
        raise ValueError("requested more distinct RL train reactions than available")
    if cfg["protocol_version"] == RELIABLE_PROTOCOL:
        # The FlowER source has 257k large trace rows. Select deterministically
        # while streaming; never materialize the complete reaction corpus.
        source = read_selected(Path(cfg["train_file"]), count, int(cfg["seed"]))
        validation = read_selected(
            Path(cfg["validation_file"]),
            int(cfg["validation_monitor_rows"]), int(cfg["seed"]),
        )
        source_reactions = int(cfg["reaction_denominator"]["train"])
    else:
        source = read_rows(cfg["train_file"])
        validation = read_rows(cfg["validation_file"])
        if len(source) != cfg["reaction_denominator"]["train"]:
            raise ValueError("source train count mismatch")
        if len(validation) != cfg["reaction_denominator"]["valid"]:
            raise ValueError("source validation count mismatch")
        random.Random(int(cfg["seed"])).shuffle(source)
        random.Random(int(cfg["seed"])).shuffle(validation)
        source_reactions = len(source)
    max_imports = int(cfg["rollout"]["max_imports"])
    selected = _attach_decisions(
        source[:count], Path(cfg["history_file"]), max_imports=max_imports,
        compact_history=cfg["protocol_version"] != STATE_DIRECT_PROTOCOL,
    )
    monitor = _attach_decisions(
        validation[: int(cfg["validation_monitor_rows"])],
        Path(cfg["history_validation_file"]), max_imports=max_imports,
        compact_history=cfg["protocol_version"] != STATE_DIRECT_PROTOCOL,
    )
    output.mkdir(parents=True, exist_ok=True)
    for round_index in range(int(cfg["rounds"])):
        start = round_index * int(cfg["products_per_round"])
        stop = start + int(cfg["products_per_round"])
        write_rows(output / f"round{round_index:02d}/source.jsonl", selected[start:stop])
    write_rows(output / "validation_monitor.jsonl", monitor)
    prepared_files = {
        f"round{index:02d}/source.jsonl": _sha256(output / f"round{index:02d}/source.jsonl")
        for index in range(int(cfg["rounds"]))
    }
    prepared_files["validation_monitor.jsonl"] = _sha256(output / "validation_monitor.jsonl")
    ids = [str(row["source_id"]) for row in selected]
    write_json(
        output / "plan.json",
        {
            "artifact_type": "earho_first_divergence_v2_plan",
            "protocol_version": cfg["protocol_version"],
            "config_sha256": config_sha256,
            "source_reactions": source_reactions,
            "selected_train_reactions": count,
            "validation_monitor_reactions": len(monitor),
            "selected_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "prepared_files": prepared_files,
            "initial_adapter_model_sha256": cfg["initial_adapter_model_sha256"],
            "reference_endpoint_model_visible": False,
            "first_divergence_from_product_rollout": True,
            **({
                "private_product_mapping_basis": "source_original_mapped_product",
                "product_only_private_remap": False,
            } if cfg["protocol_version"] == RELIABLE_PROTOCOL else {}),
            "actor_prompt_history_contract": (
                "none_state_only_v2" if cfg["protocol_version"] == STATE_DIRECT_PROTOCOL
                else "executor_compact_accepted_actions_v1"
            ),
            "test_used": False,
        },
    )
    log(stage="earho-v2-prepare", train=count, validation=len(monitor))


def _prepare_all_executable(cfg: dict[str, Any], output: Path, config_sha256: str) -> None:
    """Partition every strict train reaction once without loading all histories."""

    if cfg["protocol_version"] != RELIABLE_PROTOCOL:
        raise ValueError("all-executable preparation requires reliable FlowER")
    count = int(cfg["reaction_denominator"]["train"])
    rounds = int(cfg["rounds"])
    if rounds < 1:
        raise ValueError("all-executable preparation requires at least one round")
    marker = output / "preparation_config.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get("config_sha256") != config_sha256:
            raise ValueError("unplanned EARHO output is nonempty; refusing to overwrite it")
    output.mkdir(parents=True, exist_ok=True)
    write_json(marker, {"config_sha256": config_sha256, "coverage_mode": "all_executable"})
    source_round: dict[str, int] = {}
    round_counts = [0] * rounds
    with tempfile.TemporaryDirectory(prefix="earho_full_prepare_", dir=output) as staging_name:
        staging = Path(staging_name)
        with ExitStack() as handles:
            source_handles = [
                handles.enter_context((staging / f"source_{i:03d}.jsonl").open("w", encoding="utf-8"))
                for i in range(rounds)
            ]
            with Path(cfg["train_file"]).open(encoding="utf-8") as source_file:
                for line in source_file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identifier = str(row["source_id"])
                    if identifier in source_round:
                        raise ValueError(f"duplicate strict FlowER source_id: {identifier}")
                    bucket = int.from_bytes(
                        hashlib.sha256(f'{cfg["seed"]}:{identifier}'.encode()).digest()[:8], "big"
                    ) % rounds
                    source_round[identifier] = bucket
                    round_counts[bucket] += 1
                    source_handles[bucket].write(line)
        if len(source_round) != count or any(size == 0 for size in round_counts):
            raise ValueError(f"strict FlowER train coverage mismatch: {len(source_round)} != {count}")
        log(stage="earho-full-source-partitioned", train=count, rounds=rounds,
            smallest_round=min(round_counts), largest_round=max(round_counts))

        with ExitStack() as handles:
            history_handles = [
                handles.enter_context((staging / f"history_{i:03d}.jsonl").open("w", encoding="utf-8"))
                for i in range(rounds)
            ]
            with Path(cfg["history_file"]).open(encoding="utf-8") as history_file:
                for line in history_file:
                    if not line.strip():
                        continue
                    decision = json.loads(line)
                    identifier = str(decision.get("source_id") or "")
                    if identifier not in source_round:
                        raise ValueError(f"history has unknown strict FlowER source_id: {identifier}")
                    history_handles[source_round[identifier]].write(line)

        prepared_files: dict[str, str] = {}
        selected_ids = hashlib.sha256()
        workers = int(os.environ.get("MECHET_EARHO_PREP_WORKERS", "1"))
        if not 1 <= workers <= rounds:
            raise ValueError("MECHET_EARHO_PREP_WORKERS must be in [1, rounds]")
        log(stage="earho-full-replay-start", rounds=rounds, workers=workers)
        arguments = (
            (index, str(staging), str(output), int(cfg["rollout"]["max_imports"]))
            for index in range(rounds)
        )
        if workers == 1:
            results = (_prepare_full_round(*args) for args in arguments)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            results = pool.map(_prepare_full_round_from_args, arguments)
        try:
            for index, digest, identifiers, size in results:
                name = f"round{index:02d}/source.jsonl"
                prepared_files[name] = digest
                for identifier in identifiers:
                    selected_ids.update(identifier.encode() + b"\n")
                log(stage="earho-full-round-prepared", round=index, train=size)
        finally:
            if workers > 1:
                pool.shutdown(wait=True, cancel_futures=True)

    validation = read_selected(
        Path(cfg["validation_file"]), int(cfg["validation_monitor_rows"]), int(cfg["seed"]),
    )
    monitor = _attach_decisions(
        validation, Path(cfg["history_validation_file"]),
        max_imports=int(cfg["rollout"]["max_imports"]), compact_history=True,
    )
    write_rows(output / "validation_monitor.jsonl", monitor)
    prepared_files["validation_monitor.jsonl"] = _sha256(output / "validation_monitor.jsonl")
    write_json(output / "plan.json", {
        "artifact_type": "earho_first_divergence_v2_plan",
        "protocol_version": cfg["protocol_version"],
        "coverage_mode": "all_executable",
        "config_sha256": config_sha256,
        "source_reactions": count,
        "selected_train_reactions": count,
        "round_counts": round_counts,
        "validation_monitor_reactions": len(monitor),
        "selected_id_sha256": selected_ids.hexdigest(),
        "prepared_files": prepared_files,
        "initial_adapter_model_sha256": cfg["initial_adapter_model_sha256"],
        "reference_endpoint_model_visible": False,
        "first_divergence_from_product_rollout": True,
        "private_product_mapping_basis": "source_original_mapped_product",
        "product_only_private_remap": False,
        "actor_prompt_history_contract": "executor_compact_accepted_actions_v1",
        "test_used": False,
    })
    log(stage="earho-v2-prepare", train=count, validation=len(monitor), rounds=rounds,
        coverage_mode="all_executable")


def _critic_config(
    cfg: dict[str, Any], dataset: Path, output: Path, initial_adapter: Path
) -> Path:
    import yaml

    template = ROOT / "configs/agent/natural_language_state_value_qwen3_8b_h20.yaml"
    critic = yaml.safe_load(template.read_text())
    manifest = json.loads((dataset / "manifest.json").read_text())
    reliable = cfg["protocol_version"] == RELIABLE_PROTOCOL
    lightweight = reliable or cfg["protocol_version"] == STATE_DIRECT_PROTOCOL
    critic.update(
        condition_name=(
            "reliable_mechet_successor_value_pn"
            if reliable else "earho_v2_successor_value_pn"
        ),
        scientific_hypothesis="actor_successor_value_improves_executable_continuation",
        initial_adapter_path=str(initial_adapter),
        train_file=str(dataset / "train.jsonl"),
        validation_file=str(dataset / "valid.jsonl"),
        pretokenized_cache_dir=str(dataset / "tokens_1024"),
        output_dir=str(output),
    )
    if lightweight:
        critic["model_name_or_path"] = cfg["model_name_or_path"]
        training = critic["training"]
        training.update(
            model_revision=cfg["model_revision"], qlora=False,
            require_flash_sdp=False, use_liger_kernel=False,
            liger_kernel_config={}, per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
        )
        if (cfg.get("rollout") or {}).get("dtype") == "float16":
            training.update(bf16=False, fp16=True, tf32=False)
    contract = critic["contract"]
    contract.update(
        paper_baseline_id="EARHO-successor-value-v2",
        paper_method_name="mechet_earho_v2_transition_value",
        paper_run_role="learned_successor_value",
        stable_id_manifest=str(dataset / "manifest.json"),
        validation_report=str(dataset / "manifest.json"),
        expected_train_rows=manifest["splits"]["train"]["rows"],
        expected_validation_rows=manifest["splits"]["valid"]["rows"],
        expected_test_rows=0,
        source_dataset=(
            "flower_new_dataset_strict_proof_universe_v4" if reliable
            else "mech_uspto_31k_current_compiler_executable_trace_view"
        ),
        source_artifact=cfg["train_file"],
        reaction_denominator=cfg["reaction_denominator"],
        environment_revision="earho_v2_successor_value_pn",
        executor_revision=(
            "MECH_PROOF_v1_full_coverage_v4" if reliable
            else "mech_uspto31k_current_compiler_20260824"
        ),
    )
    path = dataset / "critic_training.yaml"
    path.write_text(yaml.safe_dump(critic, sort_keys=False))
    return path


def train_successor_critic(
    cfg: dict[str, Any], source: Path, shards: list[Path],
    round_path: Path, initial_adapter: Path,
) -> Path:
    dataset = round_path / "successor_value"
    if not (dataset / "manifest.json").is_file():
        command = [
            sys.executable, "scripts/build_earho_v2_successor_value.py",
            "--source", str(source), "--output-dir", str(dataset),
            "--max-hard-negatives", str((cfg.get("optimization") or {}).get("max_hard_negatives", 4)),
        ]
        if cfg["protocol_version"] == STATE_DIRECT_PROTOCOL:
            command.append("--state-only-observation")
        for shard in shards:
            command.extend(["--rollout", str(shard)])
        subprocess.run(command, check=True)
    report = json.loads((dataset / "manifest.json").read_text())
    if report["statistics"].get("hard_negatives", 0) == 0:
        raise ValueError("actor produced no executable hard negatives for the successor critic")
    output = round_path / "successor_critic"
    config = _critic_config(cfg, dataset, output, initial_adapter)
    if not (dataset / "tokens_1024/manifest.json").is_file():
        subprocess.run(
            ["torchrun", "--standalone", "--nproc_per_node=8",
             "scripts/prepare_tool_sft_arrow.py", "--config", str(config)],
            check=True,
        )
    tokens = json.loads((dataset / "tokens_1024/manifest.json").read_text())
    for name, split in (("train", "train"), ("validation", "valid")):
        if tokens["splits"][name]["n_rows"] != report["splits"][split]["rows"]:
            raise ValueError("successor critic token count mismatch")
        if tokens["splits"][name].get("truncation_count", 0):
            raise ValueError("successor critic tokenization was truncated")
    if not (output / "adapter_model.safetensors").is_file():
        subprocess.run(
            ["torchrun", "--standalone", "--nproc_per_node=8",
             "scripts/train_tool_sft.py", "--config", str(config)],
            check=True,
        )
    if not (output / "adapter_manifest.json").is_file():
        raise RuntimeError("successor critic training did not finish")
    return output


def run(cfg: dict[str, Any]) -> None:
    validate_contract(cfg)
    output = Path(cfg["output_dir"])
    prepare(cfg, output)
    curriculum_path = output / "curriculum.json"
    # Round markers, not a possibly half-written status file, define resume.
    curriculum = {
        "frontier": int(cfg["curriculum"]["initial_frontier"]), "history": []
    }
    for prior_index in range(int(cfg["rounds"])):
        marker = output / f"round{prior_index:02d}/round_done.json"
        if not marker.is_file():
            break
        prior = json.loads(marker.read_text())["curriculum"]
        if int(prior["frontier_before"]) != curriculum["frontier"]:
            raise ValueError("EARHO round markers have a discontinuous horizon")
        curriculum["frontier"] = int(prior["frontier_after"])
        curriculum["history"].append({"round": prior_index, **prior})
    write_json(curriculum_path, curriculum)
    actor = Path(cfg["initial_adapter_path"])
    critic = None
    baseline_cfg = dict(cfg, value_adapter_path=None)
    _, baseline = run_workers(
        baseline_cfg, output / "validation_monitor.jsonl", actor,
        output / "baseline_validation", frontier=int(curriculum["frontier"]),
        round_index=-1, evaluation=True,
    )
    best = {"actor": str(actor), "critic": None,
            "endpoint_rate": float(baseline["candidate_endpoint_rate"])}
    for round_index in range(int(cfg["rounds"])):
        round_path = output / f"round{round_index:02d}"
        done = round_path / "round_done.json"
        if done.is_file():
            record = json.loads(done.read_text())
            actor = Path(record["actor"])
            critic = Path(record["critic"])
            if not (actor / "adapter_model.safetensors").is_file():
                raise ValueError(f"completed EARHO round is missing actor: {actor}")
            if not (critic / "adapter_model.safetensors").is_file():
                raise ValueError(f"completed EARHO round is missing critic: {critic}")
            best = dict(record["best"])
            continue
        round_cfg = dict(cfg, value_adapter_path=str(critic) if critic else None)
        shards, collection = run_workers(
            round_cfg, round_path / "source.jsonl", actor,
            round_path / "rollouts", frontier=int(curriculum["frontier"]),
            round_index=round_index, evaluation=False,
        )
        next_critic = train_successor_critic(
            cfg, round_path / "source.jsonl", shards, round_path,
            critic or Path(cfg["initial_adapter_path"]),
        )
        training = round_path / "training.jsonl"
        if not training.exists():
            write_rows(
                training,
                (row for shard in shards for row in read_rows(shard)
                 if row.get("kind") != "collector_error"),
            )
        next_actor = run_train(
            cfg, training, actor, round_path / "actor_training",
            int(cfg["seed"]) + round_index,
            stage_script="scripts/natural_language_anchor_branch_stage.py",
        )
        if not (next_actor / "adapter_model.safetensors").is_file():
            raise RuntimeError("EARHO actor update did not return adapter weights")
        next_cfg = dict(cfg, value_adapter_path=str(next_critic))
        _, validation = run_workers(
            next_cfg, output / "validation_monitor.jsonl", next_actor,
            round_path / "validation", frontier=int(curriculum["frontier"]),
            round_index=-1, evaluation=True,
        )
        score = float(validation["candidate_endpoint_rate"])
        if score > float(best["endpoint_rate"]):
            best = {"actor": str(next_actor), "critic": str(next_critic),
                    "endpoint_rate": score}
        next_frontier, decision = promote_productive_horizon(
            int(curriculum["frontier"]), collection["group_summaries"],
            productive_pass_rate=float(cfg["curriculum"]["promote_pass_at_k"]),
            minimum_effective_groups=int(cfg["curriculum"]["min_effective_groups"]),
            maximum_horizon=int(cfg["curriculum"]["maximum_frontier"]),
        )
        curriculum["frontier"] = next_frontier
        curriculum["history"].append({"round": round_index, **decision})
        write_json(done, {"actor": str(next_actor), "critic": str(next_critic),
                          "best": best, "validation_endpoint_rate": score,
                          "curriculum": decision})
        write_json(curriculum_path, curriculum)
        actor, critic = next_actor, next_critic
        log(stage="earho-v2-round-complete", round=round_index,
            endpoint_rate=score, actor=str(actor), critic=str(critic), **decision)
    write_json(output / "completed.json", {
        "latest_actor": str(actor), "latest_critic": str(critic),
        "best": best, "test_used": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = _load_yaml(args.config)
    if cfg.get("protocol_version") == RELIABLE_PROTOCOL:
        data_root = os.environ.get("MECHET_RELIABLE_DATA_ROOT", "").strip()
        if data_root:
            cfg = resolve_reliable_paths(cfg, Path(data_root))
    if args.prepare_only:
        validate_contract(cfg)
        prepare(cfg, Path(cfg["output_dir"]))
    else:
        run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
