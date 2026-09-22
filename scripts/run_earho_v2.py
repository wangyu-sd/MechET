#!/usr/bin/env python3
"""Stage-III EARHO for a frozen v2 compressed-history trajectory policy.

The collector performs product-start first-divergence discovery, executor reset,
same-state branching and bounded continuation.  This driver updates the actor
and a separate successor-value adapter without loading the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.earho_v2_protocol import promote_productive_horizon, replay_reference
from scripts.run_anchor_branch_rl import log, read_rows, run_train, write_json, write_rows
from scripts.run_natural_language_anchor_branch_rl import run_workers, _sha256
from scripts.train_python_template_rlvr import _load_yaml


PROTOCOL = "trajectory_history_v2"


def validate_contract(cfg: dict[str, Any]) -> None:
    if cfg.get("protocol_version") != PROTOCOL:
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
    if history_manifest.get("decision_contract") != "unified_inventory_compressed_history_tool_decision_v2":
        raise ValueError("v2 compressed-history decision contract mismatch")
    if history_manifest.get("reaction_denominator") != expected:
        raise ValueError("history executable denominator changed")
    if history_manifest.get("full_reaction_denominator") != full:
        raise ValueError("history complete denominator changed")
    repository_root = source_dir.parents[1]
    event_dir = repository_root / str(history_manifest["source_artifact"])
    event_manifest_path = event_dir / "manifest.json"
    event_manifest = json.loads(event_manifest_path.read_text())
    if event_manifest.get("source_artifact") != str(source_dir.relative_to(repository_root)):
        raise ValueError("event supervision does not descend from selected source")
    if event_manifest.get("source_manifest_sha256") != _sha256(Path(cfg["stable_id_manifest"])):
        raise ValueError("event/source manifest lineage mismatch")
    if history_manifest.get("source_manifest_sha256") != _sha256(event_manifest_path):
        raise ValueError("history/event manifest lineage mismatch")
    for split in ("train", "valid"):
        if _sha256(history_dir / f"{split}.jsonl") != history_manifest["splits"][split]["output_sha256"]:
            raise ValueError(f"{split} history SHA-256 mismatch")
        if event_manifest["splits"][split]["source_sha256"] != source_manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} event/source file lineage mismatch")
        if history_manifest["splits"][split]["source_sha256"] != event_manifest["splits"][split]["output_sha256"]:
            raise ValueError(f"{split} history/event file lineage mismatch")
    adapter = Path(cfg["initial_adapter_path"])
    if _sha256(adapter / "adapter_model.safetensors") != cfg["initial_adapter_model_sha256"]:
        raise ValueError("Stage-II parent adapter SHA-256 mismatch")
    adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text())
    if adapter_manifest.get("environment_revision") != "natural_language_electron_event_history_v2":
        raise ValueError("parent is not the v2 Trajectory-SFT policy")
    if adapter_manifest.get("base_model_revision") != cfg["model_revision"]:
        raise ValueError("parent/base model revision mismatch")
    if adapter_manifest.get("train_file_sha256") != history_manifest["splits"]["train"]["output_sha256"]:
        raise ValueError("Stage-II parent was not trained on the selected 31k history data")
    if cfg.get("value_adapter_path"):
        raise ValueError("v2 successor value must start untrained and be learned from actor rollouts")


def _attach_decisions(
    rows: list[dict[str, Any]], history_file: Path
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
        packed = dict(row, earho_v2_reference_decisions=found[key])
        replay_reference(packed, found[key])
        output.append(packed)
    return output


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
        if plan.get("protocol_version") != PROTOCOL:
            raise ValueError("existing EARHO plan has a different protocol")
        for name, digest in (plan.get("prepared_files") or {}).items():
            if _sha256(output / name) != digest:
                raise ValueError(f"prepared EARHO source changed: {name}")
        if len(plan.get("prepared_files") or {}) != int(cfg["rounds"]) + 1:
            raise ValueError("EARHO preparation manifest is incomplete")
        return
    source = read_rows(cfg["train_file"])
    validation = read_rows(cfg["validation_file"])
    if len(source) != cfg["reaction_denominator"]["train"]:
        raise ValueError("source train count mismatch")
    if len(validation) != cfg["reaction_denominator"]["valid"]:
        raise ValueError("source validation count mismatch")
    random.Random(int(cfg["seed"])).shuffle(source)
    random.Random(int(cfg["seed"])).shuffle(validation)
    count = int(cfg["rounds"]) * int(cfg["products_per_round"])
    if count > len(source):
        raise ValueError("requested more distinct RL train reactions than available")
    selected = _attach_decisions(source[:count], Path(cfg["history_file"]))
    monitor = _attach_decisions(
        validation[: int(cfg["validation_monitor_rows"])],
        Path(cfg["history_validation_file"]),
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
            "protocol_version": PROTOCOL,
            "config_sha256": config_sha256,
            "source_reactions": len(source),
            "selected_train_reactions": count,
            "validation_monitor_reactions": len(monitor),
            "selected_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "prepared_files": prepared_files,
            "initial_adapter_model_sha256": cfg["initial_adapter_model_sha256"],
            "reference_endpoint_model_visible": False,
            "first_divergence_from_product_rollout": True,
            "actor_prompt_history_contract": "executor_compact_accepted_actions_v1",
            "test_used": False,
        },
    )
    log(stage="earho-v2-prepare", train=count, validation=len(monitor))


def _critic_config(
    cfg: dict[str, Any], dataset: Path, output: Path, initial_adapter: Path
) -> Path:
    import yaml

    template = ROOT / "configs/agent/natural_language_state_value_qwen3_8b_h20.yaml"
    critic = yaml.safe_load(template.read_text())
    manifest = json.loads((dataset / "manifest.json").read_text())
    critic.update(
        condition_name="earho_v2_successor_value_pn",
        scientific_hypothesis="actor_successor_value_improves_executable_continuation",
        initial_adapter_path=str(initial_adapter),
        train_file=str(dataset / "train.jsonl"),
        validation_file=str(dataset / "valid.jsonl"),
        pretokenized_cache_dir=str(dataset / "tokens_1024"),
        output_dir=str(output),
    )
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
        source_dataset="mech_uspto_31k_current_compiler_executable_trace_view",
        source_artifact=cfg["train_file"],
        reaction_denominator=cfg["reaction_denominator"],
        environment_revision="earho_v2_successor_value_pn",
        executor_revision="mech_uspto31k_current_compiler_20260824",
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
    if args.prepare_only:
        validate_contract(cfg)
        prepare(cfg, Path(cfg["output_dir"]))
    else:
        run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
