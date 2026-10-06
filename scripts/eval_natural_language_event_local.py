#!/usr/bin/env python3
"""Gold-state local evaluation for natural-language electron-event SFT.

The model receives exactly the product/current-state prompt used by SFT and
generates one next tool decision.  Private atom maps are reconstructed only in
the evaluator so predicted natural-language aliases can be compiled and
strictly replayed.  This is an F-oracle local diagnostic, not a product-only
closed-loop endpoint result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from mechet.a7_rescue import canonical_event, mechanism_length_stratum, stratified_sample
from mechet.agent_inference import parse_tool_calls
from mechet.assistant_masking import render_chat, render_qwen_sft_tool_prefix
from mechet.forward_expert import ElectronMove, verify_electron_step
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    extract_import_fragments,
    mapped_atom_numbers,
    mapped_state_signature,
    retain_mapped_components,
    schedule_imports,
)
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.build_natural_language_event_sft import convert_row


MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def validate_adapter_lineage(
    adapter: Path, model: str, revision: str,
    *, provisional_training_config: Path | None = None,
) -> dict[str, Any]:
    manifest_path = adapter / "adapter_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("base_model") != model:
            raise ValueError("evaluation model differs from adapter base model")
        if manifest.get("base_model_revision") != revision:
            raise ValueError("evaluation model revision differs from adapter revision")
        if provisional_training_config is not None:
            raise ValueError("a completed adapter must not be labelled provisional")
        return {"kind": "completed_adapter", "checkpoint_step": None}
    if provisional_training_config is None:
        raise FileNotFoundError(f"adapter lineage manifest missing: {manifest_path}")
    import yaml

    match = re.fullmatch(r"checkpoint-(\d+)", adapter.name)
    if match is None:
        raise ValueError("provisional adapter must be an explicit trainer checkpoint")
    config = yaml.safe_load(provisional_training_config.read_text(encoding="utf-8"))
    if config.get("model_name_or_path") != model or (
        config.get("training") or {}
    ).get("model_revision") != revision:
        raise ValueError("provisional checkpoint model/revision differs from training config")
    if Path(str(config.get("output_dir") or "")).name != adapter.parent.name:
        raise ValueError("provisional checkpoint is outside the configured training output")
    if (config.get("contract") or {}).get("stage") not in {
        "state_sft", "trajectory_sft",
    }:
        raise ValueError("provisional checkpoint is not a three-stage SFT adapter")
    state = json.loads((adapter / "trainer_state.json").read_text(encoding="utf-8"))
    step = int(match.group(1))
    if int(state.get("global_step", -1)) != step or step <= 0 or (
        int(state.get("max_steps", -1)) <= step
    ):
        raise ValueError("provisional checkpoint step does not match trainer state")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != model:
        raise ValueError("provisional checkpoint PEFT base model differs")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("provisional checkpoint has no adapter weights")
    return {
        "kind": "provisional_checkpoint", "checkpoint_step": step,
        "training_config_sha256": sha256(provisional_training_config),
        "training_stage": config["contract"]["stage"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distributed_coordinates() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not 0 <= rank < world:
        raise ValueError("invalid rank/world size")
    return rank, world, local_rank


def render_policy_prompt(
    tokenizer: Any, task: Mapping[str, Any], *, sft_aligned: bool
) -> str:
    messages = [dict(message) for message in task["messages"]]
    tools = [dict(tool) for tool in task["tools"]]
    if sft_aligned:
        return render_qwen_sft_tool_prefix(tokenizer, messages, tools=tools)
    return render_chat(
        tokenizer, messages, tools=tools, add_generation_prompt=True
    )


def _private_states(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct executor-private state aligned with every public decision."""

    plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
    steps = [dict(item) for item in plan.get("steps") or []]
    fragments = extract_import_fragments(row)
    scheduled = schedule_imports(fragments, steps)
    present_maps = set(mapped_atom_numbers(str(row["target_smiles"])))
    current = retain_mapped_components(str(steps[0]["state_before"]), present_maps)
    output: list[dict[str, Any]] = []
    event_depth = 0
    for step, imports in zip(steps, scheduled, strict=True):
        if imports:
            output.append(
                {
                    "decision_type": "import",
                    "private_state": current,
                    "reference_successor": "",
                    "event_depth": event_depth,
                }
            )
            current_with_imports = append_mapped_fragments_verbatim(current, imports)
            for fragment in imports:
                present_maps.update(mapped_atom_numbers(fragment))
        else:
            current_with_imports = current
        successor = retain_mapped_components(str(step["state_after"]), present_maps)
        output.append(
            {
                "decision_type": "event",
                "private_state": current_with_imports,
                "reference_successor": successor,
                "event_depth": event_depth + 1,
            }
        )
        current = successor
        event_depth += 1
    output.append(
        {
            "decision_type": "finish",
            "private_state": current,
            "reference_successor": current,
            "event_depth": event_depth,
        }
    )
    return output


def collect_tasks(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_reactions: int,
    seed: int,
    decision_rows: Iterable[Mapping[str, Any]] | None = None,
    replay_decision_states: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    if replay_decision_states and decision_rows is None:
        raise ValueError("reference-state replay requires frozen decision rows")
    selected = stratified_sample(rows, size=sample_reactions, seed=seed)
    selected_ids = {str(source["source_id"]) for source in selected}
    public_by_reaction: dict[str, list[dict[str, Any]]] | None = None
    if decision_rows is not None:
        public_by_reaction = defaultdict(list)
        for row in decision_rows:
            reaction_id = str(row["source_id"])
            if reaction_id in selected_ids:
                public_by_reaction[reaction_id].append(dict(row))
        for reaction_id, decisions in public_by_reaction.items():
            decisions.sort(key=lambda row: int(row["metadata"]["decision_index"]))
            indices = [int(row["metadata"]["decision_index"]) for row in decisions]
            if indices != list(range(len(decisions))):
                raise ValueError(f"{reaction_id}: public decision indices are incomplete")
    tasks: list[dict[str, Any]] = []
    reaction_ids: list[str] = []
    for source in selected:
        reaction_id = str(source["source_id"])
        public_rows = (
            public_by_reaction.get(reaction_id, [])
            if public_by_reaction is not None
            else convert_row(source)
        )
        private_rows = _private_states(source)
        if len(public_rows) != len(private_rows):
            raise ValueError(f"{source['id']}: public/private decision count mismatch")
        if replay_decision_states:
            from scripts.earho_v2_protocol import replay_reference

            contracts = {
                str((decision.get("metadata") or {}).get("decision_contract") or "")
                for decision in public_rows
            }
            if contracts == {"unified_inventory_tool_decision_v2"}:
                compact_history = False
            elif contracts == {"unified_inventory_compressed_history_tool_decision_v2"}:
                compact_history = True
            else:
                raise ValueError(f"{reaction_id}: unsupported/mixed decision contracts: {contracts}")
            reference = replay_reference(
                source, public_rows, compact_history=compact_history,
            )
            private_rows = [
                {
                    **private,
                    "private_state": reference.nodes[index].state,
                    "reference_successor": (
                        reference.nodes[index + 1].state
                        if private["decision_type"] != "import" else ""
                    ),
                }
                for index, private in enumerate(private_rows)
            ]
        stratum = mechanism_length_stratum(
            len((source["metadata"]["trace_plan"] or {})["steps"])
        )
        reaction_ids.append(reaction_id)
        for public, private in zip(public_rows, private_rows, strict=True):
            decision_type = str(public["metadata"]["decision_type"])
            if decision_type != private["decision_type"]:
                raise ValueError(f"{public['id']}: decision alignment changed")
            gold_call = public["messages"][2]["tool_calls"][0]["function"]
            tasks.append(
                {
                    "key": str(public["id"]),
                    "reaction_id": reaction_id,
                    "decision_type": decision_type,
                    "event_depth": int(private["event_depth"]),
                    "stratum": stratum,
                    "messages": public["messages"][:2],
                    "tools": public["tools"],
                    "gold_name": str(gold_call["name"]),
                    "gold_arguments": dict(gold_call["arguments"]),
                    "private_state": str(private["private_state"]),
                    "reference_successor": str(private["reference_successor"]),
                }
            )
    if len({task["key"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate evaluation decision key")
    return tasks, reaction_ids


def prediction_call(text: str, tokenizer: Any) -> tuple[str, dict[str, Any], str]:
    try:
        calls = parse_tool_calls(text, tokenizer=tokenizer)
    except Exception as exc:
        return "", {}, f"PARSE_ERROR:{type(exc).__name__}:{exc}"
    if len(calls) != 1:
        return "", {}, f"EXPECTED_ONE_TOOL_CALL:observed={len(calls)}"
    return str(calls[0].name), dict(calls[0].arguments), ""


def _normal_smiles(smiles: str) -> str:
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError(f"invalid imported SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _import_counter(
    arguments: Mapping[str, Any], *, include_purpose: bool
) -> Counter[tuple[str, str]]:
    output: Counter[tuple[str, str]] = Counter()
    fragments = arguments.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("fragments must be a nonempty list")
    for item in fragments:
        if not isinstance(item, Mapping):
            raise ValueError("fragment entry must be an object")
        count = int(item.get("count", 0))
        if count <= 0:
            raise ValueError("fragment count must be positive")
        purpose = str(item.get("purpose") or "") if include_purpose else ""
        output[(_normal_smiles(str(item.get("smiles") or "")), purpose)] += count
    return output


def _import_role_counter(
    schedule: Counter[tuple[str, str]], purpose: str,
) -> Counter[str]:
    return Counter({smiles: count for (smiles, role), count in schedule.items()
                    if role == purpose})


def _site_signature(moves: Sequence[Mapping[str, Any]], field: str) -> list[tuple[str, tuple[int, ...]]]:
    output = []
    for raw in moves:
        if raw.get("mode") == "BE_DELTA":
            continue
        move = ElectronMove.parse(raw)
        container = move.source if field == "source" else move.sink
        output.append((container.kind, tuple(container.atoms)))
    return sorted(output)


def score_prediction(
    task: Mapping[str, Any], *, predicted_name: str, predicted_arguments: Mapping[str, Any]
) -> dict[str, Any]:
    dtype = str(task["decision_type"])
    gold_name = str(task["gold_name"])
    gold_arguments = dict(task["gold_arguments"])
    correct_tool = predicted_name == gold_name
    gold_schedule = (
        _import_counter(gold_arguments, include_purpose=True)
        if dtype == "import" else Counter()
    )
    gold_participants = _import_role_counter(gold_schedule, "electron_participant")
    gold_context = _import_role_counter(gold_schedule, "endpoint_context")
    base = {
        "correct_tool": correct_tool,
        "decision_exact": False,
        "argument_compile": False,
        "formal_execute": False,
        "event_exact": False,
        "source_site_exact": False,
        "destination_site_exact": False,
        "successor_map_exact": False,
        "successor_chemical_exact": False,
        "import_fragment_exact": False,
        "import_schedule_exact": False,
        "import_participant_exact": False,
        "import_context_exact": False,
        "gold_import_participant_copies": sum(gold_participants.values()),
        "gold_import_context_copies": sum(gold_context.values()),
        "finish_exact": False,
        "execution_error": "",
    }
    if not correct_tool:
        base["execution_error"] = f"WRONG_TOOL:{predicted_name or 'NONE'}"
        return base
    try:
        if dtype == "finish":
            exact = not predicted_arguments
            base.update(decision_exact=exact, argument_compile=exact, finish_exact=exact)
            return base
        if dtype == "import":
            predicted_fragments = _import_counter(predicted_arguments, include_purpose=False)
            gold_fragments = _import_counter(gold_arguments, include_purpose=False)
            predicted_schedule = _import_counter(predicted_arguments, include_purpose=True)
            fragment_exact = predicted_fragments == gold_fragments
            schedule_exact = predicted_schedule == gold_schedule
            base.update(
                decision_exact=schedule_exact,
                argument_compile=True,
                import_fragment_exact=fragment_exact,
                import_schedule_exact=schedule_exact,
                import_participant_exact=(
                    _import_role_counter(predicted_schedule, "electron_participant")
                    == gold_participants
                ),
                import_context_exact=(
                    _import_role_counter(predicted_schedule, "endpoint_context")
                    == gold_context
                ),
            )
            return base
        if dtype != "event":
            raise ValueError(f"unsupported decision type: {dtype}")
        predicted_moves = compile_event_arguments(
            str(task["private_state"]), predicted_arguments
        )
        gold_moves = compile_event_arguments(str(task["private_state"]), gold_arguments)
        execution = verify_electron_step(str(task["private_state"]), predicted_moves)
        formal = bool(execution.get("ok"))
        reference = str(task["reference_successor"])
        predicted_state = str(execution.get("state_smiles") or "")
        map_exact = bool(
            formal
            and mapped_state_signature(predicted_state)
            == mapped_state_signature(reference)
        )
        chemical_exact = bool(
            formal
            and deterministic_unmapped_state(predicted_state).text
            == deterministic_unmapped_state(reference).text
        )
        event_exact = canonical_event(predicted_moves) == canonical_event(gold_moves)
        base.update(
            decision_exact=event_exact,
            argument_compile=True,
            formal_execute=formal,
            event_exact=event_exact,
            source_site_exact=_site_signature(predicted_moves, "source")
            == _site_signature(gold_moves, "source"),
            destination_site_exact=_site_signature(predicted_moves, "destination")
            == _site_signature(gold_moves, "destination"),
            successor_map_exact=map_exact,
            successor_chemical_exact=chemical_exact,
            execution_error=(
                ""
                if formal
                else str(execution.get("code") or execution.get("message") or "EXECUTION_FAILED")
            ),
        )
        return base
    except Exception as exc:
        base["execution_error"] = f"{type(exc).__name__}:{exc}"
        return base


def _trim_completion(ids: list[int], tokenizer: Any) -> list[int]:
    stop = {value for value in (tokenizer.eos_token_id,) if value is not None}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop.add(im_end)
    for index, token_id in enumerate(ids):
        if token_id in stop:
            return ids[: index + 1]
    return ids


def _batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run(args: argparse.Namespace) -> int:
    validate_adapter_lineage(
        args.adapter, args.model, args.model_revision,
        provisional_training_config=args.provisional_training_config,
    )
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rank, world, local_rank = distributed_coordinates()
    torch.cuda.set_device(local_rank)
    rows = read_jsonl(args.data)
    tasks, reaction_ids = collect_tasks(
        rows,
        sample_reactions=args.sample_reactions,
        seed=args.seed,
        decision_rows=read_jsonl(args.decision_data) if args.decision_data else None,
        replay_decision_states=getattr(args, "replay_reference_states", False),
    )
    selected = [task for index, task in enumerate(tasks) if index % world == rank]
    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.output / f"decisions.shard-{rank:02d}-of-{world:02d}.jsonl"
    completed = {row["key"] for row in read_jsonl(shard)} if shard.exists() else set()
    selected = [task for task in selected if task["key"] not in completed]
    if rank == 0:
        (args.output / "selection.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "sample_reactions": args.sample_reactions,
                    "reaction_ids": reaction_ids,
                    "reaction_ids_sha256": hashlib.sha256(
                        "\n".join(reaction_ids).encode()
                    ).hexdigest(),
                    "planned_decisions": len(tasks),
                },
                indent=2,
            )
            + "\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": {"": local_rank},
        "attn_implementation": "sdpa",
    }
    bnb_version = "disabled"
    if not args.no_4bit:
        try:
            bnb_version = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "bitsandbytes is required unless --no-4bit is used"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    print(
        f"[meteor-nl-event-eval] rank={rank}/{world} gpu={torch.cuda.get_device_name(local_rank)} "
        f"tasks={len(selected)} bnb={bnb_version} dtype={args.dtype} "
        f"sft_aligned={args.sft_aligned_prefix}",
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    device = next(model.parameters()).device
    with shard.open("a", encoding="utf-8") as sink:
        done = 0
        for batch in _batches(selected, args.batch_size):
            started = time.time()
            prompts = [
                render_policy_prompt(
                    tokenizer, task, sft_aligned=args.sft_aligned_prefix
                )
                for task in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            max_input = int(encoded["input_ids"].shape[1])
            if max_input + args.max_new_tokens > args.max_context:
                raise ValueError(
                    f"context budget exceeded: input={max_input} new={args.max_new_tokens}"
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.time() - started
            input_width = int(encoded["input_ids"].shape[1])
            for task, sequence in zip(batch, output, strict=True):
                completion_ids = _trim_completion(
                    [int(value) for value in sequence[input_width:].tolist()], tokenizer
                )
                generated = tokenizer.decode(completion_ids, skip_special_tokens=False)
                name, arguments, generation_error = prediction_call(generated, tokenizer)
                metrics = score_prediction(
                    task,
                    predicted_name=name,
                    predicted_arguments=arguments,
                )
                record = {
                    "key": task["key"],
                    "private_state_source": (
                        "executor_reference_replay"
                        if getattr(args, "replay_reference_states", False)
                        else "trace_plan_reconstruction"
                    ),
                    "reaction_id": task["reaction_id"],
                    "decision_type": task["decision_type"],
                    "event_depth": task["event_depth"],
                    "stratum": task["stratum"],
                    "gold_name": task["gold_name"],
                    "predicted_name": name,
                    "predicted_arguments": arguments,
                    "generated_text": generated,
                    "generation_error": generation_error,
                    "generated_tokens": len(completion_ids),
                    "batch_seconds": elapsed,
                    **metrics,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                done += 1
            counts = Counter(task["decision_type"] for task in batch)
            print(
                f"[meteor-nl-event-eval] rank={rank} done={done}/{len(selected)} "
                f"types={dict(counts)} seconds={elapsed:.2f}",
                flush=True,
            )
    return 0


METRICS = (
    "correct_tool",
    "decision_exact",
    "argument_compile",
    "formal_execute",
    "event_exact",
    "source_site_exact",
    "destination_site_exact",
    "successor_map_exact",
    "successor_chemical_exact",
    "import_fragment_exact",
    "import_schedule_exact",
    "finish_exact",
)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    for name in METRICS:
        count = sum(bool(row.get(name)) for row in rows)
        result[name] = count
        result[f"{name}_rate"] = count / max(len(rows), 1)
    result["mean_generated_tokens"] = sum(
        int(row.get("generated_tokens", 0)) for row in rows
    ) / max(len(rows), 1)
    return result


def _import_role_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    result: dict[str, Any] = {"n": n}
    for name in (
        "correct_tool", "import_fragment_exact", "import_schedule_exact",
    ):
        count = sum(bool(row[name]) for row in rows)
        result[name] = count
        result[f"{name}_rate"] = count / n if n else None
    for name, presence_key in (
        ("import_participant_exact", "gold_import_participant_copies"),
        ("import_context_exact", "gold_import_context_copies"),
    ):
        eligible = [row for row in rows if int(row[presence_key]) > 0]
        count = sum(bool(row[name]) for row in eligible)
        result[f"{name}_n"] = len(eligible)
        result[name] = count
        result[f"{name}_rate"] = count / len(eligible) if eligible else None
    return result


def aggregate(args: argparse.Namespace) -> int:
    adapter_lineage = validate_adapter_lineage(
        args.adapter, args.model, args.model_revision,
        provisional_training_config=args.provisional_training_config,
    )
    source = read_jsonl(args.data)
    tasks, reaction_ids = collect_tasks(
        source,
        sample_reactions=args.sample_reactions,
        seed=args.seed,
        decision_rows=read_jsonl(args.decision_data) if args.decision_data else None,
        replay_decision_states=getattr(args, "replay_reference_states", False),
    )
    expected = {task["key"] for task in tasks}
    rows: list[dict[str, Any]] = []
    for path in sorted(args.output.glob("decisions.shard-*-of-*.jsonl")):
        rows.extend(read_jsonl(path))
    private_state_source = (
        "executor_reference_replay"
        if getattr(args, "replay_reference_states", False)
        else "trace_plan_reconstruction"
    )
    if any(row.get("private_state_source", "trace_plan_reconstruction") != private_state_source
           for row in rows):
        raise ValueError("local evaluation shards use a different private-state contract")
    observed = [str(row["key"]) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate prediction keys")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    by_type = {
        name: _summary([row for row in rows if row["decision_type"] == name])
        for name in ("import", "event", "finish")
    }
    by_import_role = None
    if getattr(args, "import_role_breakdown", False):
        imports = [row for row in rows if row["decision_type"] == "import"]
        for row in imports:
            if not all(field in row for field in (
                "gold_import_participant_copies", "gold_import_context_copies",
                "import_participant_exact", "import_context_exact",
            )):
                raise ValueError(f"import role metrics missing from prediction {row['key']}")
            if int(row["gold_import_participant_copies"]) + int(row["gold_import_context_copies"]) < 1:
                raise ValueError(f"gold import has no recognized fragment role: {row['key']}")
        by_import_role = {
            "participant_present": _import_role_summary([
                row for row in imports if int(row["gold_import_participant_copies"]) > 0
            ]),
            "context_present": _import_role_summary([
                row for row in imports if int(row["gold_import_context_copies"]) > 0
            ]),
            "participant_only": _import_role_summary([
                row for row in imports
                if int(row["gold_import_participant_copies"]) > 0
                and int(row["gold_import_context_copies"]) == 0
            ]),
            "context_only": _import_role_summary([
                row for row in imports
                if int(row["gold_import_participant_copies"]) == 0
                and int(row["gold_import_context_copies"]) > 0
            ]),
            "mixed": _import_role_summary([
                row for row in imports
                if int(row["gold_import_participant_copies"]) > 0
                and int(row["gold_import_context_copies"]) > 0
            ]),
        }
    artifact_type = (
        "natural_language_event_gold_state_local_k1_v3_replayed_state"
        if getattr(args, "replay_reference_states", False)
        else "natural_language_event_gold_state_local_k1_v2"
        if args.sft_aligned_prefix
        else "natural_language_event_gold_state_local_k1_v1"
    )
    if adapter_lineage["kind"] == "provisional_checkpoint":
        artifact_type += "_provisional_checkpoint"
    report = {
        "artifact_type": artifact_type,
        "claim_boundary": (
            "Fixed validation F-oracle one-decision diagnostic; not product-only "
            "closed-loop rollout and not test endpoint accuracy."
            + (
                " Unfinished training checkpoint; not a final-model result."
                if adapter_lineage["kind"] == "provisional_checkpoint" else ""
            )
        ),
        "seed": args.seed,
        "planned_reactions": args.sample_reactions,
        "reaction_ids_sha256": hashlib.sha256("\n".join(reaction_ids).encode()).hexdigest(),
        "planned_decisions": len(expected),
        "completed_decisions": len(rows),
        "missing_decisions": len(missing),
        "extra_decisions": len(extra),
        "complete": not missing and not extra,
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "decision_data": str(args.decision_data) if args.decision_data else None,
        "decision_data_sha256": sha256(args.decision_data) if args.decision_data else None,
        "private_state_source": private_state_source,
        "adapter": str(args.adapter),
        "adapter_model_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "adapter_lineage": adapter_lineage,
        "model": args.model,
        "model_revision": args.model_revision,
        "compute_dtype": args.dtype,
        "sft_aligned_tool_prefix": bool(args.sft_aligned_prefix),
        "quantization": "bf16_or_fp16" if args.no_4bit else "bnb_nf4",
        "overall": _summary(rows),
        "by_type": by_type,
        "event_by_stratum": {
            name: _summary(
                [
                    row
                    for row in rows
                    if row["decision_type"] == "event" and row["stratum"] == name
                ]
            )
            for name in ("short", "medium", "long")
        },
        "generation_errors": dict(
            Counter(row["generation_error"] for row in rows if row["generation_error"])
        ),
        "execution_errors": dict(
            Counter(row["execution_error"] for row in rows if row["execution_error"])
        ),
    }
    if by_import_role is not None:
        report["by_import_role"] = by_import_role
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--decision-data",
        type=Path,
        help="frozen model-visible per-decision rows; needed for Stage-II history prompts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--provisional-training-config", type=Path,
        help="explicit one-off diagnostic of an unfinished trainer checkpoint; not a final model",
    )
    parser.add_argument("--sample-reactions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="load the base model without NF4 quantization",
    )
    parser.add_argument(
        "--sft-aligned-prefix", action="store_true",
        help="use the exact Qwen assistant boundary preceding Tool-SFT calls",
    )
    parser.add_argument(
        "--import-role-breakdown", action="store_true",
        help="report participant/context import accuracy; requires role-labelled gold rows",
    )
    parser.add_argument(
        "--replay-reference-states", action="store_true",
        help="derive private scoring states from verified frozen decision replay",
    )
    args = parser.parse_args()
    return run(args) if args.command == "run" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
