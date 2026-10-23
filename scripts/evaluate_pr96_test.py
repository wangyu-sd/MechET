#!/usr/bin/env python3
"""Execute the frozen PR96 trace-view TEST plan from public main products only.

This entry intentionally uses the unchanged, hash-bound TRAINING code package.
No ground-truth decision/state/fragment is passed to the closed-loop policy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

REFERENCE_ROOT = Path(os.environ.get("MECHET_PR96_REFERENCE_CODE_ROOT",
                                   Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(REFERENCE_ROOT), str(REFERENCE_ROOT / "src")]

from mechet.pr96_evaluation import public_root, messages_for, parsed
from mechet.pr96_eval_runtime import sha
from scripts.run_natural_language_value_search import Action, execute, visible
from scripts.build_mech31k_main_product_only_sft import same_chemical_state


def rollout(public_rows, backend, *, compact_history, max_decisions=64, journal=None):
    rows = list(public_rows)
    nodes, initials = {}, {}
    for row in rows:
        if set(row) != {"source_id", "product_smiles", "initial_mapped_product_state"}:
            raise ValueError("PR96_TEST_PUBLIC_FIELDS_REQUIRED_NO_GOLD")
        rid = str(row["source_id"])
        if rid in nodes:
            raise ValueError("PR96_TEST_DUPLICATE_PUBLIC_IDS")
        node = public_root({"target_smiles": row["product_smiles"]})
        if node.state != row["initial_mapped_product_state"]:
            raise ValueError("PR96_TEST_PUBLIC_ROOT_STATE_CHANGED")
        nodes[rid], initials[rid] = node, node.state
    errors, decisions, results = {}, Counter(), []
    def emit(value):
        if journal is not None:
            journal(value)
    def finish(rid):
        node = nodes[rid]
        value = {"reaction_id": rid, "target": node.target, "initial_state": initials[rid],
                 "terminal": node.terminal, "predicted_state": node.state,
                 "predicted_endpoint": visible(node.state), "actions": node.actions,
                 "generated_decisions": decisions[rid], "error": errors.get(rid, "")}
        results.append(value)
        emit({"mode": "episode_complete", **value})
    for depth in range(max_decisions):
        active = [(rid, node) for rid, node in nodes.items() if not node.terminal and rid not in errors]
        if not active:
            break
        prompts = [messages_for(node, compact_history) for _, node in active]
        outputs = backend.generate(prompts)
        if len(outputs) != len(active):
            raise ValueError("PR96_TEST_BACKEND_DROPPED_EPISODES")
        for (rid, node), prompt, output in zip(active, prompts, outputs):
            decisions[rid] += 1
            name, arguments, error = parsed(output)
            child = None
            if not error:
                child, error = execute(node, Action(name, arguments, output.get("raw", ""), 0,
                                                    len(output.get("generated_token_ids", []))), max_imports=32)
            if child is None:
                errors[rid] = error
            else:
                nodes[rid] = child
            emit({"mode": "closed_loop", "reaction_id": rid, "depth": depth,
                  "input_messages": prompt, "output": output,
                  "predicted_name": name, "predicted_arguments": arguments,
                  "state_before": node.state, "state_after": child.state if child else node.state,
                  "error": error, "gt_used_for_generation_or_stopping": False})
            if child is None or child.terminal:
                finish(rid)
    for rid, node in nodes.items():
        if not node.terminal and rid not in errors:
            errors[rid] = "DECISION_BUDGET_EXCEEDED"
            finish(rid)
    if len(results) != len(rows):
        raise ValueError("PR96_TEST_EPISODE_DENOMINATOR_CHANGED")
    return results


def score_endpoint(value, gold):
    """Score only after rollout; preserve source labels, stereo, charge and H."""
    from rdkit import Chem
    from mechet.endpoints import split_precursor_endpoints
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(gold["target_smiles"], parser)
    fragments = [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
                 for m in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)]
    main = sorted(f for f in fragments if same_chemical_state(f, value["target"]))
    if not main:
        raise ValueError("PR96_TEST_GOLD_MAIN_TARGET_CORRESPONDENCE_FAILED")
    expected = gold["full_precursor_state"]
    core_gt = split_precursor_endpoints(expected, main[0]).structural
    core_pred = split_precursor_endpoints(value["predicted_state"], value["initial_state"]).structural
    return {**value, "gt_endpoint": visible(expected), "gt_core": visible(core_gt) if core_gt else "",
            "predicted_core": visible(core_pred) if core_pred else "",
            "structural_exact": bool(value["terminal"] and same_chemical_state(core_pred, core_gt)),
            "full_exact": bool(value["terminal"] and same_chemical_state(value["predicted_state"], expected))}


def aggregate_results(parts, expected_ids):
    rows = [row for part in parts for row in part]
    ids = [row["reaction_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValueError("PR96_TEST_RESULT_IDS_MISSING_OR_DUPLICATED")
    total = len(rows)
    structural, full = sum(r["structural_exact"] for r in rows), sum(r["full_exact"] for r in rows)
    return {"denominator": total, "structural_hits": structural, "full_hits": full,
            "structural_accuracy": structural / max(total, 1), "full_accuracy": full / max(total, 1),
            "terminal_reached": sum(r["terminal"] for r in rows),
            "generated_decisions": sum(r["generated_decisions"] for r in rows),
            "failure_reasons": dict(Counter(r["error"] for r in rows if r["error"]))}


def main():
    import torch
    import torch.distributed as dist
    import yaml
    from peft import PeftModel
    from transformers import AutoTokenizer
    from mechet.pr96_artifacts import verify_evaluation_bindings
    from mechet.pr96_eval_runtime import HFBackend
    from mechet.pr96_trainer import validate_deployed_hardware
    from mechet.qwen35_protocol import load_language_model, OFFICIAL_REVISION
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path, required=True)
    args = p.parse_args()
    contract = json.loads(args.contract.read_text())
    for path, expected in contract["files_sha256"].items():
        if sha(path) != expected:
            raise ValueError("PR96_TEST_FROZEN_INPUT_HASH_CHANGED:" + path)
    plan = json.loads(Path(contract["plan"]).read_text())
    if (plan["original_TEST_denominator"] != 1253 or plan["strict_reference_available"] != 1252
            or plan["decoder"] != {"k": 1, "do_sample": False, "num_beams": 1,
                                   "max_new_tokens": 512, "max_decisions": 64, "max_imported_fragments": 32}):
        raise ValueError("PR96_TEST_FROZEN_PROTOCOL_CHANGED")
    rows = [json.loads(line) for line in Path(plan["public_roots_file"]).open()]
    expected_ids = [str(row["source_id"]) for row in rows]
    if len(rows) != 1253 or len(set(expected_ids)) != 1253:
        raise ValueError("PR96_TEST_FROZEN_DENOMINATOR_CHANGED")
    # No gold file is read or needed by rollout(). Endpoints are used below
    # only after that stage's on-policy generations have been persisted.
    gold_path = Path(contract["gold_source"])
    if sha(gold_path) != plan["test_source_sha256"]:
        raise ValueError("PR96_TEST_ORIGINAL_LABEL_SOURCE_CHANGED")
    validate_deployed_hardware("mech31k")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(contract["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=OFFICIAL_REVISION,
                                              local_files_only=True)
    comparison = {}
    for stage in ("state", "history"):
        cfg = yaml.safe_load(Path(contract["configs"][stage]).read_text())
        adapter = Path(plan["actors"][stage]["checkpoint"])
        if sha(adapter / "adapter_model.safetensors") != plan["actors"][stage]["adapter_sha256"]:
            raise ValueError("PR96_TEST_SELECTED_WEIGHT_CHANGED")
        bindings = verify_evaluation_bindings(cfg, adapter)
        target = output / stage
        if rank == 0:
            target.mkdir()
        dist.barrier()
        base, _ = load_language_model(cfg["model_name_or_path"], dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to(int(os.environ["LOCAL_RANK"]))
        model.eval()
        backend = HFBackend(model, tokenizer, max_new_tokens=512, batch_size=8,
                            raw_path=target / f"raw_generation.rank{rank:02d}.jsonl")
        completed = 0
        selected = rows[rank::world]
        def persist(value):
            nonlocal completed
            with (target / f"calls.rank{rank:02d}.jsonl").open("a") as sink:
                sink.write(json.dumps(value, ensure_ascii=False) + "\n")
                sink.flush()
            if value["mode"] == "episode_complete":
                completed += 1
                progress = {"stage": stage, "rank": rank, "completed": completed,
                            "total": len(selected), "percent": 100 * completed / max(len(selected), 1),
                            "reaction_id": value["reaction_id"], "terminal": value["terminal"],
                            "error": value["error"], "timestamp": datetime.now(timezone.utc).isoformat()}
                temporary = target / f"progress.rank{rank:02d}.json.building"
                temporary.write_text(json.dumps(progress) + "\n")
                temporary.replace(target / f"progress.rank{rank:02d}.json")
                print(json.dumps(progress), flush=True)
        predicted = rollout(selected, backend, compact_history=stage == "history", journal=persist)
        gold = {str(r["source_id"]): r for line in gold_path.open() if (r := json.loads(line))}
        if set(gold) != set(expected_ids):
            raise ValueError("PR96_TEST_GOLD_IDS_CHANGED")
        scored = [score_endpoint(value, gold[value["reaction_id"]]) for value in predicted]
        for value in scored:
            value["reference_unresolved"] = value["reaction_id"] in plan["unresolved_source_ids"]
        (target / f"result.rank{rank:02d}.json").write_text(json.dumps(scored, indent=2) + "\n")
        del backend, model, base
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            parts = [json.loads((target / f"result.rank{i:02d}.json").read_text()) for i in range(world)]
            metrics = aggregate_results(parts, expected_ids)
            report = {"track": "mech31k", "stage": stage, "split": "test_trace_view",
                      "full_raw_test_denominator": 3120, "is_full_raw3120_benchmark": False,
                      "metrics": metrics, "reference_available": 1252, "reference_unresolved": 1,
                      "unresolved_source_ids": plan["unresolved_source_ids"],
                      "decoder": plan["decoder"], "input_condition": "single_main_product_only",
                      "gt_used_for_policy": False, "teacher_forced": False,
                      "bindings": bindings, "test_plan_sha256": sha(contract["plan"]),
                      "test_public_roots_sha256": sha(plan["public_roots_file"]),
                      "original_test_labels_sha256": sha(gold_path),
                      "adapter": str(adapter), "adapter_sha256": plan["actors"][stage]["adapter_sha256"],
                      "timestamp": datetime.now(timezone.utc).isoformat()}
            (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            comparison[stage] = metrics
            print(json.dumps({"stage": stage, "event": "test-stage-complete", **metrics}), flush=True)
        dist.barrier()
    if rank == 0:
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
