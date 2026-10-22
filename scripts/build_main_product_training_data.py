#!/usr/bin/env python3
"""Rebuild PR96 corpora from frozen strict traces; no models or label repairs.

All original IDs remain in accepted or unresolved journals. An independent
review/authorization artifact is required before these outputs can train.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import signal
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.in_place_grounded_flow import (
    extract_import_fragments, mapped_atom_numbers, retain_mapped_components, schedule_imports,
)
from mechet.inverse_trace_data import select_mapped_target
from mechet.pr96_durable_io import DurableAppendSet
from scripts.pr96_build_recovery import (
    copy_completed_split, encoded, recover_prefix, result_payload, verify_recovery_code,
)
from scripts.build_mech31k_main_product_only_sft import same_chemical_state, sha
from scripts.build_natural_language_event_sft import convert_row, _worker_init
from scripts.run_natural_language_value_search import (
    Action, Node, execute, policy_prompt, product_only_private_state, read_selected, visible,
)

DENOMINATORS = {
    "mech31k": {"train": 10152, "valid": 1319, "test": 1253},
    "flower": {"train": 257167, "valid": 2890, "test": 28967},
}


def prepare_reference(source):
    """Select the main component using the product alone; preserve the plan.

    Missing product-side species join existing first-use imports. The stored
    reference is a private plan, not a stale full conversation with hidden root.
    """
    row = deepcopy(source)
    original_target = str(source["target_smiles"])
    target, removed, selection = select_mapped_target(
        original_target, original_target, main_product_only=True,
    )
    original_imports = extract_import_fragments(source)
    plan = row["metadata"]["trace_plan"]
    plan["target_smiles"] = target
    plan["initial_imports"] = original_imports + list(removed)
    present = set(mapped_atom_numbers(target))
    for fragment in plan["initial_imports"]:
        maps = mapped_atom_numbers(fragment)
        if maps & present:
            raise ValueError("IMPORT_MAP_OVERLAP")
        present.update(maps)
    first_state_maps = mapped_atom_numbers(plan["steps"][0]["state_before"])
    # Later step imports are represented in legacy import_fragment calls too.
    if not first_state_maps <= present:
        raise ValueError("SOURCE_ATOMS_NOT_ACCOUNTED_FOR")
    row["target_smiles"] = target
    row["metadata"]["target_selection"] = selection
    row["metadata"]["input_condition"] = "main_product_only_v1"
    row["metadata"]["original_product_side"] = original_target
    row["metadata"]["removed_product_components"] = list(removed)
    row["metadata"]["source_row_sha256"] = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    row["metadata"]["n_trace_imports"] = len(plan["initial_imports"])
    row["artifact_type"] = "private_reference_plan_not_training_conversation"
    row["messages"] = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": f"reference_import_{i}", "type": "function",
        "function": {"name": "import_fragment", "arguments": {"fragment_smiles": f}},
    }]} for i, f in enumerate(plan["initial_imports"])]
    return row


def audit_reference(inverse):
    """Independent public-root executor replay of each generated decision.

    The exact source arrows are rendered by the strict converter; every FLOW
    successor must match the original chemistry including stereo/H/charge.
    """
    decisions = convert_row(inverse, preserve_source_moves=True)
    target = decisions[0]["target_smiles"]
    if "." in target:
        raise ValueError("ROOT_NOT_SINGLE_COMPONENT")
    initial = product_only_private_state(target)
    node = Node(target=target, state=initial,
                next_map=max(mapped_atom_numbers(initial)) + 1, visited={visible(initial)})
    steps = inverse["metadata"]["trace_plan"]["steps"]
    scheduled = schedule_imports(extract_import_fragments(inverse), steps)
    source_present = set(mapped_atom_numbers(inverse["target_smiles"]))
    names = []
    flow_index = 0
    import_roles = Counter()
    for decision in decisions:
        decision["metadata"]["input_condition"] = "main_product_only_v1"
        prompt = policy_prompt(target, node.state, include_inventory=True,
                               actions=node.actions, compact_history=False)
        if prompt != decision["messages"][1]["content"]:
            raise ValueError("PUBLIC_PREFIX_MISMATCH")
        function = decision["messages"][2]["tool_calls"][0]["function"]
        name, args = function["name"], function["arguments"]
        child, error = execute(node, Action(name, args, "", 0, 0),
                               max_imports=sum(len(v) for v in scheduled),
                               reject_target_retained_finish=False)
        if child is None:
            raise ValueError(f"PUBLIC_REFERENCE_EXECUTION_FAILED:{len(names)}:{error}")
        if name == "import_fragments":
            for fragment in scheduled[flow_index]:
                source_present.update(mapped_atom_numbers(fragment))
            for fragment in args["fragments"]:
                import_roles[fragment["purpose"]] += fragment["count"]
        elif name == "apply_electron_flow":
            expected = retain_mapped_components(steps[flow_index]["state_after"], source_present)
            if not same_chemical_state(child.state, expected):
                raise ValueError(f"PUBLIC_REFERENCE_SUCCESSOR_MISMATCH:{flow_index}")
            flow_index += 1
        result = json.loads(decision["messages"][3]["content"])
        claimed = result.get("current_state", result.get("derived_precursor"))
        if not same_chemical_state(child.state, claimed):
            raise ValueError("DECISION_RESULT_MISMATCH")
        node = child
        names.append(name)
    endpoint = inverse.get("full_precursor_state") or inverse["expected_precursor"]
    if not node.terminal or flow_index != len(steps) or not same_chemical_state(node.state, endpoint):
        raise ValueError("PUBLIC_REFERENCE_ENDPOINT_MISMATCH")
    return {"inverse": inverse, "decisions": decisions, "audit": {
        "source_id": str(inverse["source_id"]), "target_smiles": target,
        "input_condition": "main_product_only_v1", "initial_components": 1,
        "first_tool": names[0], "decision_count": len(names),
        "import_decisions": names.count("import_fragments"),
        "import_roles": dict(import_roles), "flow_successors_match_source": flow_index,
        "reference_endpoint_exact": True, "original_moves_preserved": True,
        "model_prefix_parity": True, "final_state": node.state,
        "source_row_sha256": inverse["metadata"]["source_row_sha256"],
        "original_product_components": len(inverse["metadata"]["original_product_side"].split(".")),
        "removed_product_components": len(inverse["metadata"]["removed_product_components"]),
    }}


def validate_coverage(*, original, accepted, unresolved):
    if accepted & unresolved:
        raise ValueError("ACCEPTED_UNRESOLVED_OVERLAP")
    if len(accepted) + len(unresolved) != original:
        raise ValueError("REACTION_DENOMINATOR_CHANGED")


def worker(line):
    source = json.loads(line)
    try:
        signal.setitimer(signal.ITIMER_REAL, 180)
        return True, audit_reference(prepare_reference(source))
    except Exception as exc:
        return False, {
            "source_id": str(source["source_id"]),
            "error": f"{type(exc).__name__}:{exc}",
            "source_row": source,
            "source_row_sha256": hashlib.sha256(
                json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "training_allowed": False, "independent_review_required": True,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=DENOMINATORS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--recover-from", type=Path,
                        help="Read-only interrupted build; verify and copy committed work to a NEW output directory.")
    args = parser.parse_args()
    source = args.source_dir
    manifest_path = source / ("training_manifest.json" if args.track == "flower" else "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads((source / "ARTIFACT_STATUS.json").read_text())
    if status.get("training_allowed") is not True:
        raise ValueError("SOURCE_TRAINING_FORBIDDEN")
    if args.output_dir.exists():
        raise ValueError("REFUSING_TO_OVERWRITE_EXISTING_OUTPUT")
    args.output_dir.mkdir(parents=True)
    code_paths = [Path(__file__), ROOT / "scripts/build_natural_language_event_sft.py",
                  ROOT / "scripts/run_natural_language_value_search.py",
                  ROOT / "scripts/build_mech31k_main_product_only_sft.py",
                  ROOT / "src/mechet/inverse_trace_data.py",
                  ROOT / "src/mechet/in_place_grounded_flow.py",
                  ROOT / "src/mechet/natural_language_electron_flow.py",
                  ROOT / "src/mechet/forward_expert.py"]
    recovery = None
    if args.recover_from:
        verify_recovery_code(args.recover_from, code_paths, ROOT)
        recovery = {"source_directory": str(args.recover_from),
                    "source_progress_sha256": sha(args.recover_from / "progress.jsonl"),
                    "chemical_dependencies_unchanged": True, "original_files_modified": False,
                    "recovered_train_prefix": 0}
    code_paths += [ROOT / "src/mechet/pr96_durable_io.py", ROOT / "scripts/pr96_build_recovery.py"]
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_paths}
    for name in ("inverse", "state_sft", "audits"):
        (args.output_dir / name).mkdir()
    for directory in (args.output_dir, args.output_dir / "state_sft"):
        (directory / "ARTIFACT_STATUS.json").write_text(json.dumps({
            "training_allowed": False, "status": "building_reference_audit",
            "reason": "Incomplete data is never train-ready; requires full replay, independent review and token audit.",
        }, indent=2) + "\n")
    for path in code_paths:
        destination = args.output_dir / "code_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    progress_path = args.output_dir / "progress.jsonl"
    reports, split_ids = {}, {}
    start = time.monotonic()
    with DurableAppendSet({"progress": progress_path}) as progress:
        def emit(**value):
            value.update(timestamp=datetime.now(timezone.utc).isoformat(), track=args.track,
                         elapsed_seconds=round(time.monotonic() - start, 2))
            text = json.dumps(value)
            print(text, flush=True)
            progress.commit({"progress": (text + "\n").encode()})
        def io_event(value):
            emit(**value)
        # This is a GT data-integrity check explicitly required by PR96, not a
        # small-subset model experiment. The complete split audit follows it.
        selected = read_selected(source / "valid.jsonl", 64, 17)
        with DurableAppendSet({"preflight": args.output_dir / "valid_preflight.jsonl"}, on_retry=io_event) as preflight:
            for i, row in enumerate(selected, 1):
                _worker_init()
                ok, value = worker(json.dumps(row))
                preflight.commit({"preflight": encoded({"ok": ok, "result": value})})
                emit(stage="valid-reference-preflight", processed=i, total=len(selected),
                     source_id=str(row["source_id"]), reference_pass=ok, percent=100*i/len(selected))
        for split in ("valid", "test", "train"):
            inp = source / f"{split}.jsonl"
            expected = DENOMINATORS[args.track][split]
            if manifest["splits"][split]["rows"] != expected or sha(inp) != manifest["splits"][split]["sha256"]:
                raise ValueError(f"SOURCE_HASH_OR_DENOMINATOR_MISMATCH:{split}")
            paths = {k: args.output_dir / k / f"{split}.jsonl" for k in ("inverse", "state_sft", "audits")}
            bad_path = args.output_dir / "audits" / f"{split}.unresolved.jsonl"
            accepted, unresolved, counter = set(), set(), Counter()
            if args.recover_from and (args.recover_from / f"{split}.report.json").exists():
                report = copy_completed_split(args.recover_from, split, manifest["splits"][split]["sha256"],
                                              expected, paths, bad_path, io_event)
                accepted, unresolved = set(report["accepted_ids"]), set(report["unresolved_ids"])
                validate_coverage(original=expected, accepted=accepted, unresolved=unresolved)
                split_ids[split], reports[split] = accepted | unresolved, report
                (args.output_dir / f"{split}.report.json").write_text(json.dumps(report, indent=2) + "\n")
                emit(stage="reference-replay", split=split, processed=expected, total=expected,
                     percent=100, reference_pass=len(accepted), unresolved=len(unresolved),
                     counts=report["counts"], reused_completed_split=True)
                continue
            emit(stage="reference-replay", split=split, processed=0, total=expected, percent=0)
            with inp.open() as reader, DurableAppendSet(dict(paths, unresolved=bad_path), on_retry=io_event) as writer:
                if args.recover_from and split == "train":
                    last = None
                    with (args.recover_from / "progress.jsonl").open() as saved_progress:
                        for line in saved_progress:
                            entry = json.loads(line)
                            if entry.get("stage") == "reference-replay" and entry.get("split") == split:
                                last = entry
                    if last and last["processed"]:
                        old_paths = {k: args.recover_from / k / f"{split}.jsonl" for k in paths}
                        old_paths["unresolved"] = args.recover_from / "audits" / f"{split}.unresolved.jsonl"
                        accepted, unresolved, counter = recover_prefix(reader, old_paths, last, writer, io_event)
                        recovery["recovered_train_prefix"] = last["processed"]
                        recovery["recovered_progress_record"] = last
                        emit(stage="reference-replay", split=split, processed=last["processed"], total=expected,
                             percent=100 * last["processed"] / expected, reference_pass=len(accepted),
                             unresolved=len(unresolved), counts=dict(counter),
                             committed_offsets=dict(writer.offsets), recovered_prefix=True)
                pending = {k: bytearray() for k in writer.paths}
                with mp.get_context("fork").Pool(max(1, args.workers), initializer=_worker_init) as pool:
                  for ok, value in pool.imap(worker, reader, chunksize=2):
                    rid = value["audit"]["source_id"] if ok else value["source_id"]
                    if rid in accepted or rid in unresolved:
                        raise ValueError(f"DUPLICATE_SOURCE_ID:{split}:{rid}")
                    if ok:
                        accepted.add(rid)
                    else:
                        unresolved.add(rid)
                    payload, increment = result_payload(ok, value, split)
                    counter.update(increment)
                    for key, raw in payload.items():
                        pending[key].extend(raw)
                    processed = len(accepted) + len(unresolved)
                    if processed % 50 == 0 or processed == expected:
                        offsets = writer.commit({k: bytes(v) for k,v in pending.items()})
                        pending = {k: bytearray() for k in writer.paths}
                        emit(stage="reference-replay", split=split, processed=processed, total=expected,
                             percent=round(100 * processed / expected, 3), reference_pass=len(accepted),
                             unresolved=len(unresolved), counts=dict(counter), committed_offsets=offsets)
            validate_coverage(original=expected, accepted=accepted, unresolved=unresolved)
            split_ids[split] = accepted | unresolved
            reports[split] = {
                "original_count": expected, "reference_pass": len(accepted), "unresolved": len(unresolved),
                "coverage": len(accepted) / expected, "counts": dict(counter), "source_sha256": sha(inp),
                "accepted_ids": sorted(accepted), "unresolved_ids": sorted(unresolved),
                "outputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()},
                "unresolved_path": str(bad_path), "unresolved_sha256": sha(bad_path),
            }
            (args.output_dir / f"{split}.report.json").write_text(json.dumps(reports[split], indent=2) + "\n")
        overlap = {f"{a}_{b}": len(split_ids[a] & split_ids[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
        if any(overlap.values()):
            raise ValueError("SPLIT_ID_OVERLAP")
        if any(sha(p) != code_hashes[str(p.relative_to(ROOT))] for p in code_paths):
            raise ValueError("CODE_CHANGED_DURING_BUILD")
        result = {
            "artifact_type": "main_product_only_source_arrows_v2", "track": args.track,
            "input_condition": "main_product_only_v1", "reports": reports,
            "source_manifest_sha256": sha(manifest_path), "source_manifest": str(manifest_path),
            "split_id_overlap": overlap, "code_sha256": code_hashes,
            "original_moves_preserved": True, "full_source_denominators_accounted": True,
            "training_allowed": False, "status": "awaiting_independent_review_and_token_audit",
        }
        if recovery:
            result["recovery"] = recovery
        (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        for directory in (args.output_dir, args.output_dir / "state_sft"):
            (directory / "ARTIFACT_STATUS.json").write_text(json.dumps({
                "training_allowed": False, "status": result["status"],
                "reason": "PR96 requires separate independent review, quarantine authorization and tokenizer audit.",
            }, indent=2) + "\n")
        emit(stage="reference-build-complete", reports={k: {x:v[x] for x in ("original_count", "reference_pass", "unresolved")} for k,v in reports.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
