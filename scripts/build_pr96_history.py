#!/usr/bin/env python3
"""Build history from corrected decisions and independently replay saved rows.

Accepted-prefix history is checked against the actual online executor, not
against private source step states. This never trains a model or enables data.
"""
import argparse
from collections import Counter
from itertools import groupby, zip_longest
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.in_place_grounded_flow import mapped_atom_numbers
from scripts.build_mech31k_main_product_only_sft import same_chemical_state, sha
from scripts.build_natural_language_event_sft import convert_row, SYSTEM, TOOLS, _worker_init
from scripts.build_natural_language_history_sft import transform_rows
from scripts.run_natural_language_value_search import (
    Action, Node, execute, policy_prompt, product_only_private_state,
)


def reaction_history(rows, reference=None):
    rows = list(rows)
    if not rows:
        raise ValueError("EMPTY_REACTION")
    if reference is not None:
        expected = convert_row(reference, preserve_source_moves=True)
        if len(expected) != len(rows):
            raise ValueError("SOURCE_DECISION_COUNT_MISMATCH")
        for a,b in zip(rows, expected):
            if a["messages"] != b["messages"] or a["target_smiles"] != b["target_smiles"]:
                raise ValueError("ORIGINAL_ELECTRON_ACTION_OR_STATE_CHANGED")
    target = rows[0]["target_smiles"]
    if "." in target:
        raise ValueError("MULTI_COMPONENT_PUBLIC_ROOT")
    state = product_only_private_state(target)
    node = Node(target=target, state=state, next_map=max(mapped_atom_numbers(state)) + 1,
                visited={target})
    output = list(transform_rows(rows))
    for original, converted in zip(rows, output):
        if original["messages"][0] != {"role": "system", "content": SYSTEM} or original["tools"] != TOOLS:
            raise ValueError("SYSTEM_OR_TOOL_SCHEMA_MISMATCH")
        state_prefix = policy_prompt(target, node.state, include_inventory=True,
                                     actions=node.actions, compact_history=False)
        if original["messages"][1]["content"] != state_prefix:
            raise ValueError("STATE_PREFIX_NOT_REACHABLE_FROM_PUBLIC_ROOT")
        history_prefix = policy_prompt(target, node.state, include_inventory=True,
                                       actions=node.actions, compact_history=True)
        if converted["messages"][1]["content"] != history_prefix:
            raise ValueError("ONLINE_HISTORY_PREFIX_MISMATCH")
        fn = original["messages"][2]["tool_calls"][0]["function"]
        child, error = execute(node, Action(fn["name"], fn["arguments"], "", 0, 0),
                               max_imports=32, reject_target_retained_finish=False)
        if child is None:
            raise ValueError(f"SAVED_ACTION_NOT_EXECUTABLE:{error}")
        result = json.loads(original["messages"][3]["content"])
        wanted = result.get("current_state", result.get("derived_precursor"))
        if not wanted or not same_chemical_state(child.state, wanted):
            raise ValueError("SAVED_SUCCESSOR_CHANGED")
        converted["metadata"]["online_history_prefix_parity"] = True
        node = child
    if not node.terminal or not same_chemical_state(node.state, rows[-1]["expected_precursor"]):
        raise ValueError("SAVED_REFERENCE_ENDPOINT_CHANGED")
    return output


def _history_worker(job):
    rows, reference = job
    try:
        return reaction_history(rows, reference)
    except Exception as exc:
        raise ValueError(f"INDEPENDENT_HISTORY_REPLAY_FAILED:{reference['source_id']}:{exc}") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    source_manifest = args.data_dir / "manifest.json"
    data = json.loads(source_manifest.read_text())
    if data.get("input_condition") != "main_product_only_v1" or not data.get("full_source_denominators_accounted"):
        raise ValueError("NOT_A_COMPLETED_MAIN_ONLY_REFERENCE_AUDIT")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "training_allowed": False, "status": "building_independent_history_audit"}) + "\n")
    reports = {}
    started = time.monotonic()
    with (args.output_dir / "progress.jsonl").open("x") as progress:
        for split in ("valid", "test", "train"):
            record = data["reports"][split]
            state_path = args.data_dir / "state_sft" / f"{split}.jsonl"
            inverse_path = args.data_dir / "inverse" / f"{split}.jsonl"
            for kind, path in (("state_sft", state_path), ("inverse", inverse_path)):
                if sha(path) != record["outputs"][kind]["sha256"]:
                    raise ValueError("REFERENCE_DATA_HASH_MISMATCH")
            target = args.output_dir / f"{split}.jsonl"
            counts = Counter()
            with state_path.open() as state_in, inverse_path.open() as reference_in, target.open("x") as sink, mp.get_context("fork").Pool(max(1, args.workers), initializer=_worker_init) as pool:
                groups = groupby((json.loads(l) for l in state_in), lambda r: str(r["source_id"]))
                def jobs():
                    for group, line in zip_longest(groups, reference_in):
                        if group is None or line is None:
                            raise ValueError("REFERENCE_STATE_GROUP_COUNT_MISMATCH")
                        rid, rows = group
                        reference = json.loads(line)
                        if rid != str(reference["source_id"]):
                            raise ValueError("REFERENCE_STATE_ID_MISMATCH")
                        yield list(rows), reference
                for history in pool.imap(_history_worker, jobs(), chunksize=2):
                    for row in history:
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[row["metadata"]["decision_type"]] += 1
                    counts["reactions"] += 1
                    if counts["reactions"] % 50 == 0 or counts["reactions"] == record["reference_pass"]:
                        sink.flush()
                        event = {"stage": "independent-history-replay", "track": data["track"], "split": split,
                                 "processed": counts["reactions"], "total": record["reference_pass"],
                                 "percent": 100 * counts["reactions"] / record["reference_pass"],
                                 "elapsed_seconds": time.monotonic() - started}
                        text = json.dumps(event)
                        print(text, flush=True)
                        progress.write(text + "\n")
                        progress.flush()
            if counts["reactions"] != record["reference_pass"]:
                raise ValueError("REFERENCE_HISTORY_DENOMINATOR_CHANGED")
            reports[split] = {"original_count": record["original_count"], "reactions": counts["reactions"],
                              "unresolved": record["unresolved"], "counts": dict(counts),
                              "path": str(target), "sha256": sha(target),
                              "source_state_sha256": sha(state_path), "all_online_prefixes_match": True}
            (args.output_dir / f"{split}.report.json").write_text(json.dumps(reports[split], indent=2) + "\n")
    result = {"artifact_type": "pr96_main_only_accepted_history_v1", "track": data["track"],
              "source_reference_manifest": str(source_manifest), "source_manifest_sha256": sha(source_manifest),
              "reports": reports, "accepted_subset_independently_replayed": True,
              "history_contains_failed_actions": False, "history_contains_future": False,
              "training_allowed": False, "status": "awaiting_quarantine_review_and_token_audit"}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "training_allowed": False, "status": result["status"]}) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
