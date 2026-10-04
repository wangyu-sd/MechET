#!/usr/bin/env python3
"""Filter existing Qwen3-8B direct/open/closed controls to one H2 ID split.

All three representations originate from the original *training* pool. The
new H2 held-out rows are not copied from official FlowER valid/test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


CONDITIONS = {
    "direct": "canonical_strict_trace_source_id",
    "open_flow": "canonical_source_id",
    "closed_loop": "canonical_source_id",
}
FIXED_TOOL_BUDGET = 40
INITIAL_OBSERVATION_MARKER = "INITIAL ENVIRONMENT OBSERVATION:\n"


def strict_trace_to_direct(row: dict[str, Any]) -> dict[str, Any]:
    """Make an outcome-only control from the *same* frozen trace endpoint.

    The older reaction-level Direct dataset uses a different endpoint policy
    for many IDs, so filtering it by ID is not a matched experiment.
    """
    target = str(row.get("target_smiles") or "")
    structural = str(row.get("structural_precursor") or "")
    if not target or not structural:
        raise ValueError("strict trace row lacks product or structural endpoint")
    metadata = {
        "source_dataset": "flower_inverse_tool_sft_action_delta_v1",
        "source_view": "strict_trace_endpoint_direct_control",
        "source_split": "train",
        "endpoint_policy": "frozen_strict_trace_structural_precursor",
        "endpoint_source": "independent_answer",
        "mechanism_supervision": False,
        "executable_trace": False,
        "assistant_only_loss": True,
        "task_type": "outcome_only_retro",
    }
    return {
        "id": row["id"],
        "source_id": row["source_id"],
        "artifact_type": "supervision",
        "task_type": "outcome_only_retro",
        "target_smiles": target,
        "structural_precursor": structural,
        "expected_precursor": structural,
        "full_precursor_state": row.get("full_precursor_state"),
        "auxiliary_fragments": row.get("auxiliary_fragments", []),
        "messages": [
            {"role": "system", "content": (
                "Given only a mapped product SMILES, predict the frozen "
                "atom-contributing structural precursor endpoint. Output "
                "exactly <answer> followed by the precursor and </answer>."
            )},
            {"role": "user", "content": f"TARGET: {target}"},
            {"role": "assistant", "content": f"<answer>\n{structural}\n</answer>"},
        ],
        "metadata": metadata,
    }


def normalize_closed_loop_budget(row: dict[str, Any]) -> dict[str, Any]:
    """Remove the gold-trace-length-dependent initial tool budget from H2.

    The original trace rows use ``required_calls+2``. That number reveals
    something about the recorded trajectory before product-only inference.
    The H2 condition instead exposes a common 40-call budget, in both the
    initial preamble and teacher-forced tool observations.
    """
    messages = [dict(message) for message in row.get("messages") or []]
    user_messages = [item for item in messages if item.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError("closed-loop row must have one initial user message")
    user = user_messages[0]
    content = str(user.get("content") or "")
    if INITIAL_OBSERVATION_MARKER not in content:
        raise ValueError("closed-loop initial observation marker missing")
    prefix, payload = content.split(INITIAL_OBSERVATION_MARKER, 1)
    observation = json.loads(payload)
    original_budget = int(observation.get("max_tool_calls") or 0)
    if not 1 <= original_budget <= FIXED_TOOL_BUDGET:
        raise ValueError(f"invalid source tool budget: {original_budget}")
    observation["max_tool_calls"] = FIXED_TOOL_BUDGET
    user["content"] = prefix + INITIAL_OBSERVATION_MARKER + json.dumps(observation, ensure_ascii=False)
    call_index = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_index += 1
        result = json.loads(str(message.get("content") or ""))
        if "remaining_tool_calls" in result:
            if int(result["remaining_tool_calls"]) != original_budget - call_index:
                raise ValueError("source tool observation budget/count mismatch")
            result["remaining_tool_calls"] = FIXED_TOOL_BUDGET - call_index
            message["content"] = json.dumps(result, ensure_ascii=False)
    if call_index > FIXED_TOOL_BUDGET:
        raise ValueError("closed-loop trace exceeds fixed tool budget")
    row["messages"] = messages
    metadata = dict(row.get("metadata") or {})
    metadata["nmi_fixed_tool_budget"] = FIXED_TOOL_BUDGET
    metadata["nmi_original_gold_dependent_tool_budget"] = original_budget
    row["metadata"] = metadata
    return row


def _split_ids(split_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = json.loads((split_dir / "manifest.json").read_text())
    assignments = {}
    for split in ("train", "valid", "test"):
        content = (split_dir / f"{split}.ids.txt").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["split_id_sha256"][split]:
            raise ValueError(f"frozen {split} ID hash mismatch")
        ids = content.decode().splitlines()
        if len(ids) != manifest["rows"][split]:
            raise ValueError(f"frozen {split} ID count mismatch")
        for identifier in ids:
            if identifier in assignments:
                raise ValueError(f"reaction assigned to multiple splits: {identifier}")
            assignments[identifier] = split
    return assignments, manifest


def canonical_source_id(row: dict[str, Any], condition: str) -> str:
    source_id = str(row.get("source_id") or "")
    if condition in CONDITIONS:
        return source_id
    raise ValueError(f"unknown condition {condition}")


def materialize(
    source: Path, *, condition: str, source_sha256: str,
    source_rows: int, split_dir: Path, output_dir: Path,
    structural_audit: Path,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    assignments, split_manifest = _split_ids(split_dir)
    audit = json.loads(structural_audit.read_text())
    if audit.get("scope") != "full_frozen_split" or audit.get("artifact_type") != "nmi_mechcomp_structural_overlap_v2":
        raise ValueError("full v2 structural audit is required before materialization")
    if audit.get("split_manifest_sha256") != hashlib.sha256((split_dir / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("structural audit does not match frozen split")
    if not all(audit.get("gates", {}).values()) or audit.get("reaction_center_undefined_count"):
        raise ValueError("structural exact-reaction/center audit did not pass")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    handles = {split: (temporary / f"{split}.jsonl").open("wb") for split in ("train", "valid", "test")}
    digest = hashlib.sha256()
    output_hashes = {split: hashlib.sha256() for split in handles}
    row_counts: Counter[str] = Counter()
    unmatched_ids: list[str] = []
    seen: set[str] = set()
    total = 0
    try:
        with source.open("rb") as handle:
            for line in handle:
                digest.update(line)
                total += 1
                row = json.loads(line)
                identifier = canonical_source_id(row, condition)
                split = assignments.get(identifier)
                if split is None:
                    unmatched_ids.append(str(row.get("source_id") or ""))
                    continue
                if identifier in seen:
                    raise ValueError(f"duplicate selected ID in {condition}: {identifier}")
                seen.add(identifier)
                if condition == "direct":
                    row = strict_trace_to_direct(row)
                metadata = dict(row.get("metadata") or {})
                metadata["nmi_split"] = split
                metadata["nmi_parent_source_id"] = identifier
                row["source_id"] = identifier
                row["metadata"] = metadata
                if condition == "closed_loop":
                    row = normalize_closed_loop_budget(row)
                encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                handles[split].write(encoded)
                output_hashes[split].update(encoded)
                row_counts[split] += 1
        if total != source_rows or digest.hexdigest() != source_sha256:
            raise ValueError(f"{condition} input row-count/SHA contract mismatch")
        if seen != set(assignments):
            raise ValueError(f"{condition} selected stable IDs do not match frozen split")
        expected_unmatched = 0
        if len(unmatched_ids) != expected_unmatched or len(set(unmatched_ids)) != len(unmatched_ids):
            raise ValueError(f"{condition} unexpected out-of-universe source IDs: {unmatched_ids[:10]}")
        if any(row_counts[split] != split_manifest["rows"][split] for split in handles):
            raise ValueError(f"{condition} selected split counts do not match frozen split")
        for handle in handles.values():
            handle.close()
        report = {
            "artifact_type": "nmi_matched_representation_data_v1",
            "condition": condition,
            "source": str(source.resolve()),
            "source_sha256": digest.hexdigest(),
            "source_rows": total,
            "out_of_strict_universe_source_rows": len(unmatched_ids),
            "out_of_strict_universe_source_ids": unmatched_ids,
            "parent_split_manifest_sha256": hashlib.sha256((split_dir / "manifest.json").read_bytes()).hexdigest(),
            "structural_audit_sha256": hashlib.sha256(structural_audit.read_bytes()).hexdigest(),
            "rows": {split: row_counts[split] for split in handles},
            "output_sha256": {split: output_hashes[split].hexdigest() for split in handles},
            "splits": {
                split: {
                    "file": str((output_dir / f"{split}.jsonl").resolve()),
                    "rows": row_counts[split],
                    "sha256": output_hashes[split].hexdigest(),
                } for split in handles
            },
            "tasks": {
                condition: {
                    split: {
                        "path": str((output_dir / f"{split}.jsonl").resolve()),
                        "rows": row_counts[split],
                        "sha256": output_hashes[split].hexdigest(),
                    } for split in handles
                }
            },
            "stable_reaction_ids": split_manifest["split_id_sha256"],
            "training_allowed": True,
            "model_selection_policy": "H2 valid only; H2 held-out test never used for checkpoint choice",
        }
        if condition == "closed_loop":
            report["observation_mode"] = "compact_full_state_v1"
            report["intermediate_state_model_visible"] = True
            report["initial_tool_budget_policy"] = "fixed_40_no_gold_trace_length_leakage"
        if condition == "direct":
            report["endpoint_policy"] = "strict_trace_endpoint_exact_matched"
            report["mechanism_supervision"] = False
        (temporary / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output_dir)
        return report
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-rows", type=int, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        args.source, condition=args.condition,
        source_sha256=args.source_sha256, source_rows=args.source_rows,
        split_dir=args.split_dir, structural_audit=args.structural_audit,
        output_dir=args.output_dir,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
