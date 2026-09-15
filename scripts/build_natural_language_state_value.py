#!/usr/bin/env python3
"""Build executor-grounded state-value supervision for electron retrosynthesis.

Labels are deliberately one-token decisions:

* ``A`` -- a reference-prefix state that should continue;
* ``B`` -- the reference precursor endpoint, where the policy should finish;
* ``C`` -- an executable counterfactual state whose frozen reference
  continuation no longer reaches the precursor endpoint.  This is a
  lower-value/off-reference label, not a proof that no alternative mechanism
  can ever reach the endpoint.

The counterfactuals change one public temporary atom choice in a reference
electron event, execute it, and replay the untouched reference suffix.  They
therefore target the observed failure mode: chemically legal local decisions
that have poor long-horizon value.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.a7_rescue import stable_sample_key
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    extract_import_fragments,
    mapped_atom_numbers,
    mapped_state_signature,
    retain_mapped_components,
    schedule_imports,
)
from mechet.natural_language_electron_flow import build_inventory, compile_event_arguments
from scripts.build_natural_language_event_sft import convert_row


VERSION = "natural_language_state_value_v2"
VALUE_SYSTEM = (
    "You are the MechET state-value critic. Judge a retrosynthetic electronic "
    "state relative to its target product. Reply with exactly one label: A if "
    "the state is a productive nonterminal state and electron reasoning should "
    "continue; B if it is a complete precursor endpoint and should finish; C if "
    "it is an executable but lower-value or off-reference state. Do not explain "
    "the label."
)
ALIAS_RE = re.compile(r"\bA\d{2,}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _visible(mapped: str) -> str:
    return deterministic_unmapped_state(mapped).text


def value_prompt(target: str, state: str) -> str:
    return (
        f"TARGET PRODUCT SMILES: {target}\n"
        f"CANDIDATE CURRENT STATE SMILES: {_visible(state)}\n\n"
        "Return A, B, or C."
    )


def value_row(
    *, reaction_id: str, key: str, target: str, state: str, label: str, provenance: str
) -> dict[str, Any]:
    if label not in {"A", "B", "C"}:
        raise ValueError(f"invalid value label: {label}")
    return {
        "id": f"{reaction_id}::value::{key}",
        "source_id": reaction_id,
        "artifact_type": "supervision",
        "task_type": VERSION,
        "messages": [
            {"role": "system", "content": VALUE_SYSTEM},
            {"role": "user", "content": value_prompt(target, state)},
            {"role": "assistant", "content": label},
        ],
        "tools": [],
        "metadata": {
            "label": label,
            "provenance": provenance,
            "model_visible_atom_maps": False,
            "endpoint_used_as_model_input": False,
        },
    }


def _replace_alias(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return re.sub(rf"\b{re.escape(old)}\b", new, value)
    if isinstance(value, list):
        return [_replace_alias(item, old, new) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _replace_alias(item, old, new) for key, item in value.items()}
    return value


def _reference_suffix_reaches_endpoint(
    state: str,
    *,
    next_event: int,
    steps: Sequence[Mapping[str, Any]],
    scheduled: Sequence[Sequence[str]],
    expected: str,
) -> bool:
    current = state
    try:
        for index in range(next_event, len(steps)):
            current = append_mapped_fragments_verbatim(current, scheduled[index])
            replay = verify_electron_step(current, steps[index]["moves"])
            if not replay.get("ok"):
                return False
            current = str(replay["state_smiles"])
        return _visible(current) == _visible(expected)
    except Exception:
        return False


def counterfactual_state(
    event_state: str,
    gold_arguments: Mapping[str, Any],
    reference_successor: str,
    *,
    next_event: int,
    steps: Sequence[Mapping[str, Any]],
    scheduled: Sequence[Sequence[str]],
    expected: str,
    seed_key: str,
    forbidden_visible_states: set[str],
) -> str | None:
    aliases = sorted(build_inventory(event_state).atom_to_map)
    used = sorted(set(ALIAS_RE.findall(json.dumps(gold_arguments, sort_keys=True))))
    replacements = [
        (old, new)
        for old in used
        for new in aliases
        if new != old
    ]
    replacements.sort(key=lambda pair: stable_sample_key(f"{seed_key}:{pair}", 17))
    for old, new in replacements[:96]:
        arguments = _replace_alias(dict(gold_arguments), old, new)
        try:
            moves = compile_event_arguments(event_state, arguments)
            replay = verify_electron_step(event_state, moves)
            if not replay.get("ok"):
                continue
            candidate = str(replay["state_smiles"])
            # The model never sees private atom maps.  A symmetry-related map
            # permutation is therefore the *same input*, not a negative.
            if _visible(candidate) in forbidden_visible_states:
                continue
            if mapped_state_signature(candidate) == mapped_state_signature(
                reference_successor
            ):
                continue
            if _reference_suffix_reaches_endpoint(
                candidate,
                next_event=next_event,
                steps=steps,
                scheduled=scheduled,
                expected=expected,
            ):
                continue
            return candidate
        except Exception:
            continue
    return None


def convert_reaction(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = dict(row.get("metadata") or {})
    steps = [dict(value) for value in (metadata.get("trace_plan") or {}).get("steps") or []]
    if not steps:
        raise ValueError("missing trace steps")
    scheduled = schedule_imports(extract_import_fragments(row), steps)
    public = convert_row(row)
    event_public = [
        value for value in public if value["metadata"]["decision_type"] == "event"
    ]
    if len(event_public) != len(steps):
        raise ValueError("public/private event mismatch")
    reaction_id = str(row["source_id"])
    target_maps = set(mapped_atom_numbers(str(row["target_smiles"])))
    present_maps = set(target_maps)
    current = retain_mapped_components(str(steps[0]["state_before"]), present_maps)
    target = _visible(current)
    expected = str(row.get("full_precursor_state") or row["expected_precursor"])
    # Build the complete set of model-visible successful-prefix states before
    # mining negatives.  This excludes symmetry-equivalent states as well as a
    # counterfactual that happens to coincide with any later successful state.
    positive_visible_states = {_visible(current)}
    probe = current
    probe_maps = set(target_maps)
    for step, imports in zip(steps, scheduled, strict=True):
        event_state = append_mapped_fragments_verbatim(probe, imports)
        if imports:
            positive_visible_states.add(_visible(event_state))
        for fragment in imports:
            probe_maps.update(mapped_atom_numbers(fragment))
        probe = retain_mapped_components(str(step["state_after"]), probe_maps)
        positive_visible_states.add(_visible(probe))
    output = [
        value_row(
            reaction_id=reaction_id,
            key="initial_A",
            target=target,
            state=current,
            label="A",
            provenance="reference_prefix",
        )
    ]
    for index, (step, imports) in enumerate(zip(steps, scheduled, strict=True)):
        event_state = append_mapped_fragments_verbatim(current, imports)
        for fragment in imports:
            present_maps.update(mapped_atom_numbers(fragment))
        if imports:
            output.append(
                value_row(
                    reaction_id=reaction_id,
                    key=f"event_{index:03d}_imported_A",
                    target=target,
                    state=event_state,
                    label="A",
                    provenance="reference_prefix_after_import",
                )
            )
        successor = retain_mapped_components(str(step["state_after"]), present_maps)
        label = "B" if index + 1 == len(steps) else "A"
        output.append(
            value_row(
                reaction_id=reaction_id,
                key=f"event_{index:03d}_{label}",
                target=target,
                state=successor,
                label=label,
                provenance="reference_endpoint" if label == "B" else "reference_prefix",
            )
        )
        gold_arguments = event_public[index]["messages"][2]["tool_calls"][0]["function"]
        gold_arguments = dict(gold_arguments["arguments"])
        negative = counterfactual_state(
            event_state,
            gold_arguments,
            successor,
            next_event=index + 1,
            steps=steps,
            scheduled=scheduled,
            expected=expected,
            seed_key=f"{reaction_id}:{index}",
            forbidden_visible_states=positive_visible_states,
        )
        if negative is not None:
            output.append(
                value_row(
                    reaction_id=reaction_id,
                    key=f"event_{index:03d}_counterfactual_C",
                    target=target,
                    state=negative,
                    label="C",
                    provenance="executable_counterfactual_reference_suffix_failed",
                )
            )
        current = successor
    return output


def remove_model_visible_label_conflicts(path: Path) -> dict[str, int]:
    """Drop only C rows that contradict A/B for exactly the same model input.

    A/B conflicts are never resolved heuristically because they indicate a
    broken reference trace.  The second pass is global across reaction IDs so
    duplicated chemistry cannot reintroduce contradictory supervision.
    """

    labels_by_prompt: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = json.dumps(row["messages"][:2], sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(prompt.encode()).hexdigest()
            labels_by_prompt.setdefault(key, set()).add(str(row["metadata"]["label"]))
    conflicts = {key: labels for key, labels in labels_by_prompt.items() if len(labels) > 1}
    hard = {key: labels for key, labels in conflicts.items() if {"A", "B"} <= labels}
    if hard:
        raise ValueError(f"A/B model-visible label conflicts: {len(hard)}")

    temporary = path.with_suffix(path.suffix + ".conflict-filter.tmp")
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row["metadata"]["label"])
            prompt = json.dumps(row["messages"][:2], sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if key in conflicts and label == "C":
                counts["dropped_conflicting_counterfactuals"] += 1
                continue
            sink.write(line)
            counts[f"label_{label}"] += 1
            counts["rows"] += 1
    temporary.replace(path)

    # Fail closed if the cleanup did not achieve the actual model contract.
    final_labels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = json.dumps(row["messages"][:2], sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(prompt.encode()).hexdigest()
            final_labels.setdefault(key, set()).add(str(row["metadata"]["label"]))
    counts["model_visible_conflicts_after"] = sum(
        len(labels) > 1 for labels in final_labels.values()
    )
    if counts["model_visible_conflicts_after"]:
        raise ValueError("model-visible label conflicts remain after cleanup")
    return dict(counts)


def _worker_init() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    def timeout(*_: Any) -> None:
        raise TimeoutError("VALUE_CONVERSION_TIMEOUT")

    signal.signal(signal.SIGALRM, timeout)


def _worker(line: str) -> tuple[bool, Any]:
    row = json.loads(line)
    try:
        signal.setitimer(signal.ITIMER_REAL, 180)
        return True, convert_reaction(row)
    except Exception as exc:
        return False, {
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "exception": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _selected_lines(path: Path, limit: int, seed: int) -> list[str]:
    """Select the lowest stable hashes with O(limit) rather than O(dataset) RAM."""

    if limit < 1:
        raise ValueError("reaction limit must be positive")
    selected: list[tuple[int, int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            key = int(stable_sample_key(str(row["id"]), seed), 16)
            item = (-key, -line_number, line.rstrip("\n"))
            if len(selected) < limit:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
    selected.sort(key=lambda item: (-item[0], -item[1]))
    return [item[2] for item in selected]


def build_split(
    source: Path, target: Path, *, reactions: int, workers: int, seed: int
) -> dict[str, Any]:
    lines = _selected_lines(source, reactions, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    unresolved = target.with_suffix(".unresolved.jsonl")
    counts: Counter[str] = Counter()
    started = time.monotonic()
    with (
        target.open("w", encoding="utf-8") as out,
        unresolved.open("w", encoding="utf-8") as bad,
        mp.get_context("fork").Pool(max(1, workers), initializer=_worker_init) as pool,
    ):
        for number, (ok, value) in enumerate(pool.imap(_worker, lines, chunksize=2), 1):
            if not ok:
                bad.write(json.dumps(value, ensure_ascii=False) + "\n")
                counts["unresolved_reactions"] += 1
            else:
                for item in value:
                    label = str(item["metadata"]["label"])
                    counts[f"label_{label}"] += 1
                    out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            if number % 100 == 0:
                out.flush()
                bad.flush()
                print(
                    json.dumps(
                        {
                            "stage": "value-data",
                            "split": source.stem,
                            "reactions": number,
                            "rows": sum(counts[f"label_{x}"] for x in "ABC"),
                            "unresolved": counts["unresolved_reactions"],
                            "seconds": round(time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
    visible_audit = remove_model_visible_label_conflicts(target)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "selected_reactions": len(lines),
        **dict(counts),
        **visible_audit,
        "output_sha256": sha256(target),
        "unresolved_sha256": sha256(unresolved),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-reactions", type=int, default=20000)
    parser.add_argument("--valid-reactions", type=int, default=512)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 64))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    reports = {
        "train": build_split(
            args.source_dir / "train.jsonl",
            args.output_dir / "train.jsonl",
            reactions=args.train_reactions,
            workers=args.workers,
            seed=args.seed,
        ),
        "valid": build_split(
            args.source_dir / "valid.jsonl",
            args.output_dir / "valid.jsonl",
            reactions=args.valid_reactions,
            workers=args.workers,
            seed=args.seed,
        ),
    }
    unresolved = sum(int(value.get("unresolved_reactions", 0)) for value in reports.values())
    visible_conflicts = sum(
        int(value.get("model_visible_conflicts_after", 0)) for value in reports.values()
    )
    manifest = {
        "artifact_type": VERSION,
        "status": "validated_pilot" if unresolved == 0 and visible_conflicts == 0 else "invalid",
        "training_allowed": unresolved == 0 and visible_conflicts == 0,
        "source_artifact": str(args.source_dir),
        "seed": args.seed,
        "reports": reports,
        "labels": {
            "A": "reference_consistent_continue",
            "B": "reference_endpoint_finish",
            "C": "lower_value_off_reference_counterfactual",
        },
        "counterfactual_contract": "model-visible-novel executable alias corruption then frozen reference suffix replay",
        "counterfactual_semantics": "lower_value_off_reference_not_proven_globally_dead",
        "model_visible_atom_maps": False,
        "reference_endpoint_model_visible": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps(
            {
                "artifact_id": VERSION,
                "status": manifest["status"],
                "training_allowed": manifest["training_allowed"],
                "reason": "bounded value-policy improvement pilot",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest), flush=True)
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
