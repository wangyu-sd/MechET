#!/usr/bin/env python3
"""Freeze deterministic decision-row smoke samples without test-driven resampling.

The unit of training budget is a State-SFT *decision row*. Stratification and
contamination exclusion are reaction-level. Scientific freeze requires all
evaluation-source manifests to be present before selecting a training row.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterator

import yaml


EVALUATION_ROW_BOUNDS = {
    "r1_multi_reference": (200, 300),
    "r2_plausibility": (800, 800),
    "r3_corruptions": (288, 288),
    "r4_pmechdb_challenging": (1, None),
    "r4_pmechrp_pathways": (350, 350),
    "r4_literature_cycles": (12, 20),
    "r5_external_predictions": (200, 200),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_hash(*parts: object) -> int:
    encoded = "\0".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")


def resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def product_key(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for reaction-level decontamination") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid product SMILES: {smiles}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def row_reaction_id(row: dict[str, Any]) -> str:
    value = row.get("metadata", {}).get("reaction_id") or row.get("source_id")
    if not value:
        raise ValueError(f"missing reaction ID for {row.get('id')}")
    return str(value)


def grouped_decisions(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    seen: set[str] = set()
    current: str | None = None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            reaction_id = row_reaction_id(row)
            if current is not None and reaction_id != current:
                yield current, rows
                seen.add(current)
                rows = []
            if reaction_id in seen:
                raise ValueError(f"noncontiguous reaction {reaction_id} in {path}:{line_no}")
            current = reaction_id
            rows.append(row)
    if current is not None:
        yield current, rows


def bucket_length(events: int) -> str:
    if events <= 1:
        return "Q1"
    if events == 2:
        return "Q2"
    if events <= 4:
        return "Q3"
    return "Q4"


def group_stratum(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    event_moves: list[int] = []
    for row in rows:
        if row.get("metadata", {}).get("decision_type") != "event":
            continue
        calls = [call for message in row.get("messages", []) for call in message.get("tool_calls", [])]
        flows = [call.get("function", {}).get("arguments", {}).get("electron_flow", [])
                 for call in calls if call.get("function", {}).get("name") == "apply_electron_flow"]
        if len(flows) != 1:
            raise ValueError(f"expected one electron event in {row.get('id')}")
        event_moves.append(len(flows[0]))
    coordination = max(event_moves, default=0)
    coordination_bucket = "1" if coordination <= 1 else "2" if coordination == 2 else "3+"
    metadata = rows[0].get("metadata", {})
    # These labels are not invented from product frequency or SMILES length.
    return (
        bucket_length(len(event_moves)), coordination_bucket,
        str(metadata.get("structural_novelty_bucket") or "unavailable"),
        str(metadata.get("reaction_frequency_bucket") or "unavailable"),
        str(metadata.get("mechanism_class") or "unavailable"),
    )


def allocate(counts: dict[tuple[str, ...], int], quota: int) -> dict[tuple[str, ...], int]:
    """Minimum-per-cell then proportional fill, with deterministic tie breaks."""
    if quota < 0:
        raise ValueError("negative quota")
    cells = sorted(cell for cell, count in counts.items() if count)
    allocation = {cell: 0 for cell in cells}
    for cell in cells[:quota]:
        allocation[cell] = 1
    remaining = min(quota, sum(counts.values())) - sum(allocation.values())
    while remaining:
        capacity = {cell: counts[cell] - allocation[cell] for cell in cells}
        total = sum(capacity.values())
        if not total:
            break
        portions = {cell: remaining * capacity[cell] / total for cell in cells}
        assigned = 0
        for cell in cells:
            extra = min(capacity[cell], int(portions[cell]))
            allocation[cell] += extra
            assigned += extra
        remaining -= assigned
        if remaining:
            order = sorted(cells, key=lambda cell: (-(portions[cell] % 1), cell))
            for cell in order:
                if remaining == 0:
                    break
                if allocation[cell] < counts[cell]:
                    allocation[cell] += 1
                    remaining -= 1
    return allocation


def eval_exclusions(config: dict[str, Any], root: Path) -> tuple[set[str], set[str], dict[str, str]]:
    ids: set[str] = set()
    products: set[str] = set()
    hashes: dict[str, str] = {}
    for name, raw in config["evaluation_sources"].items():
        path = resolve(root, raw)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"scientific freeze requires evaluation source {name}: {raw}")
        hashes[name] = verify_evaluation_source(path, name=name)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                reaction = row.get("reaction_id") or row.get("source_id")
                product = row.get("product_smiles") or row.get("target_smiles")
                if not product:
                    raise ValueError(f"{name} evaluation row lacks product SMILES")
                if reaction:
                    ids.add(str(reaction))
                products.add(product_key(str(product)))
    return ids, products, hashes


def verify_evaluation_source(path: Path, *, name: str | None = None) -> str:
    """Require a frozen cohort manifest; reject superseded or drifted sources."""
    source_hash = digest(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"evaluation source lacks a frozen cohort manifest: {path}")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("cohort_sha256")
    if not expected or source_hash != expected:
        raise ValueError(f"evaluation source hash differs from {manifest_path}")
    status_path = path.parent / "ARTIFACT_STATUS.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("evaluation_allowed") is not True:
            raise ValueError(f"evaluation source is forbidden by {status_path}")
        expected = status.get("cohort_sha256")
        if expected and source_hash != expected:
            raise ValueError(f"evaluation source hash differs from {status_path}")
    if name in EVALUATION_ROW_BOUNDS:
        minimum, maximum = EVALUATION_ROW_BOUNDS[name]
        with path.open(encoding="utf-8") as stream:
            rows = sum(bool(line.strip()) for line in stream)
        if rows < minimum or (maximum is not None and rows > maximum):
            raise ValueError(f"{name} evaluation cohort has {rows} rows; expected {minimum}"
                             + (f"–{maximum}" if maximum != minimum and maximum is not None else ""))
    return source_hash


def candidates(
    path: Path, source: str, seed: int, maximum: int,
    excluded_ids: set[str], excluded_products: set[str],
) -> tuple[dict[tuple[str, ...], int], dict[tuple[str, ...], list[tuple[int, str]]], dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    heaps: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
    excluded = 0
    reactions = 0
    unavailable: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for reaction_id, rows in grouped_decisions(path):
        reactions += 1
        if reactions % 50000 == 0:
            print(json.dumps({"stage": "scan", "source": source,
                              "reactions": reactions, "decision_rows": len(seen_ids)}),
                  flush=True)
        product = str(rows[0].get("target_smiles") or "")
        if not product:
            raise ValueError(f"{source}/{reaction_id} lacks target SMILES")
        if reaction_id in excluded_ids or (excluded_products and product_key(product) in excluded_products):
            excluded += len(rows)
            continue
        stratum = group_stratum(rows)
        for index, label in enumerate(("novelty", "frequency", "class"), 2):
            if stratum[index] == "unavailable":
                unavailable[label] += len(rows)
        for row in rows:
            if not bool(row.get("metadata", {}).get("executor_replayed")):
                raise ValueError(f"unreplayed State-SFT decision {row.get('id')}")
            if source == "curated":
                provenance = row.get("metadata", {})
                required = ("source_record_id", "source_license", "mechanism_class")
                missing = [key for key in required if not provenance.get(key)]
                if missing:
                    raise ValueError(f"curated row {row.get('id')} lacks provenance {missing}")
            row_id = str(row["id"])
            if row_id in seen_ids:
                raise ValueError(f"duplicate decision ID: {row_id}")
            seen_ids.add(row_id)
            counts[stratum] += 1
            priority = stable_hash(seed, source, row_id)
            heap = heaps[stratum]
            item = (-priority, row_id)
            if len(heap) < maximum:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return dict(counts), heaps, {"reactions": reactions, "excluded_rows": excluded,
                                  "unavailable_rows": dict(unavailable)}


def selected_ids(
    counts: dict[tuple[str, ...], int], heaps: dict[tuple[str, ...], list[tuple[int, str]]],
    quota: int, *, class_index: int | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    if class_index is None:
        allocation = allocate(counts, quota)
        class_quotas: dict[str, int] | None = None
    else:
        # First allocate by mechanism family and cap each family at half the
        # curated budget. Then preserve minimum-per-cell/proportional sampling
        # *within* each family. A post-hoc majority rejection would discard an
        # otherwise usable, replayed source without constructing the matched
        # scientific smoke that the frozen PR explicitly requests.
        available = sum(counts.values())
        target = min(quota, available)
        maximum = target // 2
        class_counts: Counter[tuple[str, ...]] = Counter()
        for cell, count in counts.items():
            class_counts[(cell[class_index],)] += count
        grouped = allocate(dict(class_counts), target)
        while any(number > maximum for number in grouped.values()):
            excess = 0
            for family in sorted(grouped):
                if grouped[family] > maximum:
                    excess += grouped[family] - maximum
                    grouped[family] = maximum
            capacity = {
                family: min(class_counts[family], maximum) - grouped[family]
                for family in grouped
            }
            if sum(capacity.values()) < excess:
                raise ValueError("curated source cannot satisfy no-majority-class quota")
            extra = allocate(capacity, excess)
            for family, number in extra.items():
                grouped[family] += number
        allocation = {}
        for family, family_quota in grouped.items():
            subcounts = {cell: count for cell, count in counts.items()
                         if cell[class_index] == family[0]}
            allocation.update(allocate(subcounts, family_quota))
        class_quotas = {family[0]: number for family, number in grouped.items()}
    selected: dict[str, tuple[str, ...]] = {}
    for cell, number in allocation.items():
        ranked = sorted(heaps[cell], key=lambda item: (-item[0], item[1]))
        for _, row_id in ranked[:number]:
            selected[row_id] = cell
    report = {
        "requested": quota, "selected": len(selected), "underfilled": quota - len(selected),
        "cells": [{"stratum": cell, "available": counts[cell], "selected": allocation[cell],
                   "underfilled": max(0, allocation[cell] - counts[cell])}
                  for cell in sorted(allocation)],
    }
    if class_quotas is not None:
        report["mechanism_class_quotas"] = dict(sorted(class_quotas.items()))
    return selected, report


def class_balanced_capacity(
    counts: dict[tuple[str, ...], int], maximum: int, *, class_index: int,
) -> int:
    """Largest no-majority-class budget available after eval exclusions."""

    families: Counter[str] = Counter()
    for cell, number in counts.items():
        families[cell[class_index]] += number
    if len(families) < 2:
        return 0
    for quota in range(min(maximum, sum(families.values())), 1, -1):
        cap = quota // 2
        if sum(min(number, cap) for number in families.values()) >= quota:
            return quota
    return 0


def freeze(config: dict[str, Any], root: Path, output: Path, *, engineering: bool) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"frozen output already exists: {output}")
    excluded_ids: set[str] = set()
    excluded_products: set[str] = set()
    eval_hashes: dict[str, str] = {}
    if not engineering:
        excluded_ids, excluded_products, eval_hashes = eval_exclusions(config, root)
    seed = int(config["seed"])
    sources = config["sources"]
    source_paths = {name: resolve(root, spec.get("train")) for name, spec in sources.items()}
    if not all(source_paths[name] and source_paths[name].is_file()
               for name in ("flower", "mech_uspto_31k")):
        raise FileNotFoundError("both base State-SFT source files are required")
    curated = source_paths.get("curated")
    if curated is not None and not curated.is_file():
        raise FileNotFoundError(f"configured curated source does not exist: {curated}")
    quotas = ({"engineering": {"flower": 16, "mech_uspto_31k": 16}}
              if engineering else {key: dict(value) for key, value in
                                   (("base", config["scientific_smoke"]["base"]),
                                    ("mech", config["scientific_smoke"]["mech"]))})
    # The curated shortfall is measured before any source selection.
    curated_available = 0
    if curated and not engineering:
        curated_counts, _, _ = candidates(curated, "curated", seed, 3000,
                                           excluded_ids, excluded_products)
        curated_available = class_balanced_capacity(curated_counts, 3000, class_index=4)
    if not engineering:
        target = int(quotas["mech"]["curated"])
        shortfall = max(0, target - curated_available)
        quotas["mech"]["curated"] = min(target, curated_available)
        quotas["mech"]["flower"] += (shortfall + 1) // 2
        quotas["mech"]["mech_uspto_31k"] += shortfall // 2
    source_results: dict[str, Any] = {}
    selected_by_condition: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    for source, path in source_paths.items():
        needed = {condition: int(parts.get(source, 0)) for condition, parts in quotas.items()}
        if max(needed.values(), default=0) == 0:
            continue
        assert path is not None
        status_path = path.parent / "ARTIFACT_STATUS.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if not bool(status.get("training_allowed")):
                raise ValueError(f"{source} artifact explicitly forbids training")
        elif source == "curated":
            raise ValueError("curated source needs an ARTIFACT_STATUS.json replay audit")
        counts, heaps, audit = candidates(path, source, seed, max(needed.values()),
                                           excluded_ids, excluded_products)
        source_hash = digest(path)
        sidecar = resolve(root, sources[source].get("manifest"))
        if source == "curated" and sidecar is None:
            raise ValueError("curated source needs a frozen source manifest")
        if sidecar is not None:
            if not sidecar.is_file():
                raise FileNotFoundError(f"source manifest missing: {sidecar}")
            source_manifest = json.loads(sidecar.read_text())
            train_split = (source_manifest.get("splits") or {}).get("train") or {}
            expected_hash = train_split.get("output_sha256") or train_split.get("sha256")
            if expected_hash and source_hash != expected_hash:
                raise ValueError(f"{source} train file differs from its frozen source manifest")
        source_results[source] = {"path": str(path), "sha256": source_hash,
                                  "source_manifest": str(sidecar) if sidecar else None,
                                  "source_manifest_sha256": digest(sidecar) if sidecar else None,
                                  **audit}
        for condition, requested in needed.items():
            if requested:
                picked, report = selected_ids(
                    counts, heaps, requested,
                    class_index=4 if source == "curated" else None,
                )
                if report["underfilled"]:
                    raise ValueError(f"{source}/{condition} cannot meet fixed quota: {report}")
                if source == "curated" and picked:
                    by_class = Counter(cell[4] for cell in picked.values())
                    if max(by_class.values()) > len(picked) // 2:
                        raise ValueError("curated sample is dominated by one mechanism class")
                selected_by_condition.setdefault(condition, {})[source] = picked
                source_results[source][condition] = report
    output.mkdir(parents=True)
    manifests = output / "manifests"
    manifests.mkdir()
    files: dict[str, Any] = {}
    for condition, by_source in selected_by_condition.items():
        selection_payload = [
            [source, row_id, cell]
            for source in sorted(by_source)
            for row_id, cell in sorted(by_source[source].items())
        ]
        selection_hash = hashlib.sha256(json.dumps(
            selection_payload, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        selected_path = manifests / f"{condition}_state_sft.jsonl"
        strata_path = manifests / f"{condition}_strata.jsonl"
        with selected_path.open("w", encoding="utf-8") as selected_stream, \
             strata_path.open("w", encoding="utf-8") as strata_stream:
            for source in sorted(by_source):
                ids = by_source[source]
                with source_paths[source].open(encoding="utf-8") as input_stream:
                    for line in input_stream:
                        row = json.loads(line)
                        row_id = str(row["id"])
                        if row_id in ids:
                            selected_stream.write(line)
                            strata_stream.write(json.dumps({
                                "stable_id": row_id, "reaction_id": row_reaction_id(row),
                                "source": source, "split": "train", "stratum": ids[row_id],
                                "inclusion_reason": "deterministic_min_cell_proportional_fill",
                                "selection_manifest_sha256": selection_hash,
                            }, sort_keys=True) + "\n")
        files[condition] = {"train": str(selected_path), "train_sha256": digest(selected_path),
                            "strata": str(strata_path), "strata_sha256": digest(strata_path),
                            "selection_manifest_sha256": selection_hash,
                            "rows": sum(map(len, by_source.values()))}
    result = {"artifact_type": "autoresearch_frozen_state_sft_manifest_v1",
              "campaign_id": config["campaign_id"], "seed": seed,
              "engineering_only": engineering, "evaluation_hashes": eval_hashes,
              "curated_accepted_rows": curated_available, "resolved_quotas": quotas,
              "sources": source_results, "files": files,
              "mech_comparison_identifiable": engineering is False and curated_available > 0}
    (manifests / "freeze.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = freeze(config, args.data_root, args.output, engineering=args.engineering)
    print(json.dumps({"output": str(args.output), "files": report["files"],
                      "mech_comparison_identifiable": report["mech_comparison_identifiable"]},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
