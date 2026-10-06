#!/usr/bin/env python3
"""Audit product-only MechET rollouts on the frozen strict-proof FlowER test."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from scripts.audit_reliable_full_endpoint_eval import _jsonl, _sha256


def validate_source(
    *, source: Path, manifest: Path, expected_rows: int = 28967,
    check_product_only_mapping: bool = False,
) -> tuple[dict[str, str], str]:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    test = metadata["splits"]["test"]
    if (int(test["rows"]) != expected_rows or
            int(test["unique_ids"]) != expected_rows):
        raise ValueError("strict test denominator disagrees with frozen manifest")
    digest = _sha256(source)
    if digest != test["sha256"]:
        raise ValueError("strict test source SHA-256 mismatch")
    if check_product_only_mapping:
        from scripts.run_natural_language_value_search import (
            product_only_private_state, visible,
        )
    identifiers: dict[str, str] = {}
    for row in _jsonl(source):
        identifier = str(row["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate strict test ID: {identifier}")
        if not row.get("structural_precursor"):
            raise ValueError(f"missing strict structural precursor: {identifier}")
        if check_product_only_mapping:
            try:
                visible(product_only_private_state(str(row["target_smiles"])))
            except Exception as error:
                raise ValueError(f"product-only mapping failed: {identifier}") from error
        identifiers[identifier] = str(row["source_id"])
    if len(identifiers) != expected_rows:
        raise ValueError("strict test source rows are incomplete")
    return identifiers, digest


def audit(
    *, source: Path, manifest: Path, results_dir: Path,
    expected_rows: int = 28967,
) -> dict[str, Any]:
    source_ids, source_sha = validate_source(
        source=source, manifest=manifest, expected_rows=expected_rows,
    )
    shards = sorted(results_dir.glob("results.shard-*.jsonl"))
    if not shards:
        raise ValueError("no strict-test inference shards")
    observed: set[str] = set()
    counts: Counter[str] = Counter()
    rejected_causes: Counter[str] = Counter()
    for shard in shards:
        for row in _jsonl(shard):
            identifier = str(row["id"])
            if identifier not in source_ids:
                raise ValueError(f"prediction outside strict test: {identifier}")
            if identifier in observed:
                raise ValueError(f"duplicate strict-test prediction: {identifier}")
            if str(row.get("source_id")) != source_ids[identifier]:
                raise ValueError(f"strict-test source ID mismatch: {identifier}")
            if row.get("endpoint_metric") != "structural":
                raise ValueError(f"wrong strict-test endpoint metric: {identifier}")
            if bool(row["top1_exact"]) != bool(row["top1_structural_exact"]):
                raise ValueError(f"contradictory structural exactness: {identifier}")
            if bool(row["top1_exact"]) and not bool(row["top_terminal"]):
                raise ValueError(f"nonterminal exact prediction: {identifier}")
            attempts = list(row.get("attempts") or [])
            rejected = {str(k): int(v) for k, v in (row.get("rejected") or {}).items()}
            actual_rejected = sum(not bool(attempt["accepted"]) for attempt in attempts)
            if sum(rejected.values()) != actual_rejected:
                raise ValueError(f"rejected-action counts disagree: {identifier}")
            if int(row["n_actions"]) != len(row.get("top_actions") or []):
                raise ValueError(f"selected-action count disagrees: {identifier}")
            observed.add(identifier)
            counts["structural_exact"] += bool(row["top1_exact"])
            counts["full_state_exact_secondary"] += bool(row["top1_full_exact"])
            counts["terminal"] += bool(row["top_terminal"])
            counts["episodes_with_rejected_decision"] += actual_rejected > 0
            counts["proposed_decisions_observed"] += len(attempts)
            counts["accepted_decisions_observed"] += len(attempts) - actual_rejected
            counts["rejected_decisions_observed"] += actual_rejected
            rejected_causes.update(rejected)

    return {
        "artifact_type": "reliable_mechet_strict_process_test_audit_v1",
        "source": str(source), "source_sha256": source_sha,
        "manifest": str(manifest),
        "test_denominator": expected_rows,
        "observed_predictions": len(observed),
        "missing_predictions": expected_rows - len(observed),
        "structural_exact": counts["structural_exact"],
        "structural_accuracy": counts["structural_exact"] / expected_rows,
        "full_state_exact_secondary": counts["full_state_exact_secondary"],
        "terminal": counts["terminal"],
        "episodes_with_rejected_decision": counts["episodes_with_rejected_decision"],
        "proposed_decisions_observed": counts["proposed_decisions_observed"],
        "accepted_decisions_observed": counts["accepted_decisions_observed"],
        "rejected_decisions_observed": counts["rejected_decisions_observed"],
        "rejected_causes_observed": dict(rejected_causes),
        "shards": [{"path": str(path), "sha256": _sha256(path)} for path in shards],
        "interpretation": (
            "Missing predictions count as endpoint and terminal failures. Decision "
            "acceptance counts apply only to observed episodes. Executor acceptance "
            "does not establish chemical truth; reference-path divergence and "
            "hallucination require separate, parity-aware adjudication."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        source=args.source, manifest=args.manifest, results_dir=args.results_dir,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
