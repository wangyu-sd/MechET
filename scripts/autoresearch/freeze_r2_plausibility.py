#!/usr/bin/env python3
"""Freeze R2 only after all recorded positives and audited negatives exist.

This is an intake and accounting gate, not a chemical-validity oracle. Source
builders/reviewers remain responsible for substantiating each negative label.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


NEGATIVE_CLASSES = (
    "wrong_nucleophile",
    "wrong_electrophile",
    "wrong_leaving_group",
    "formal_charge_corruption",
    "regioisomeric_disconnection",
    "bond_order_corruption",
    "missing_necessary_fragment",
    "executor_valid_wrong_successor",
)
INDEPENDENT_EVIDENCE_KINDS = {
    "experimental_outcome",
    "literature_mechanistic_constraint",
    "independently_verified_mechanistic_constraint",
}


def _read_source(path: Path, *, positive: bool, negative_class: str | None = None,
                 expected: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_path = path.parent / "manifest.json"
    status_path = path.parent / "ARTIFACT_STATUS.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError(f"R2 source requires a manifest and status: {path}")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    cohort_hash = digest(path)
    if manifest.get("cohort_sha256") != cohort_hash or status.get("cohort_sha256") != cohort_hash:
        raise ValueError(f"R2 source cohort hash drifted: {path}")
    if status.get("evaluation_allowed") is not False or status.get("training_allowed") is not False:
        raise ValueError(f"R2 component must remain a non-evaluation, non-training source: {path}")
    permission = "positive_source_allowed" if positive else "evidence_audited"
    if status.get(permission) is not True:
        raise ValueError(f"R2 source lacks {permission}: {path}")
    count_key = "positive_proposals" if positive else "negative_proposals"
    if manifest.get(count_key) != expected:
        raise ValueError(f"R2 source manifest has wrong {count_key}: {path}")
    if not positive and manifest.get("negative_class") != negative_class:
        raise ValueError(f"R2 source manifest has wrong negative class: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"R2 source has {len(rows)} rows; expected {expected}: {path}")
    return rows, {
        "cohort": str(path.resolve()),
        "cohort_sha256": cohort_hash,
        "manifest_sha256": digest(manifest_path),
        "status_sha256": digest(status_path),
    }


def _validate_proposal(row: dict[str, Any]) -> tuple[str, str, str]:
    proposal_id = row.get("proposal_id")
    model_input = row.get("model_input")
    label = row.get("private_label")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("R2 proposal needs a stable proposal_id")
    if not isinstance(model_input, dict) or set(model_input) != {
        "product_smiles", "proposed_precursors"
    } or not isinstance(label, dict):
        raise ValueError(f"R2 proposal has invalid model/private fields: {proposal_id}")
    product = model_input["product_smiles"]
    precursors = model_input["proposed_precursors"]
    if not isinstance(product, str) or not isinstance(precursors, str):
        raise ValueError(f"R2 proposal has non-string SMILES: {proposal_id}")
    if row.get("product_smiles") != product or product_key(product) != product:
        raise ValueError(f"R2 proposal has noncanonical/mismatched product: {proposal_id}")
    if product_key(precursors) != precursors:
        raise ValueError(f"R2 proposal has noncanonical precursors: {proposal_id}")
    if row.get("source_split") != "test":
        raise ValueError(f"R2 proposal is not from held-out test: {proposal_id}")
    return proposal_id, product, precursors


def _validate_negative(row: dict[str, Any], negative_class: str) -> None:
    label = row["private_label"]
    if label.get("negative_class") != negative_class or row.get("strata", {}).get(
        "negative_class"
    ) != negative_class:
        raise ValueError(f"R2 negative class mismatch: {row['proposal_id']}")
    evidence_kind = label.get("evidence_kind")
    if not isinstance(evidence_kind, str) or not evidence_kind.strip() or evidence_kind in {
        "different_from_gt", "not_recorded", "executor_accepted", "non_reference_successor"
    }:
        raise ValueError(f"R2 negative lacks substantive evidence: {row['proposal_id']}")
    if negative_class == "missing_necessary_fragment":
        if (evidence_kind != "product_element_inventory_deficit_after_fragment_omission"
                or label.get("scope") != "closed_stated_precursor_inventory_atom_conservation"
                or not isinstance(label.get("missing_element_counts"), dict)
                or not label["missing_element_counts"]
                or not all(isinstance(n, int) and n > 0
                           for n in label["missing_element_counts"].values())):
            raise ValueError(f"R2 missing-fragment witness is absent: {row['proposal_id']}")
    else:
        # These classes have no repository-native formal witness yet. Require a
        # separately reviewed, content-addressed evidence artifact; a label or
        # executor success alone cannot establish chemical inconsistency.
        evidence = label.get("independent_negative_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"R2 negative lacks independent evidence: {row['proposal_id']}")
        if (evidence.get("kind") not in INDEPENDENT_EVIDENCE_KINDS
                or not isinstance(evidence.get("locator"), str)
                or not evidence["locator"].strip()
                or not isinstance(evidence.get("sha256"), str)
                or len(evidence["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in evidence["sha256"])
                or not isinstance(evidence.get("reviewer_id"), str)
                or not evidence["reviewer_id"].strip()):
            raise ValueError(f"R2 independent evidence is incomplete: {row['proposal_id']}")
    if negative_class == "executor_valid_wrong_successor":
        replay = label.get("executor_replay")
        if not isinstance(replay, dict) or replay.get("accepted") is not True or not replay.get(
            "successor_smiles"
        ):
            raise ValueError(f"R2 executor-valid class lacks accepted replay: {row['proposal_id']}")


def freeze(positives: Path, negatives: dict[str, Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R2 evaluation cohort already exists: {output}")
    if set(negatives) != set(NEGATIVE_CLASSES) or len(set(negatives.values())) != len(NEGATIVE_CLASSES):
        raise ValueError("R2 requires eight distinct negative-class sources")
    positive_rows, positive_source = _read_source(positives, positive=True, expected=400)
    sources = {"positives": positive_source}
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    positive_products: set[str] = set()
    positive_counts: Counter[str] = Counter()
    for row in positive_rows:
        proposal_id, product, precursors = _validate_proposal(row)
        label = row["private_label"]
        kind = row.get("strata", {}).get("positive_type")
        if (label.get("known_recorded_positive") is not True
                or not isinstance(label.get("proposal_record_ids"), list)
                or not label["proposal_record_ids"]
                or kind not in {"recorded_precursor", "documented_alternative"}):
            raise ValueError(f"R2 positive has no recorded support: {proposal_id}")
        if kind == "recorded_precursor" and label.get("evidence_kind") != "heldout_recorded_precursor":
            raise ValueError(f"R2 recorded positive has wrong evidence: {proposal_id}")
        if kind == "documented_alternative":
            primary_ids = label.get("primary_record_ids")
            if (label.get("evidence_kind") != "independent_heldout_recorded_alternative"
                    or not isinstance(primary_ids, list) or not primary_ids
                    or set(map(str, primary_ids)) & set(map(str, label["proposal_record_ids"]))
                    or not isinstance(label.get("primary_precursors"), str)
                    or product_key(label["primary_precursors"]) == precursors):
                raise ValueError(f"R2 alternative positive lacks independent record: {proposal_id}")
        if proposal_id in seen_ids or product in positive_products:
            raise ValueError(f"R2 duplicate positive ID/product: {proposal_id}")
        seen_ids.add(proposal_id)
        seen_pairs.add((product, precursors))
        positive_products.add(product)
        positive_counts[kind] += 1
        rows.append(row)
    if positive_counts != {"recorded_precursor": 200, "documented_alternative": 200}:
        raise ValueError(f"R2 positive strata are not 200/200: {dict(positive_counts)}")
    for negative_class in NEGATIVE_CLASSES:
        path = negatives[negative_class]
        negative_rows, source = _read_source(path, positive=False,
                                             negative_class=negative_class, expected=50)
        sources[negative_class] = source
        for row in negative_rows:
            proposal_id, product, precursors = _validate_proposal(row)
            _validate_negative(row, negative_class)
            if product not in positive_products:
                raise ValueError(f"R2 negative product lacks a matched positive: {proposal_id}")
            if proposal_id in seen_ids or (product, precursors) in seen_pairs:
                raise ValueError(f"R2 duplicate or contradictory proposal: {proposal_id}")
            seen_ids.add(proposal_id)
            seen_pairs.add((product, precursors))
            rows.append(row)
    if len(rows) != 800:
        raise AssertionError("R2 frozen denominator must be 800")
    rows.sort(key=lambda row: row["proposal_id"])
    output.mkdir(parents=True)
    cohort = output / "r2_plausibility.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "artifact_type": "r2_plausibility_manifest_v1",
        "cohort": str(cohort.resolve()), "cohort_sha256": digest(cohort),
        "rows": 800, "positive_proposals": 400, "negative_proposals": 400,
        "positive_strata": dict(positive_counts),
        "negative_strata": {name: 50 for name in NEGATIVE_CLASSES},
        "sources": sources,
        "claim_boundary": "Recorded positives and independently audited negative proposals; source evidence is preserved privately. Formal execution alone is not chemical truth.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r2_plausibility_status_v1",
        "cohort_sha256": report["cohort_sha256"],
        "evaluation_allowed": True, "training_allowed": False,
        "independent_negative_evidence_required": True,
    }, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--negative-class", action="append", required=True,
                        metavar="CLASS=JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    negatives: dict[str, Path] = {}
    for item in args.negative_class:
        name, separator, path = item.partition("=")
        if not separator or not path or name not in NEGATIVE_CLASSES or name in negatives:
            parser.error(f"invalid --negative-class {item!r}")
        negatives[name] = Path(path)
    print(json.dumps(freeze(args.positives, negatives, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
