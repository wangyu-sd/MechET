#!/usr/bin/env python3
"""Authorize a separately audited PR96 reference subset; tokens still gate SFT.

Never alters PR95 or the original source data. This produces new explicit
lineage, full denominators and fixed VALID views, not a manual status override.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.in_place_grounded_flow import extract_import_fragments
from scripts.build_main_product_training_data import DENOMINATORS, validate_coverage
from scripts.build_mech31k_main_product_only_sft import sha
from scripts.pr96_source_product_side import prepare_source_reference


def verify_source_identity(source, reference):
    digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if str(source["source_id"]) != str(reference["source_id"]) or digest != reference["metadata"]["source_row_sha256"]:
        raise ValueError("ORIGINAL_SOURCE_ID_OR_HASH_CHANGED")
    if reference["metadata"]["trace_plan"]["steps"] != source["metadata"]["trace_plan"]["steps"]:
        raise ValueError("REFERENCE_STEPS_CHANGED")
    original_endpoint = source.get("full_precursor_state") or source["expected_precursor"]
    if reference.get("full_precursor_state", reference["expected_precursor"]) != original_endpoint:
        raise ValueError("ORIGINAL_ENDPOINT_CHANGED")
    expected_reference = prepare_source_reference(source)
    main = expected_reference["target_smiles"]
    if reference["target_smiles"] != main or "." in main:
        raise ValueError("PUBLIC_MAIN_ROOT_CHANGED")
    wanted_imports = sorted(extract_import_fragments(expected_reference))
    if sorted(extract_import_fragments(reference)) != wanted_imports:
        raise ValueError("MISSING_OR_DUPLICATED_FIRST_USE_IMPORT")
    if expected_reference["metadata"].get("product_side_definition") == "flower_full_compiled_product_side_v1":
        for key in ("product_side_definition", "original_product_side", "legacy_selected_target"):
            if reference["metadata"].get(key) != expected_reference["metadata"][key]:
                raise ValueError("FULL_FLOWER_PRODUCT_SIDE_NOT_VERIFIED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--history-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    data_path, hist_path = args.data_dir / "manifest.json", args.history_dir / "manifest.json"
    data, hist = (json.loads(p.read_text()) for p in (data_path, hist_path))
    data_sha = sha(data_path)
    review_path, cohort_path = args.data_dir / "quarantine_review.json", args.data_dir / "frozen_valid_cohorts.json"
    review, cohorts = (json.loads(p.read_text()) for p in (review_path, cohort_path))
    if hist.get("source_manifest_sha256") != data_sha or hist.get("track") != data["track"]:
        raise ValueError("HISTORY_PARENT_MISMATCH")
    if not hist.get("accepted_subset_independently_replayed") or hist.get("history_contains_future") is not False:
        raise ValueError("INDEPENDENT_HISTORY_AUDIT_MISSING")
    if review.get("source_manifest_sha256") != data_sha or not review.get("all_failures_reproduced") or not review.get("all_classified"):
        raise ValueError("QUARANTINE_NOT_INDEPENDENTLY_REVIEWED")
    if cohorts.get("data_manifest_sha256") != data_sha or not cohorts.get("frozen_before_training") or cohorts.get("test_used"):
        raise ValueError("VALID_COHORT_PARENT_MISMATCH")
    paired, independent = set(cohorts["paired_ids"]), set(cohorts["independent_ids"])
    valid_ids = set(data["reports"]["valid"]["accepted_ids"])
    if len(paired) != 64 or len(independent) != 64 or paired & independent or not (paired | independent) <= valid_ids:
        raise ValueError("INVALID_FIXED_VALID_COHORTS")
    raw_manifest_path = Path(data["source_manifest"])
    if sha(raw_manifest_path) != data["source_manifest_sha256"]:
        raise ValueError("ORIGINAL_SOURCE_MANIFEST_HASH_MISMATCH")
    raw_manifest = json.loads(raw_manifest_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_ids, reports = {}, {}
    for split, expected in DENOMINATORS[data["track"]].items():
        record, hrecord = data["reports"][split], hist["reports"][split]
        source_path = raw_manifest_path.parent / f"{split}.jsonl"
        if (raw_manifest["splits"][split]["rows"] != expected
                or sha(source_path) != record["source_sha256"]
                or sha(source_path) != raw_manifest["splits"][split]["sha256"]):
            raise ValueError("ORIGINAL_SOURCE_FILE_HASH_OR_DENOMINATOR_MISMATCH")
        inverse_path = Path(record["outputs"]["inverse"]["path"])
        if sha(inverse_path) != record["outputs"]["inverse"]["sha256"]:
            raise ValueError("REFERENCE_FILE_HASH_MISMATCH")
        accepted, unresolved, original_ids = set(), set(record["unresolved_ids"]), set()
        reviewed = {str(c["source_id"]) for c in review["cases"]
                    if c["split"] == split and c.get("training_allowed") is False}
        if reviewed != unresolved:
            raise ValueError("QUARANTINE_REVIEW_ID_MISMATCH")
        reference_output = args.output_dir / f"{split}.reference_integrity.jsonl"
        with inverse_path.open() as reader, source_path.open() as source, reference_output.open("x") as audit:
            next_reference = json.loads(next(reader, "null"))
            for line in source:
                row = json.loads(line)
                rid = str(row["source_id"])
                if rid in original_ids:
                    raise ValueError("DUPLICATE_ORIGINAL_ID")
                original_ids.add(rid)
                if next_reference is not None and rid == str(next_reference["source_id"]):
                    verify_source_identity(row, next_reference)
                    accepted.add(rid)
                    audit.write(json.dumps({"source_id": rid, "source_row_sha256": next_reference["metadata"]["source_row_sha256"],
                                            "original_steps_and_endpoint_unchanged": True,
                                            "main_root_and_missing_import_partition_verified": True}) + "\n")
                    next_reference = json.loads(next(reader, "null"))
                elif rid not in unresolved:
                    raise ValueError(f"ORIGINAL_REACTION_NOT_ACCOUNTED_FOR:{rid}")
                if len(original_ids) % 500 == 0:
                    event = {"stage": "original-source-integrity", "track": data["track"], "split": split,
                             "processed": len(original_ids), "total": expected,
                             "percent": 100 * len(original_ids) / expected}
                    text = json.dumps(event)
                    print(text, flush=True)
                    with (args.output_dir / "progress.jsonl").open("a") as progress:
                        progress.write(text + "\n")
            if next_reference is not None:
                raise ValueError("UNBOUND_REFERENCE_ROWS")
        validate_coverage(original=expected, accepted=accepted, unresolved=unresolved)
        if accepted != set(record["accepted_ids"]) or accepted | unresolved != original_ids:
            raise ValueError("SOURCE_REFERENCE_ID_SET_MISMATCH")
        if hrecord["reactions"] != len(accepted) or hrecord["original_count"] != expected or not hrecord["all_online_prefixes_match"]:
            raise ValueError("HISTORY_COVERAGE_MISMATCH")
        state_hash = record["outputs"]["state_sft"]["sha256"]
        if (sha(Path(record["outputs"]["state_sft"]["path"])) != state_hash
                or hrecord["source_state_sha256"] != state_hash
                or sha(Path(hrecord["path"])) != hrecord["sha256"]):
            raise ValueError("STATE_OR_HISTORY_DATA_CHANGED_AFTER_INDEPENDENT_AUDIT")
        split_ids[split] = original_ids
        reports[split] = {"original_count": expected, "reference_pass": len(accepted), "unresolved": len(unresolved),
                          "coverage": len(accepted)/expected, "reference_integrity_journal_sha256": sha(reference_output)}
        print(json.dumps({"stage": "source-integrity-verified", "track": data["track"], "split": split, **reports[split]}), flush=True)
    if any(split_ids[a] & split_ids[b] for a,b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("SOURCE_SPLIT_ID_OVERLAP")
    # Freeze all decisions for each selected VALID reaction, never a row-count
    # prefix that silently cuts a reaction. Independent IDs are not loss-selection data.
    views = {}
    for stage, path, expected_hash in (
        ("state", args.data_dir / "state_sft/valid.jsonl", data["reports"]["valid"]["outputs"]["state_sft"]["sha256"]),
        ("history", args.history_dir / "valid.jsonl", hist["reports"]["valid"]["sha256"]),
    ):
        if sha(path) != expected_hash:
            raise ValueError("VALID_VIEW_HASH_MISMATCH")
        target = args.output_dir / f"valid_{stage}_paired64.jsonl"
        seen, count = set(), 0
        with target.open("x") as out, path.open() as reader:
            for line in reader:
                row = json.loads(line)
                if str(row["source_id"]) in paired:
                    out.write(line)
                    seen.add(str(row["source_id"]))
                    count += 1
        if seen != paired:
            raise ValueError("PAIRED_VALID_REACTIONS_MISSING")
        views[stage] = {"path": str(target), "sha256": sha(target), "decision_rows": count, "reactions": 64}
    approval = {"authorization": "PR96 explicitly reviewed replay-passed training subset, not raw full corpus",
                "reference_training_authorized": True, "training_allowed": False,
                "pending": ["actual_deployed_gpu_preflight", "full_pinned_tokenizer_cache_audit"],
                "track": data["track"], "reports": reports, "validation_views": views,
                "reference_manifest": str(data_path), "reference_manifest_sha256": data_sha,
                "history_manifest": str(hist_path), "history_manifest_sha256": sha(hist_path),
                "quarantine_review_sha256": sha(review_path), "frozen_cohorts_sha256": sha(cohort_path),
                "source_manifest_sha256": sha(raw_manifest_path), "approval_code_sha256": sha(Path(__file__)),
                "core_metric": "derive target-atom-containing precursor components from corrected main root, not legacy product-side annotation",
                "complete_test_evaluation_executed": False}
    with (args.output_dir / "reference_approval.json").open("x") as out:
        out.write(json.dumps(approval, indent=2) + "\n")
    print(json.dumps({"stage": "reference-subset-authorized-tokens-still-required", "track": data["track"],
                      "approval": str(args.output_dir / "reference_approval.json"), "training_allowed": False}), flush=True)


if __name__ == "__main__":
    main()
