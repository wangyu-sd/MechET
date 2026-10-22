#!/usr/bin/env python3
"""Reproduce unresolved GT references and preserve actual failure states.

Removing stereo below is a DIAGNOSTIC classification only. It never changes a
reference, accepts a failed trajectory, or permits any quarantined row to train.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.build_main_product_training_data import prepare_reference, audit_reference
from scripts.build_mech31k_main_product_only_sft import same_chemical_state, sha


def _without_stereo_for_diagnosis(text):
    from rdkit import Chem
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(text, params)
    if mol is None:
        return ""
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def reproduce_failure(source):
    inverse = None
    try:
        inverse = prepare_reference(source)
        audit_reference(inverse)
    except Exception as exc:
        states = {}
        frame = exc.__traceback__
        while frame:
            loc = frame.tb_frame.f_locals
            if frame.tb_frame.f_code.co_name == "convert_row":
                states.update({k: loc[k] for k in ("current", "runtime_expected", "target", "event_state") if k in loc})
            if frame.tb_frame.f_code.co_name == "audit_reference" and loc.get("child") is not None:
                states["executed_successor"] = loc["child"].state
                states["expected_successor"] = loc.get("expected", "")
                states["event_index"] = loc.get("flow_index")
            frame = frame.tb_next
        error = str(exc)
        actual, expected, category = "", "", "unclassified_failure_needs_investigation"
        if "public root is not stereo-isomorphic" in error:
            actual, expected = states.get("current", ""), states.get("target", "")
            category = "root_stereo_roundtrip_mismatch"
        elif "natural-language trace endpoint changed" in error:
            actual, expected = states.get("current", ""), states.get("runtime_expected", "")
            category = "public_endpoint_replay_disagreement"
        elif "PUBLIC_REFERENCE_SUCCESSOR_MISMATCH" in error:
            actual, expected = states.get("executed_successor", ""), states.get("expected_successor", "")
            category = "public_event_successor_disagreement"
        elif "MAIN_PRODUCT_AMBIGUOUS" in error or "MAIN_PRODUCT_NOT_FOUND" in error:
            category = "ambiguous_or_absent_unique_main_product"
        strict = same_chemical_state(actual, expected) if actual and expected else None
        no_stereo_same = (same_chemical_state(_without_stereo_for_diagnosis(actual),
                                            _without_stereo_for_diagnosis(expected))
                          if actual and expected else None)
        if category == "public_endpoint_replay_disagreement" and not strict and no_stereo_same:
            category = "endpoint_stereo_mismatch"
        if strict is True:
            raise ValueError("REVIEW_CHEMICALLY_EQUAL_FAILURE_REQUIRES_FIX_NOT_QUARANTINE") from exc
        return {
            "source_id": str(source["source_id"]), "error": f"{type(exc).__name__}:{error}",
            "classification": category, "source_state": expected, "executed_state": actual,
            "strict_chemical_equal": strict, "same_only_after_stereo_removed_for_diagnosis": no_stereo_same,
            "original_product_side": source["target_smiles"],
            "corrected_mapped_main_root": inverse["target_smiles"] if inverse else None,
            "original_full_endpoint": source.get("full_precursor_state", source.get("expected_precursor")),
            "context": states, "training_allowed": False,
            "source_row_sha256": hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
    raise ValueError(f"REVIEW_NOW_REPLAYABLE_MUST_REPAIR_DATA:{source['source_id']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases, counts = [], Counter()
    for split in ("train", "valid", "test"):
        report = manifest["reports"][split]
        path = Path(report["unresolved_path"])
        if sha(path) != report["unresolved_sha256"]:
            raise ValueError("UNRESOLVED_JOURNAL_HASH_MISMATCH")
        seen = set()
        for line in path.open():
            value = json.loads(line)
            result = reproduce_failure(value["source_row"])
            if result["source_row_sha256"] != value["source_row_sha256"]:
                raise ValueError("UNRESOLVED_SOURCE_ROW_HASH_MISMATCH")
            if result["error"] != value["error"]:
                raise ValueError("UNRESOLVED_FAILURE_CHANGED_REQUIRES_REVIEW")
            result["split"] = split
            rid = result["source_id"]
            if rid in seen:
                raise ValueError("DUPLICATE_UNRESOLVED_ID")
            seen.add(rid)
            cases.append(result)
            counts[result["classification"]] += 1
            print(json.dumps({"stage": "quarantine-independent-reproduction", "split": split,
                              "source_id": rid, "classification": result["classification"],
                              "training_allowed": False}), flush=True)
        if seen != set(report["unresolved_ids"]):
            raise ValueError("UNRESOLVED_IDS_CHANGED")
    result = {"track": manifest["track"], "source_manifest_sha256": sha(manifest_path),
              "review_version": "pr96_reproduce_states_without_relabeling_v1",
              "quarantined_rows_training_allowed": False, "counts": dict(counts), "cases": cases,
              "all_failures_reproduced": True,
              "all_classified": not counts["unclassified_failure_needs_investigation"],
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "review_code_sha256": sha(Path(__file__))}
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "quarantine-review-complete", "counts": dict(counts),
                      "output": str(args.output), "all_classified": result["all_classified"]}), flush=True)


if __name__ == "__main__":
    main()
