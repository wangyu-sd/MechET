#!/usr/bin/env python3
"""Reproduce unresolved GT references and preserve actual failure states.

Removing stereo below is a DIAGNOSTIC classification only. It never changes a
reference, accepts a failed trajectory, or permits any quarantined row to train.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
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
                if "electron-source bond is absent" in str(exc) and "step" in loc:
                    from mechet.in_place_grounded_flow import retain_mapped_components
                    from scripts.build_natural_language_event_sft import _translate_mapped_smiles
                    states["original_source_moves"] = loc.get("source_moves", [])
                    states["runtime_moves"] = loc.get("moves", [])
                    states["source_step_index"] = loc["step"].get("step_index")
                    states["projected_source_before"] = _translate_mapped_smiles(
                        retain_mapped_components(loc["step"]["state_before"], loc["present_maps"]),
                        loc["translation"],
                    )
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
        elif "electron-source bond is absent" in error:
            from rdkit import Chem
            params = Chem.SmilesParserParams()
            params.removeHs = False
            actual = states.get("event_state", states.get("current", ""))
            expected = states.get("projected_source_before", "")
            mol = Chem.MolFromSmiles(actual, params)
            reference = Chem.MolFromSmiles(expected, params)
            if mol is not None and reference is not None:
                indexes = [{a.GetAtomMapNum(): a.GetIdx() for a in m.GetAtoms()} for m in (mol, reference)]
                missing, originally_present = set(), []
                for move in states.get("runtime_moves", []):
                    container = move.get("source", {})
                    if container.get("kind") != "BOND":
                        continue
                    pair = tuple(sorted(container["atoms"]))
                    if len(pair) != 2:
                        continue
                    if not set(pair) <= indexes[0].keys() or mol.GetBondBetweenAtoms(*[indexes[0][a] for a in pair]) is None:
                        missing.add(pair)
                        originally_present.append(set(pair) <= indexes[1].keys() and reference.GetBondBetweenAtoms(*[indexes[1][a] for a in pair]) is not None)
                states["missing_runtime_source_bonds"] = [list(p) for p in sorted(missing)]
                states["missing_bonds_present_in_projected_source_before"] = originally_present
                if missing:
                    category = ("source_bond_absent_after_public_prefix_replay" if all(originally_present)
                                else "source_bond_absent_in_recorded_reference")
        strict = same_chemical_state(actual, expected) if actual and expected else None
        no_stereo_same = (same_chemical_state(_without_stereo_for_diagnosis(actual),
                                            _without_stereo_for_diagnosis(expected))
                          if actual and expected else None)
        if category == "public_endpoint_replay_disagreement" and not strict and no_stereo_same:
            category = "endpoint_stereo_mismatch"
        before_failed_action = category.startswith("source_bond_absent_")
        if strict is True and not before_failed_action:
            raise ValueError("REVIEW_CHEMICALLY_EQUAL_FAILURE_REQUIRES_FIX_NOT_QUARANTINE") from exc
        return {
            "source_id": str(source["source_id"]), "error": f"{type(exc).__name__}:{error}",
            "classification": category, "source_state": expected, "executed_state": actual,
            "strict_chemical_equal": strict, "same_only_after_stereo_removed_for_diagnosis": no_stereo_same,
            "comparison_phase": "before_failed_action" if before_failed_action else "failed_successor_or_endpoint",
            "original_product_side": source["target_smiles"],
            "corrected_mapped_main_root": inverse["target_smiles"] if inverse else None,
            "original_full_endpoint": source.get("full_precursor_state", source.get("expected_precursor")),
            "context": states, "training_allowed": False,
            "source_row_sha256": hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
    raise ValueError(f"REVIEW_NOW_REPLAYABLE_MUST_REPAIR_DATA:{source['source_id']}")


def _review_line(line):
    value = json.loads(line)
    return value, reproduce_failure(value["source_row"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases, counts = [], Counter()
    if args.output.exists():
        raise ValueError("REFUSING_TO_OVERWRITE_EXISTING_REVIEW")
    cases_path = args.output.with_suffix(".cases.jsonl")
    progress_path = args.output.with_suffix(".progress.jsonl")
    total = sum(len(r["unresolved_ids"]) for r in manifest["reports"].values())
    with cases_path.open("x") as case_sink, progress_path.open("x") as progress_sink:
        for split in ("train", "valid", "test"):
            report = manifest["reports"][split]
            path = Path(report["unresolved_path"])
            if sha(path) != report["unresolved_sha256"]:
                raise ValueError("UNRESOLVED_JOURNAL_HASH_MISMATCH")
            seen = set()
            context = mp.get_context("fork").Pool(args.workers) if args.workers > 1 else nullcontext(None)
            with path.open() as source, context as pool:
                reviewed = pool.imap(_review_line, source, chunksize=1) if pool else map(_review_line, source)
                for value, result in reviewed:
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
                    case_sink.write(json.dumps(result) + "\n")
                    case_sink.flush()
                    os.fsync(case_sink.fileno())
                    progress = {"stage": "quarantine-independent-reproduction", "split": split,
                                "source_id": rid, "classification": result["classification"],
                                "training_allowed": False, "processed": len(cases), "total": total,
                                "percent": 100 * len(cases) / total,
                                "timestamp": datetime.now(timezone.utc).isoformat()}
                    progress_sink.write(json.dumps(progress) + "\n")
                    progress_sink.flush()
                    os.fsync(progress_sink.fileno())
                    print(json.dumps(progress), flush=True)
            if seen != set(report["unresolved_ids"]):
                raise ValueError("UNRESOLVED_IDS_CHANGED")
    result = {"track": manifest["track"], "source_manifest_sha256": sha(manifest_path),
              "review_version": "pr96_reproduce_states_without_relabeling_v1",
              "quarantined_rows_training_allowed": False, "counts": dict(counts), "cases": cases,
              "all_failures_reproduced": True,
              "all_classified": not counts["unclassified_failure_needs_investigation"],
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "review_code_sha256": sha(Path(__file__)),
              "cases_journal": {"path": str(cases_path), "sha256": sha(cases_path)},
              "progress_journal": {"path": str(progress_path), "sha256": sha(progress_path)}}
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "quarantine-review-complete", "counts": dict(counts),
                      "output": str(args.output), "all_classified": result["all_classified"]}), flush=True)


if __name__ == "__main__":
    main()
