#!/usr/bin/env python3
"""Build main-product-only FlowER v2 SFT, independently replayed and auditable.

Every original source ID remains in the frozen denominator; unrepresentable
cases stay in unresolved.jsonl and are never treated as model mistakes.
This builder never enables training by itself.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.main_product_root import rebase_flower_inverse_row
from scripts.build_natural_language_event_sft import convert_row, _worker_init

EXPECTED = {"train": 257167, "valid": 2890, "test": 28967}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for piece in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _work(raw: str) -> tuple[bool, dict[str, Any]]:
    row = json.loads(raw)
    try:
        signal.setitimer(signal.ITIMER_REAL, 180)
        corrected = rebase_flower_inverse_row(row)
        public = convert_row(corrected)
        if not public or public[0]["metadata"]["decision_index"] != 0:
            raise ValueError("MAIN_PRODUCT_MISSING_ROOT_ACTION")
        if "." in public[0]["target_smiles"]:
            raise ValueError("MAIN_PRODUCT_VISIBLE_ROOT_HAS_MULTIPLE_COMPONENTS")
        for decision in public:
            if "expected_precursor" in decision["messages"][1]["content"]:
                raise ValueError("MAIN_PRODUCT_GT_ENDPOINT_LEAKED_IN_PROMPT")
            if "atom_map" in decision["messages"][1]["content"]:
                raise ValueError("MAIN_PRODUCT_PRIVATE_MAP_LEAKED_IN_PROMPT")
        nflows = sum(d["metadata"]["decision_type"] == "event" for d in public)
        if nflows != len(row["metadata"]["trace_plan"]["steps"]):
            raise ValueError("MAIN_PRODUCT_ELECTRON_EVENT_COUNT_CHANGED")
        first = public[0]["messages"][2]["tool_calls"][0]["function"]["name"]
        if first not in ("import_fragments", "apply_electron_flow"):
            raise ValueError("MAIN_PRODUCT_INVALID_INITIAL_TOOL")
        return True, {
            "source": corrected, "decisions": public,
            "audit": {
                "id": str(row["source_id"]),
                "main_product": str(public[0]["target_smiles"]),
                "input_condition": "main_product_only_v1",
                "reference_pass": True,
                "flow_events": nflows,
                "import_decisions": sum(d["metadata"]["decision_type"] == "import" for d in public),
                "first_tool": first,
                "original_product_components": corrected["metadata"]["target_selection"]["original_product_component_count"],
                "new_root_missing_components": corrected["metadata"]["target_selection"]["withheld_product_side_components"],
                "final_endpoint_source": "original_trace",
                "model_visible_atom_maps": False,
            },
        }
    except Exception as exc:
        return False, {
            "source_id": str(row.get("source_id") or row.get("id") or ""),
            "error": f"{type(exc).__name__}:{exc}",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir
    manifest_file = source_dir / "training_manifest.json"
    sidecar = source_dir / "ARTIFACT_STATUS.json"
    if not manifest_file.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Frozen FlowER strict source manifest/status missing")
    source_manifest = json.loads(manifest_file.read_text())
    status = json.loads(sidecar.read_text())
    if (status.get("training_allowed") is not True
            or source_manifest.get("strict_trace_universe_complete") is not True):
        raise ValueError("FlowER strict source has no validated training authority")
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite existing main-product data")
    for sub in ("source", "state_sft", "audits"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    source_ids: dict[str, set[str]] = {}
    started = time.monotonic()
    for split in ("valid", "test", "train"):
        source = source_dir / f"{split}.jsonl"
        if sha256(source) != source_manifest["splits"][split]["sha256"]:
            raise ValueError(f"FLOWER_SOURCE_SHA_MISMATCH:{split}")
        good_src = args.output_dir / "source" / f"{split}.jsonl"
        good_sft = args.output_dir / "state_sft" / f"{split}.jsonl"
        good_audit = args.output_dir / "audits" / f"{split}.jsonl"
        bad_path = args.output_dir / "audits" / f"{split}.unresolved.jsonl"
        ids: set[str] = set()
        counts: Counter[str] = Counter()
        with (source.open(encoding="utf-8") as inp,
              good_src.open("w", encoding="utf-8") as sr,
              good_sft.open("w", encoding="utf-8") as st,
              good_audit.open("w", encoding="utf-8") as au,
              bad_path.open("w", encoding="utf-8") as bad,
              mp.get_context("fork").Pool(args.workers, initializer=_worker_init) as pool):
            lines = inp if not args.limit_reactions else (
                line for i, line in enumerate(inp) if i < args.limit_reactions
            )
            for ok, value in pool.imap(_work, lines, chunksize=4):
                ident = str(value["audit"]["id"] if ok else value["source_id"])
                if not ident or ident in ids:
                    raise ValueError(f"DUPLICATE_OR_MISSING_FLOWER_ID:{split}:{ident}")
                ids.add(ident)
                counts["read"] += 1
                if ok:
                    sr.write(json.dumps(value["source"], ensure_ascii=False) + "\n")
                    au.write(json.dumps(value["audit"], ensure_ascii=False) + "\n")
                    for decision in value["decisions"]:
                        st.write(json.dumps(decision, ensure_ascii=False) + "\n")
                        counts["decisions"] += 1
                    counts["reference_pass"] += 1
                    counts["flow_events"] += value["audit"]["flow_events"]
                    counts["import_decisions"] += value["audit"]["import_decisions"]
                    counts["first_" + value["audit"]["first_tool"]] += 1
                    counts["withheld_components"] += value["audit"]["new_root_missing_components"]
                else:
                    bad.write(json.dumps(value, ensure_ascii=False) + "\n")
                    counts["unresolved"] += 1
                if counts["read"] % 1000 == 0:
                    print(json.dumps({
                        "phase": "FLOWER_MAIN_PRODUCT_REPLAY", "split": split,
                        "processed": counts["read"],
                        "expected": EXPECTED[split] if not args.limit_reactions else args.limit_reactions,
                        "passed": counts["reference_pass"],
                        "unresolved": counts["unresolved"],
                        "seconds": round(time.monotonic()-started, 1),
                    }), flush=True)
            for output in (sr, st, au, bad):
                output.flush()
                os.fsync(output.fileno())
        if not args.limit_reactions and counts["read"] != EXPECTED[split]:
            raise ValueError(f"FLOWER_SOURCE_DENOMINATOR_MISMATCH:{split}")
        source_ids[split] = ids
        summaries[split] = {
            **dict(counts), "source_sha256": sha256(source),
            "source_output_sha256": sha256(good_src),
            "state_sft_sha256": sha256(good_sft),
            "audit_sha256": sha256(good_audit),
            "unresolved_sha256": sha256(bad_path),
        }
        print(json.dumps({"phase": "FLOWER_SPLIT_AUDITED", "split": split,
                          **summaries[split]}), flush=True)
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if source_ids[a] & source_ids[b]:
            raise ValueError(f"FLOWER_SPLIT_ID_OVERLAP:{a}:{b}")

    complete = all(r.get("unresolved", 0) == 0 for r in summaries.values())
    manifest = {
        "schema": "flower_main_product_only_audit_v1",
        "corpus": "flower_new_dataset_strict_executable_trace",
        "source_manifest_sha256": sha256(manifest_file),
        "source_dir": str(source_dir),
        "input_condition": "main_product_only_v1",
        "expected_full_strict_reactions": EXPECTED,
        "reports": summaries,
        "complete": complete,
        "reference_replay_verified": complete,
        "training_allowed": False,
        "denominator_not_silently_filtered": True,
        "split_reaction_id_overlap": 0,
        "build_seconds": round(time.monotonic()-started, 1),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps({
            "status": "reference_validated" if complete else "unresolved_retained",
            "training_allowed": False, "reference_replay_verified": complete,
            "reason": "requires explicit promotion after audited exclusions and model-tokenization check",
        }, indent=2) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--limit-reactions", type=int, default=0)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
