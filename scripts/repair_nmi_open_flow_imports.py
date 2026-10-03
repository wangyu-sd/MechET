#!/usr/bin/env python3
"""Version the H2 Open-Flow control with all frozen step imports declared.

OPEN_FLOW v1 is one-shot and requires every import before STEP 0. The older
A4 formatter emitted only trace_plan.initial_imports, omitting fragments
introduced before later steps. This repairs that representation only: same
reaction IDs, product prompts, electron moves, endpoints and split assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.open_flow_program import execute_open_flow, parse_open_flow
from scripts.build_flower_a1_a4_a5_a6 import format_open_flow_program


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    messages = list(row.get("messages") or [])
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError(f"expected one Open-Flow answer: {row.get('source_id')}")
    plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
    if not plan.get("steps"):
        raise ValueError(f"frozen trace plan missing: {row.get('source_id')}")
    original = str(assistants[0].get("content") or "")
    old_imports, old_steps = parse_open_flow(original)
    repaired = format_open_flow_program(plan)
    new_imports, new_steps = parse_open_flow(repaired)
    if old_steps != new_steps:
        raise ValueError(f"repair changed electron moves: {row.get('source_id')}")
    if old_imports != list(plan.get("initial_imports") or []):
        raise ValueError(f"unexpected old Open-Flow imports: {row.get('source_id')}")
    expected_imports = old_imports + [
        str(fragment) for step in plan["steps"] for fragment in step.get("imports") or []
    ]
    if new_imports != expected_imports:
        raise ValueError(f"repair import ordering mismatch: {row.get('source_id')}")
    changed = repaired != original
    if changed:
        assistants[0]["content"] = repaired
    return row, changed


def _verify_repaired_reference(row: dict[str, Any]) -> None:
    target = str(row.get("target_smiles") or "")
    expected = str(row.get("structural_precursor") or "")
    result = execute_open_flow(row["messages"][-1]["content"], target)
    if not result.get("execute_ok"):
        raise ValueError(f"repaired reference execution failed for {row.get('source_id')}: {result}")
    derived = split_precursor_endpoints(str(result.get("derived_precursor") or ""), target).structural
    if not structural_exact(derived, expected):
        raise ValueError(f"repaired reference endpoint mismatch for {row.get('source_id')}")


def build(parent_dir: Path, output_dir: Path, parent_config: Path,
          *, model_output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing repaired data: {output_dir}")
    parent_manifest_path = parent_dir / "manifest.json"
    parent = json.loads(parent_manifest_path.read_text())
    if parent.get("condition") != "open_flow" or not parent.get("training_allowed"):
        raise ValueError("parent is not a frozen train-ready H2 Open-Flow artifact")
    parent_verification_path = parent_dir.parent / "verification.json"
    parent_verification = json.loads(parent_verification_path.read_text())
    if (parent_verification.get("passed") is not True
            or parent_verification.get("parent_split_manifest_sha256")
            != parent.get("parent_split_manifest_sha256")
            or parent_verification.get("stable_reaction_ids")
            != parent.get("stable_reaction_ids")):
        raise ValueError("parent H2 product/endpoint parity verification mismatch")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        hashes = {}
        changed_ids = {}
        for split in ("train", "valid", "test"):
            source = parent_dir / f"{split}.jsonl"
            if _sha(source) != parent["output_sha256"][split]:
                raise ValueError(f"parent Open-Flow {split} SHA mismatch")
            target = temporary / f"{split}.jsonl"
            source_hash = hashlib.sha256()
            output_hash = hashlib.sha256()
            seen: set[str] = set()
            affected = []
            with source.open("rb") as reader, target.open("wb") as writer:
                for line in reader:
                    source_hash.update(line)
                    row = json.loads(line)
                    identifier = str(row.get("source_id") or "")
                    if not identifier or identifier in seen:
                        raise ValueError(f"duplicate/missing source ID in {split}: {identifier}")
                    seen.add(identifier)
                    row, changed = repair_row(row)
                    if changed:
                        _verify_repaired_reference(row)
                        affected.append(identifier)
                        encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                    else:
                        encoded = line
                    writer.write(encoded)
                    output_hash.update(encoded)
            if source_hash.hexdigest() != parent["output_sha256"][split]:
                raise ValueError(f"parent Open-Flow {split} changed while building")
            if len(seen) != int(parent["rows"][split]):
                raise ValueError(f"repaired {split} row count mismatch")
            hashes[split] = output_hash.hexdigest()
            changed_ids[split] = affected
            print(json.dumps({"split": split, "rows": len(seen),
                              "changed": len(affected), "sha256": hashes[split]}), flush=True)
        manifest = dict(parent)
        manifest.update({
            "artifact_type": "nmi_h2_open_flow_all_step_imports_v2",
            "source": str(parent_dir.resolve()),
            "source_sha256": _sha(parent_manifest_path),
            "repair_rule": "initial_imports plus every trace_plan.steps[*].imports, in recorded order, all before STEP 0",
            "parent_open_flow_manifest_sha256": _sha(parent_manifest_path),
            "output_sha256": hashes,
            "changed_rows": {split: len(ids) for split, ids in changed_ids.items()},
            "changed_source_ids": changed_ids,
            "changed_reference_replay": "strict_execution_and_structural_endpoint_exact_for_every_changed_row",
            "training_allowed": True,
        })
        for split in ("train", "valid", "test"):
            destination = output_dir / f"{split}.jsonl"
            manifest["splits"][split] = {
                "file": str(destination.resolve()), "rows": parent["rows"][split],
                "sha256": hashes[split],
            }
            manifest["tasks"]["open_flow"][split] = {
                "path": str(destination.resolve()), "rows": parent["rows"][split],
                "sha256": hashes[split],
            }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verification = {
            "artifact_type": "nmi_h2_open_flow_import_repair_verification_v2",
            "parent_split_manifest_sha256": parent["parent_split_manifest_sha256"],
            "stable_reaction_ids": parent["stable_reaction_ids"],
            "source_representation_verification_sha256": _sha(parent_verification_path),
            "split_rows": parent["rows"],
            "changed_rows": manifest["changed_rows"],
            "unchanged_row_policy": "parent bytes copied exactly",
            "changed_row_policy": "only assistant flow content updated; every changed program strictly executes to the frozen structural endpoint",
            "passed": True,
        }
        verification_path = temporary / "verification.json"
        verification_path.write_text(json.dumps(verification, indent=2) + "\n")
        config = yaml.safe_load(parent_config.read_text())
        if int(config["training"]["max_steps"]) != 10494 or int(config["training"]["seed"]) != 17:
            raise ValueError("parent Open-Flow optimizer budget/seed drifted")
        if config.get("condition_name") != "nmi_h2_open_flow_qwen3_8b_seed17":
            raise ValueError("parent Open-Flow condition name drifted")
        config["train_file"] = str((output_dir / "train.jsonl").resolve())
        config["validation_file"] = str((output_dir / "valid.jsonl").resolve())
        config["test_file"] = str((output_dir / "test.jsonl").resolve())
        config["pretokenized_cache_dir"] = str((output_dir / "qwen3_8b_tokens_16384").resolve())
        config["output_dir"] = str(model_output_dir.resolve())
        config["contract"]["stable_id_manifest"] = str((output_dir / "manifest.json").resolve())
        config["contract"]["paper_run_role"] = "issue79_h2_matched_open_flow_import_repair_v2"
        config_dir = temporary / "configs"
        config_dir.mkdir()
        config_path = config_dir / "open_flow.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        config_manifest = {
            "artifact_type": "nmi_h2_open_flow_import_repair_sft_config_v2",
            "split_manifest_sha256": parent["parent_split_manifest_sha256"],
            "stable_reaction_ids": parent["stable_reaction_ids"],
            "representation_verification_sha256": _sha(verification_path),
            "equal_optimizer_updates": int(config["training"]["max_steps"]),
            "config_sha256": {"open_flow": _sha(config_path)},
            "model_revision": config["training"]["model_revision"],
            "seed": int(config["training"]["seed"]),
            "gpu_count": int(config["pretokenization_world_size"]),
            "equal_token_budget": False,
        }
        (config_dir / "manifest.json").write_text(json.dumps(config_manifest, indent=2) + "\n")
        temporary.rename(output_dir)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--model-output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.parent_dir, args.output_dir, args.parent_config,
                   model_output_dir=args.model_output_dir)
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"),
                      "changed_rows": result["changed_rows"],
                      "output_sha256": result["output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
