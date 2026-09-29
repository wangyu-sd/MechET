#!/usr/bin/env python3
"""Compile a pinned SynEPD release into replayed product-only State-SFT rows.

This is a *candidate* PR69 curated training source, never an R4 replacement.
Only records with an unambiguous principal organic product and exact inverse
replay under the unchanged MechET executor enter the output. No source record
is silently dropped: every rejected record is written to quarantine.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_natural_language_event_sft import convert_row  # noqa: E402
from mechet.forward_expert import verify_electron_step  # noqa: E402
from mechet.in_place_grounded_flow import (  # noqa: E402
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    map_unmapped_fragment,
    mapped_atom_numbers,
    mapped_state_signature,
)
from mechet.natural_language_electron_flow import compile_event_arguments  # noqa: E402


SOURCE_COMMIT = "4fefc016fd4d4e4305dec92586e593bafe3b3e5b"
SOURCE_SHA256 = "84b3d907cc1595269e34ba34be0163c4863f4314f646cd768b44967648210a29"
RELEASE_MANIFEST_SHA256 = "5161fdcf96b2adc1cebdf51f8fdcc7691151a87f66226a43062a0fad0aebe5c2"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_VERSION = "v0.4.1"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _mapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError("invalid mapped SMILES")
    maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if not maps or any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
        raise ValueError("invalid atom maps")
    return molecule


def principal_product(products: str) -> tuple[str, list[str]]:
    """Apply the existing largest-organic-fragment policy, but reject ties."""

    fragments = [part for part in products.split(".") if part]
    if not fragments:
        raise ValueError("empty product mixture")
    scores: list[tuple[int, int, int]] = []
    for fragment in fragments:
        molecule = _mapped_mol(fragment)
        carbons = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
        scores.append((int(carbons > 0), molecule.GetNumHeavyAtoms(), carbons))
    best = max(scores)
    if scores.count(best) != 1:
        raise ValueError("ambiguous principal product")
    index = scores.index(best)
    return fragments[index], [fragment for i, fragment in enumerate(fragments) if i != index]


def inverse_moves(epd: list[list[Any]]) -> list[dict[str, Any]]:
    def container(atoms: list[int], *, source: bool) -> dict[str, Any]:
        if len(atoms) not in {1, 2}:
            raise ValueError("unsupported electron container")
        kind = ("LP" if source else "ATOM") if len(atoms) == 1 else "BOND"
        return {"kind": kind, "atoms": [int(atom) for atom in atoms]}

    moves = []
    for code, forward_source, forward_sink in reversed(epd):
        if not isinstance(code, str) or "/" not in code:
            raise ValueError("unsupported arrow code")
        moves.append({
            "source": container(forward_sink, source=True),
            "sink": container(forward_source, source=False),
            "electrons": 2,
        })
    if not moves:
        raise ValueError("empty electron program")
    return moves


def audit_public_decisions(target: str, reactants: str, decisions: list[dict[str, Any]]) -> None:
    """Replay only emitted public calls; never read the private source moves."""

    current = target
    next_private_map = max(mapped_atom_numbers(target)) + 1
    target_visible = deterministic_unmapped_state(target).text
    for row in decisions:
        user_prompt = str(row["messages"][1]["content"])
        visible = deterministic_unmapped_state(current).text
        if f"TARGET PRODUCT SMILES: {target_visible}\n" not in user_prompt:
            raise ValueError("public target/prompt drift")
        if f"CURRENT STATE SMILES: {visible}\n" not in user_prompt:
            raise ValueError("public state/prompt drift")
        call = row["messages"][2]["tool_calls"][0]["function"]
        name, arguments = call["name"], call["arguments"]
        observed = json.loads(row["messages"][3]["content"])
        if name == "import_fragments":
            imported = []
            for fragment in arguments["fragments"]:
                for _ in range(int(fragment["count"])):
                    mapped, next_private_map = map_unmapped_fragment(
                        str(fragment["smiles"]), first_map=next_private_map
                    )
                    imported.append(mapped)
            current = append_mapped_fragments_verbatim(current, imported)
            if observed.get("current_state") != deterministic_unmapped_state(current).text:
                raise ValueError("public import replay drift")
        elif name == "apply_electron_flow":
            compiled = compile_event_arguments(current, arguments)
            outcome = verify_electron_step(current, compiled)
            if not outcome.get("ok"):
                raise ValueError("public event replay failed")
            current = str(outcome["state_smiles"])
            if observed.get("current_state") != deterministic_unmapped_state(current).text:
                raise ValueError("public event successor drift")
        elif name == "finish_trace":
            if observed.get("derived_precursor") != visible:
                raise ValueError("public finish endpoint drift")
        else:
            raise ValueError("unexpected public tool")
    # Imported public fragments receive fresh runtime-private map IDs, so
    # source-map equality is not an inference-time endpoint condition.
    if deterministic_unmapped_state(current).text != deterministic_unmapped_state(reactants).text:
        raise ValueError("public trace endpoint mismatch")


def convert_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    source_id = int(record["id"])
    reactants, products = str(record["rsmi"]).split(">>")
    target, imports = principal_product(products)
    # This is a complete-mixture check. It is not enough by itself to claim
    # product-only supervision; the existing v2 builder separately compiles
    # each public action against the actual partial-state observation.
    if mapped_atom_numbers(reactants) != mapped_atom_numbers(products):
        raise ValueError("mapped atoms changed between sides")
    moves = inverse_moves(record["epd"])
    replay = verify_electron_step(products, moves)
    if not replay.get("ok") or mapped_state_signature(str(replay.get("state_smiles") or "")) != mapped_state_signature(reactants):
        raise ValueError("inverse complete-mixture replay mismatch")
    trace_source = {
        "source_id": f"synepd:{SOURCE_VERSION}:{source_id}",
        "target_smiles": target,
        "full_precursor_state": reactants,
        "messages": [{
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "import_fragment", "arguments": {"fragment_smiles": fragment}}
            } for fragment in imports],
        }],
        "metadata": {
            "trace_plan": {"steps": [{"moves": moves, "state_after": reactants}]},
            "program_version": f"synepd_{SOURCE_VERSION}_reverse",
        },
    }
    decisions = convert_row(trace_source)
    kinds = [row["metadata"]["decision_type"] for row in decisions]
    if kinds != (["import"] if imports else []) + ["event", "finish"]:
        raise ValueError("unexpected decision sequence")
    audit_public_decisions(target, reactants, decisions)
    family = ".".join(str(record["tax_code"]).split(".")[:2])
    for row in decisions:
        row["metadata"].update({
            "source_dataset": "SynEPD",
            "source_record_id": f"synepd:{SOURCE_VERSION}:{source_id}",
            "source_license": SOURCE_LICENSE,
            "source_release": SOURCE_VERSION,
            "source_commit": SOURCE_COMMIT,
            "mechanism_class": family,
            "mechanism_tax_code": str(record["tax_code"]),
            "principal_product_policy": "unique_largest_organic_fragment",
        })
    return decisions


def build(source: Path, release_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("SynEPD polar.json differs from pinned source")
    if sha256(release_manifest) != RELEASE_MANIFEST_SHA256:
        raise ValueError("SynEPD release manifest differs from pinned source")
    release = json.loads(release_manifest.read_text(encoding="utf-8"))["dataset_release"]
    if release.get("license") != SOURCE_LICENSE or release.get("version") != SOURCE_VERSION:
        raise ValueError("SynEPD release/license mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload["records"]
    if payload.get("count") != len(records) or len(records) != 1926:
        raise ValueError("SynEPD record count changed")
    output.mkdir(parents=True)
    accepted_path = output / "train.jsonl"
    quarantine_path = output / "train.quarantine.jsonl"
    counts: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    seen: set[int] = set()
    with accepted_path.open("w", encoding="utf-8") as accepted, quarantine_path.open("w", encoding="utf-8") as rejected:
        for record in records:
            record_id = int(record["id"])
            if record_id in seen:
                raise ValueError(f"duplicate source record id {record_id}")
            seen.add(record_id)
            try:
                decisions = convert_record(record)
            except (KeyError, TypeError, ValueError) as exc:
                code = str(exc).split(":", 1)[0]
                counts[f"quarantine:{code}"] += 1
                rejected.write(json.dumps({"source_record_id": record_id, "reason": code}, ensure_ascii=False) + "\n")
                continue
            counts["accepted_reactions"] += 1
            counts["accepted_decisions"] += len(decisions)
            classes[decisions[0]["metadata"]["mechanism_class"]] += len(decisions)
            for decision in decisions:
                accepted.write(json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n")
    if counts["accepted_reactions"] + sum(value for key, value in counts.items() if key.startswith("quarantine:")) != len(records):
        raise ValueError("source coverage lost")
    manifest = {
        "artifact_type": "curated_state_sft_candidate",
        "source_dataset": "SynEPD",
        "source_release": SOURCE_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_license": SOURCE_LICENSE,
        "source_polar_sha256": SOURCE_SHA256,
        "source_release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "principal_product_policy": "unique_largest_organic_fragment",
        "executor_semantics": "unchanged_mechet_two_electron_polar",
        "splits": {"train": {"rows": counts["accepted_decisions"], "reactions": counts["accepted_reactions"], "output_sha256": sha256(accepted_path)}},
        "quarantine": {"rows": len(records) - counts["accepted_reactions"], "sha256": sha256(quarantine_path), "reasons": dict(sorted((key.removeprefix("quarantine:"), value) for key, value in counts.items() if key.startswith("quarantine:")))},
        "decision_rows_by_mechanism_family": dict(sorted(classes.items())),
        "evaluation_status": "training_source_only_not_R4",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "status": "validated_candidate",
        "training_allowed": True,
        "accepted_reactions": counts["accepted_reactions"],
        "accepted_decisions": counts["accepted_decisions"],
        "quarantined_reactions": len(records) - counts["accepted_reactions"],
        "source_sha256": SOURCE_SHA256,
        "note": "Scientific campaign sampling still requires all R1-R5 evaluation sources to be frozen and decontaminated.",
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.release_manifest, args.output), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
