#!/usr/bin/env python3
"""Prepare and validate provenance-bound textbook knowledge extractions.

This script deliberately creates *candidate* knowledge, never released chemistry
rules.  ``prepare`` turns the frozen passage corpus into deterministic extraction
tasks.  ``validate`` schema-checks model responses and binds them back to the
exact passage hash.  Chemical review and executor replay remain separate gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_store import TextbookPassage, TextbookStore


PROTOCOL_VERSION = "textbook-knowledge-extraction-v1"

REACTION_TOPICS = {
    "acid_base",
    "addition",
    "aromatic",
    "carbonyl",
    "elimination",
    "oxidation_reduction",
    "pericyclic",
    "photochemistry",
    "proton_transfer",
    "radical",
    "rearrangement",
    "substitution",
}
PHYSICAL_ORGANIC_TOPICS = {
    "bonding_and_structure",
    "electrochemistry",
    "kinetics",
    "solvent_effects",
    "stereochemistry",
    "thermodynamics",
}
ANALYTICAL_TOPICS = {
    "chromatography",
    "fragmentation",
    "ionization",
    "mass_spectrometry",
    "spectroscopy",
}

EXTRACTION_SCHEMA: dict[str, Any] = {
    "candidate_name": "string | UNKNOWN",
    "knowledge_type": (
        "terminology | mechanistic_principle | reaction_family | physical_organic | "
        "analytical_or_gas_phase | scope_or_warning | UNKNOWN"
    ),
    "explicit_claims": [
        {
            "claim": "independently worded claim supported by the passage",
            "evidence_quote": "short exact supporting span or UNKNOWN",
            "support": "explicit | inferred | absent",
        }
    ],
    "participants": [
        {
            "semantic_role": "nucleophile/electrophile/etc. or UNKNOWN",
            "description": "text description or UNKNOWN",
            "support": "explicit | inferred | absent",
        }
    ],
    "electron_moves": [
        {
            "source_kind": "LP | BOND | ATOM | UNKNOWN",
            "source_roles": ["semantic role"],
            "sink_kind": "LP | BOND | ATOM | UNKNOWN",
            "sink_roles": ["semantic role"],
            "electrons": "1 | 2 | UNKNOWN",
            "support": "explicit | inferred | absent",
        }
    ],
    "preconditions": ["only conditions explicitly stated in the passage"],
    "warnings_or_exceptions": ["only boundaries explicitly stated in the passage"],
    "competing_pathways": ["only competitors explicitly stated in the passage"],
    "stereochemical_effects": ["only effects explicitly stated in the passage"],
    "phase_and_medium": {
        "phase": "solution | gas | condensed | surface | unspecified | UNKNOWN",
        "solvent_or_medium": "string | UNKNOWN",
        "temperature_or_pressure": "string | UNKNOWN",
        "support": "explicit | inferred | absent",
    },
    "analytical_context": {
        "ionization_method": "string | UNKNOWN",
        "ion_or_radical_state": "string | UNKNOWN",
        "instrument_or_collision_context": "string | UNKNOWN",
        "support": "explicit | inferred | absent",
    },
    "uncertain_fields": ["every field requiring inference or chemistry review"],
    "review_notes": ["possible ambiguity, scope limitation, or missing context"],
}

SYSTEM_PROMPT = """You extract candidate organic-chemistry knowledge from one bounded textbook passage.
Return one JSON object matching extraction_schema. Use only the supplied passage.
Use UNKNOWN when the passage does not support a field. Distinguish an explicit
statement from an inference. Do not invent atom mappings, SMARTS, electron arrows,
conditions, selectivity, kinetics, feasibility, gas-phase behavior, or a complete
mechanism. Keep evidence quotes short. Write claims independently rather than
copying long textbook prose. The result is unreviewed soft evidence: it is not an
answer, an executor rule, or chemical ground truth."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _license_layers(spec: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for layer_name, layer in dict(spec.get("layers") or {}).items():
        if not isinstance(layer, Mapping) or layer.get("include_in_corpus") is False:
            continue
        for license_name in layer.get("allowed_licenses") or ():
            license_name = str(license_name)
            previous = output.get(license_name)
            if previous and previous != str(layer_name):
                raise ValueError(
                    f"license {license_name!r} appears in multiple corpus layers"
                )
            output[license_name] = str(layer_name)
    if not output:
        raise ValueError("corpus spec declares no allowed licenses")
    return output


def _profile(passage: TextbookPassage) -> str:
    topics = set(passage.topics)
    modalities = set(passage.modalities)
    phases = set(passage.phases)
    if "gas_phase" in phases or "mass_spectrometry" in modalities or topics & {
        "fragmentation",
        "ionization",
        "mass_spectrometry",
    }:
        return "gas_phase_and_mass_spectrometry"
    if topics & REACTION_TOPICS or "mechanism" in modalities:
        return "reaction_mechanism"
    if topics & PHYSICAL_ORGANIC_TOPICS:
        return "physical_organic"
    if topics & ANALYTICAL_TOPICS or "spectroscopy" in modalities:
        return "analytical_chemistry"
    return "general_organic_chemistry"


def _candidate_id(passage: TextbookPassage) -> str:
    identity = f"{PROTOCOL_VERSION}|{passage.passage_id}|{passage.evidence_sha256}"
    return "textbook-knowledge:" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _user_prompt(passage: TextbookPassage, profile: str) -> str:
    context = {
        "extraction_profile": profile,
        "candidate_topic_tags": list(passage.topics),
        "candidate_phase_tags": list(passage.phases),
        "candidate_modality_tags": list(passage.modalities),
        "warning": "tags are retrieval candidates, not chemical labels",
    }
    return (
        "EXTRACTION CONTEXT\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
        + "\n\nEXTRACTION SCHEMA\n"
        + json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, sort_keys=True)
        + "\n\nBOUNDED EVIDENCE PASSAGE\n"
        + passage.text
    )


def make_task(
    passage: TextbookPassage,
    *,
    corpus_digest: str,
    corpus_sha256: str,
    license_layer: str,
) -> dict[str, Any]:
    profile = _profile(passage)
    metadata = dict(passage.metadata or {})
    return {
        "artifact_type": "textbook_knowledge_extraction_task",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": _candidate_id(passage),
        "passage_id": passage.passage_id,
        "passage_sha256": passage.evidence_sha256,
        "corpus_digest": corpus_digest,
        "corpus_sha256": corpus_sha256,
        "source": {
            "source_id": passage.source_id,
            "locator": passage.locator,
            "revision": passage.revision,
            "license": passage.license,
            "license_layer": license_layer,
            "source_url": passage.source_url,
            "artifact_path": metadata.get("artifact_path"),
            "artifact_sha256": metadata.get("artifact_sha256"),
        },
        "extraction_profile": profile,
        "candidate_tags": {
            "topics": list(passage.topics),
            "reaction_families": list(passage.reaction_families),
            "functional_groups": list(passage.functional_groups),
            "phases": list(passage.phases),
            "modalities": list(passage.modalities),
        },
        "evidence_span": passage.text,
        "extraction_schema": EXTRACTION_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(passage, profile)},
        ],
        "status": "unreviewed_extraction_task",
        "candidate_evidence_only": True,
        "released_knowledge_anchor": False,
        "formal_validity_source": False,
    }


def _load_manifest(path: Path, store: TextbookStore) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing corpus manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    observed = store.manifest()
    for key in ("n_passages", "corpus_digest"):
        if manifest.get(key) != observed.get(key):
            raise ValueError(
                f"corpus manifest {key} mismatch: "
                f"declared={manifest.get(key)!r}, observed={observed.get(key)!r}"
            )
    return manifest


def prepare(args: argparse.Namespace) -> int:
    store = TextbookStore.load(args.corpus)
    corpus_manifest = _load_manifest(args.corpus_manifest, store)
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8")) or {}
    layers = _license_layers(spec)
    corpus_sha256 = _sha256_file(args.corpus)
    tasks: list[dict[str, Any]] = []
    skipped = Counter()
    for passage in sorted(store.passages, key=lambda item: item.passage_id):
        layer = layers.get(passage.license)
        if not layer:
            raise ValueError(
                f"passage {passage.passage_id} has license outside protocol allowlist: "
                f"{passage.license!r}"
            )
        if layer == "noncommercial_research" and not args.accept_noncommercial:
            skipped["noncommercial_requires_explicit_acceptance"] += 1
            continue
        section_kind = str((passage.metadata or {}).get("section_kind") or "").lower()
        if any(token in section_kind for token in ("exercise", "answer", "solution")):
            skipped["evaluation_or_answer_content"] += 1
            continue
        tasks.append(
            make_task(
                passage,
                corpus_digest=str(corpus_manifest["corpus_digest"]),
                corpus_sha256=corpus_sha256,
                license_layer=layer,
            )
        )

    if args.limit:
        tasks = _balanced_prefix(tasks, args.limit)
    if not tasks:
        raise ValueError("no eligible textbook passages under the selected protocol")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")

    profiles = Counter(str(task["extraction_profile"]) for task in tasks)
    sources = Counter(str(task["source"]["source_id"]) for task in tasks)
    licenses = Counter(str(task["source"]["license"]) for task in tasks)
    task_digest = hashlib.sha256(
        "\n".join(str(task["candidate_id"]) for task in tasks).encode()
    ).hexdigest()
    output_manifest = {
        "artifact_type": "textbook_knowledge_extraction_manifest",
        "protocol_version": PROTOCOL_VERSION,
        "status": "unreviewed_candidate_queue",
        "candidate_evidence_only": True,
        "released_knowledge_anchors": 0,
        "chemical_review_required": True,
        "executor_replay_required_for_executable_promotion": True,
        "input": {
            "corpus": str(args.corpus),
            "corpus_sha256": corpus_sha256,
            "corpus_manifest": str(args.corpus_manifest),
            "corpus_manifest_sha256": _sha256_file(args.corpus_manifest),
            "corpus_digest": corpus_manifest["corpus_digest"],
            "spec": str(args.spec),
            "spec_sha256": _sha256_file(args.spec),
        },
        "selection": {
            "accept_noncommercial": bool(args.accept_noncommercial),
            "limit": int(args.limit),
            "ordering": (
                "deterministic_balanced_round_robin_by_profile_then_passage_id"
                if args.limit
                else "deterministic_passage_id"
            ),
        },
        "protocol": {
            "extraction_schema_sha256": hashlib.sha256(
                json.dumps(EXTRACTION_SCHEMA, sort_keys=True).encode()
            ).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "extractor_script_sha256": _sha256_file(Path(__file__)),
        },
        "n_tasks": len(tasks),
        "task_id_digest": task_digest,
        "output": str(args.output),
        "output_sha256": _sha256_file(args.output),
        "distributions": {
            "profiles": dict(sorted(profiles.items())),
            "sources": dict(sorted(sources.items())),
            "licenses": dict(sorted(licenses.items())),
        },
        "skipped": dict(sorted(skipped.items())),
    }
    output_manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, indent=2, ensure_ascii=False))
    return 0


def _balanced_prefix(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a deterministic profile-balanced prefix for smoke/review queues."""

    if limit <= 0 or limit >= len(tasks):
        return tasks
    buckets: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        buckets.setdefault(str(task["extraction_profile"]), []).append(task)
    output: list[dict[str, Any]] = []
    index = 0
    names = sorted(buckets)
    while len(output) < limit:
        progressed = False
        for name in names:
            if index < len(buckets[name]):
                output.append(buckets[name][index])
                progressed = True
                if len(output) == limit:
                    break
        if not progressed:
            break
        index += 1
    return output


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def _extraction_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("extraction")
    if isinstance(value, Mapping):
        return value
    value = row.get("response")
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError("response must contain an object in extraction or JSON in response")


def _exact_keys(value: Any, expected: Iterable[str], field: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{field} must be an object"]
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    extra = sorted(set(value) - expected_set)
    return [f"{field} keys mismatch; missing={missing}, extra={extra}"] if missing or extra else []


def _validate_extraction_schema(value: Mapping[str, Any]) -> list[str]:
    errors = _exact_keys(value, EXTRACTION_SCHEMA, "extraction")
    if errors:
        return errors
    for field in ("candidate_name", "knowledge_type"):
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            errors.append(f"{field} must be a non-empty string or UNKNOWN")
    allowed_knowledge = {
        "terminology",
        "mechanistic_principle",
        "reaction_family",
        "physical_organic",
        "analytical_or_gas_phase",
        "scope_or_warning",
        "UNKNOWN",
    }
    if value.get("knowledge_type") not in allowed_knowledge:
        errors.append(f"knowledge_type must be one of {sorted(allowed_knowledge)}")
    for field in (
        "explicit_claims",
        "participants",
        "electron_moves",
        "preconditions",
        "warnings_or_exceptions",
        "competing_pathways",
        "stereochemical_effects",
        "uncertain_fields",
        "review_notes",
    ):
        if not isinstance(value.get(field), list):
            errors.append(f"{field} must be a list")
    for field in (
        "preconditions",
        "warnings_or_exceptions",
        "competing_pathways",
        "stereochemical_effects",
        "uncertain_fields",
        "review_notes",
    ):
        if isinstance(value.get(field), list) and not all(
            isinstance(item, str) for item in value[field]
        ):
            errors.append(f"{field} entries must be strings")
    allowed_support = {"explicit", "inferred", "absent"}
    if isinstance(value.get("explicit_claims"), list):
        for index, item in enumerate(value["explicit_claims"]):
            field = f"explicit_claims[{index}]"
            errors.extend(_exact_keys(item, ("claim", "evidence_quote", "support"), field))
            if isinstance(item, Mapping):
                if item.get("support") not in allowed_support:
                    errors.append(f"{field}.support is invalid")
                for key in ("claim", "evidence_quote"):
                    if not isinstance(item.get(key), str):
                        errors.append(f"{field}.{key} must be a string")
    if isinstance(value.get("participants"), list):
        for index, item in enumerate(value["participants"]):
            field = f"participants[{index}]"
            errors.extend(
                _exact_keys(item, ("semantic_role", "description", "support"), field)
            )
            if isinstance(item, Mapping) and item.get("support") not in allowed_support:
                errors.append(f"{field}.support is invalid")
    if isinstance(value.get("electron_moves"), list):
        move_fields = (
            "source_kind",
            "source_roles",
            "sink_kind",
            "sink_roles",
            "electrons",
            "support",
        )
        for index, item in enumerate(value["electron_moves"]):
            field = f"electron_moves[{index}]"
            errors.extend(_exact_keys(item, move_fields, field))
            if not isinstance(item, Mapping):
                continue
            if item.get("source_kind") not in {"LP", "BOND", "ATOM", "UNKNOWN"}:
                errors.append(f"{field}.source_kind is invalid")
            if item.get("sink_kind") not in {"LP", "BOND", "ATOM", "UNKNOWN"}:
                errors.append(f"{field}.sink_kind is invalid")
            if item.get("electrons") not in {1, 2, "1", "2", "UNKNOWN"}:
                errors.append(f"{field}.electrons is invalid")
            if item.get("support") not in allowed_support:
                errors.append(f"{field}.support is invalid")
            for key in ("source_roles", "sink_roles"):
                if not isinstance(item.get(key), list) or not all(
                    isinstance(role, str) for role in item.get(key, [])
                ):
                    errors.append(f"{field}.{key} must be a list of strings")
    object_fields = {
        "phase_and_medium": (
            "phase",
            "solvent_or_medium",
            "temperature_or_pressure",
            "support",
        ),
        "analytical_context": (
            "ionization_method",
            "ion_or_radical_state",
            "instrument_or_collision_context",
            "support",
        ),
    }
    for field, keys in object_fields.items():
        item = value.get(field)
        errors.extend(_exact_keys(item, keys, field))
        if isinstance(item, Mapping) and item.get("support") not in allowed_support:
            errors.append(f"{field}.support is invalid")
    if isinstance(value.get("phase_and_medium"), Mapping) and value[
        "phase_and_medium"
    ].get("phase") not in {
        "solution",
        "gas",
        "condensed",
        "surface",
        "unspecified",
        "UNKNOWN",
    }:
        errors.append("phase_and_medium.phase is invalid")
    return errors


def validate(args: argparse.Namespace) -> int:
    tasks = {str(row.get("candidate_id")): row for row in _load_jsonl(args.tasks)}
    if "" in tasks or not tasks:
        raise ValueError("task file is empty or contains a missing candidate_id")
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, row in enumerate(_load_jsonl(args.responses), 1):
        candidate_id = str(row.get("candidate_id") or "")
        task = tasks.get(candidate_id)
        if task is None:
            errors.append(f"line {line_number}: unknown candidate_id {candidate_id!r}")
            continue
        if candidate_id in seen:
            errors.append(f"line {line_number}: duplicate candidate_id {candidate_id!r}")
            continue
        seen.add(candidate_id)
        try:
            extraction = dict(_extraction_payload(row))
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        schema_errors = _validate_extraction_schema(extraction)
        if schema_errors:
            errors.append(f"line {line_number}: " + "; ".join(schema_errors))
            continue
        if row.get("passage_sha256") not in (None, task["passage_sha256"]):
            errors.append(f"line {line_number}: passage_sha256 mismatch")
            continue
        accepted.append(
            {
                "artifact_type": "textbook_knowledge_extraction_candidate",
                "protocol_version": PROTOCOL_VERSION,
                "candidate_id": candidate_id,
                "passage_id": task["passage_id"],
                "passage_sha256": task["passage_sha256"],
                "source": task["source"],
                "extraction_profile": task["extraction_profile"],
                "extraction": extraction,
                "model_metadata": dict(row.get("model_metadata") or {}),
                "status": "unreviewed_model_extraction",
                "candidate_evidence_only": True,
                "released_knowledge_anchor": False,
                "formal_validity_source": False,
            }
        )

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"response validation failed ({len(errors)} errors):\n{preview}")
    if not accepted:
        raise ValueError("no valid model extractions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "artifact_type": "textbook_knowledge_validation_report",
        "protocol_version": PROTOCOL_VERSION,
        "n_tasks": len(tasks),
        "n_responses": len(seen),
        "n_valid_candidates": len(accepted),
        "missing_response_count": len(set(tasks) - seen),
        "output": str(args.output),
        "output_sha256": _sha256_file(args.output),
        "chemical_review_required": True,
        "released_knowledge_anchors": 0,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="write extraction tasks")
    prepare_parser.add_argument("--corpus", type=Path, required=True)
    prepare_parser.add_argument("--corpus-manifest", type=Path, required=True)
    prepare_parser.add_argument(
        "--spec", type=Path, default=Path("knowledge/corpus_v2_spec.yaml")
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path)
    prepare_parser.add_argument(
        "--accept-noncommercial",
        action="store_true",
        help="include the physically separate CC-BY-NC-SA research layer",
    )
    prepare_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="profile-balanced deterministic limit; zero keeps every eligible passage",
    )
    prepare_parser.set_defaults(func=prepare)

    validate_parser = commands.add_parser(
        "validate", help="validate model JSON responses without promoting them"
    )
    validate_parser.add_argument("--tasks", type=Path, required=True)
    validate_parser.add_argument("--responses", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.set_defaults(func=validate)
    return root


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "limit", 0) < 0:
        raise ValueError("--limit must be non-negative")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
