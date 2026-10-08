#!/usr/bin/env python3
"""Audit whether endpoint-context imports affect reference electron-flow chemistry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import mapped_atom_numbers, mapped_state_signature
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.audit_reliable_product_mapping_parity import sha256
from scripts.earho_v2_protocol import decision_action, replay_reference
from scripts.run_natural_language_value_search import read_selected


class ContextChemicallyActive(ValueError):
    """A labelled context atom became connected to a retained component."""


def strip_context_components(
    state: str, context_maps: set[int], *, decision_index: int | None = None,
) -> str:
    """Remove whole context components without reserializing retained atoms."""
    kept = []
    for component in state.split("."):
        atom_maps = mapped_atom_numbers(component)
        if atom_maps & context_maps:
            if atom_maps - context_maps:
                raise ContextChemicallyActive(
                    "context merged with an electron-participant component"
                    f" at decision {decision_index}; context maps "
                    f"{sorted(atom_maps & context_maps)}; retained maps "
                    f"{sorted(atom_maps - context_maps)}"
                )
        else:
            kept.append(component)
    if not kept:
        raise ValueError("context projection removed the entire molecular state")
    return ".".join(kept)


def project_reference(
    source: dict[str, Any], decisions: list[dict[str, Any]],
) -> Counter[str]:
    """Replay gold, then test its electron moves on context-free mapped states."""
    reference = replay_reference(source, decisions, compact_history=False)
    context_maps: set[int] = set()
    counts: Counter[str] = Counter()
    for index, decision in enumerate(decisions):
        name, arguments, _ = decision_action(decision)
        before = reference.nodes[index].state
        after = reference.nodes[index + 1].state
        if name == "import_fragments":
            expanded = [
                item for item in arguments["fragments"]
                for _ in range(int(item["count"]))
            ]
            if not after.startswith(before + "."):
                raise ValueError(f"import did not append components at decision {index}")
            appended = after[len(before) + 1 :].split(".")
            if len(appended) != len(expanded):
                raise ValueError(f"imported fragment/component count differs at {index}")
            for item, component in zip(expanded, appended, strict=True):
                if item["purpose"] == "endpoint_context":
                    context_maps.update(mapped_atom_numbers(component))
                    counts["context_copies_removed"] += 1
                elif item["purpose"] != "electron_participant":
                    raise ValueError(f"unknown fragment purpose at {index}")
        elif name == "apply_electron_flow":
            # Resolve Axx aliases in the *original* inventory, then execute the
            # identical private-map electron moves after removing spectators.
            moves = compile_event_arguments(before, arguments)
            projected_before = strip_context_components(
                before, context_maps, decision_index=index,
            )
            projected_after = strip_context_components(
                after, context_maps, decision_index=index,
            )
            replay = verify_electron_step(projected_before, moves)
            if not replay.get("ok"):
                raise ValueError(
                    f"projected electron event failed at {index}: {replay.get('code')}"
                )
            if (mapped_state_signature(str(replay["state_smiles"]))
                    != mapped_state_signature(projected_after)):
                raise ValueError(f"projected successor differs at decision {index}")
            counts["projected_events_ok"] += 1
    final = strip_context_components(reference.nodes[-1].state, context_maps)
    structural = split_precursor_endpoints(final, str(source["target_smiles"])).structural
    if not structural_exact(structural, str(source["structural_precursor"])):
        raise ValueError("projected structural precursor differs from frozen reference")
    counts["projected_structural_exact"] += 1
    return counts


def audit(
    *, source: Path, source_manifest: Path,
    decisions: Path, decision_manifest: Path,
    n_reactions: int, seed: int = 17,
) -> dict[str, Any]:
    source_info = json.loads(source_manifest.read_text(encoding="utf-8"))
    decision_info = json.loads(decision_manifest.read_text(encoding="utf-8"))
    if n_reactions < 1 or n_reactions > int(source_info["splits"]["valid"]["rows"]):
        raise ValueError("invalid validation reaction count")
    source_sha, decision_sha = sha256(source), sha256(decisions)
    if source_sha != source_info["splits"]["valid"]["sha256"]:
        raise ValueError("source validation SHA-256 mismatch")
    if decision_sha != decision_info["splits"]["valid"]["output_sha256"]:
        raise ValueError("decision validation SHA-256 mismatch")
    selected = read_selected(source, n_reactions, seed)
    wanted = {str(row["source_id"]) for row in selected}
    if len(selected) != n_reactions or len(wanted) != n_reactions:
        raise ValueError("selected reaction IDs are incomplete or repeated")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with decisions.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if str(row["source_id"]) in wanted:
                    by_source[str(row["source_id"])].append(row)
    if set(by_source) != wanted:
        raise ValueError("selected reactions lack frozen decisions")
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for index, row in enumerate(selected, 1):
        identifier = str(row["source_id"])
        gold = sorted(by_source[identifier], key=lambda item: int(item["metadata"]["decision_index"]))
        try:
            counts.update(project_reference(row, gold))
        except ContextChemicallyActive as error:
            counts["context_chemically_active"] += 1
            failures.append({"source_id": identifier, "error": f"{type(error).__name__}:{error}"})
        except Exception as error:
            counts["projection_failed"] += 1
            failures.append({"source_id": identifier, "error": f"{type(error).__name__}:{error}"})
        if index % 100 == 0:
            print({"checked": index, "projection_failed": counts["projection_failed"],
                   "context_chemically_active": counts["context_chemically_active"]}, flush=True)
    counts["reactions"] = n_reactions
    return {
        "artifact_type": "reliable_mechet_reference_context_projection_v1",
        "source": str(source), "source_sha256": source_sha,
        "decisions": str(decisions), "decisions_sha256": decision_sha,
        "n_reactions": n_reactions, "seed": seed,
        "counts": dict(counts), "failures": failures,
        "interpretation": (
            "Gold electron moves are resolved using original Axx aliases before "
            "counterfactual removal of endpoint-context components. This tests "
            "whether context affects executable chemistry and structural endpoint. "
            "Context-chemically-active cases are not safely projectable, not "
            "counterexamples to the exact result on the remaining cases; "
            "it is not a model rollout, a new training condition, or proof that "
            "literal Axx actions survive changed fragment inventories."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--decision-manifest", type=Path, required=True)
    parser.add_argument("--n-reactions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        source=args.source, source_manifest=args.source_manifest,
        decisions=args.decisions, decision_manifest=args.decision_manifest,
        n_reactions=args.n_reactions, seed=args.seed,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
