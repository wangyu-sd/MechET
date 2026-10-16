"""Streaming descriptive histograms with explicit missing and exposure groups."""
from collections import Counter


def summarize_enrichment(edge_rows, program_rows, planning_rows, *, run_id):
    counts = {key: Counter() for key in ["flow_delta_status", "changed_atoms", "changed_bonds", "cycle_rank_delta", "net_charge_delta",
        "source_sink_types", "component_moves", "program_classes", "primitive_frequency", "planning_visits"]}
    planning_missing = 0
    for edge in edge_rows:
        if edge["run_id"] != run_id:
            continue
        counts["flow_delta_status"][edge["delta_status"]] += 1
        delta = edge.get("delta")
        if delta:
            counts["changed_atoms"][str(len(delta["changed_atom_maps"]))] += 1
            counts["changed_bonds"][str(len(delta["changed_bonds"]))] += 1
            for key in ["cycle_rank_delta", "net_charge_delta"]:
                counts[key][str(delta[key])] += 1
        if edge.get("source_sink_types") is not None:
            counts["source_sink_types"].update(edge["source_sink_types"])
            counts["component_moves"][str(len(edge["source_sink_types"]))] += 1
    for program in program_rows:
        if program["run_id"] != run_id:
            continue
        if not program.get("actor_exposure_authenticated") or not program.get("complete_signature"):
            counts["program_classes"]["unavailable_or_unauthenticated"] += 1
            continue
        group = "seen_ordered_signature" if program["complete_signature_training_count"] > 0 else "unseen_ordered_signature"
        counts["program_classes"][group] += 1
        freq = program.get("minimum_primitive_training_frequency")
        if freq is not None:
            counts["primitive_frequency"]["0" if freq == 0 else "1–4" if freq < 5 else "5–99" if freq < 100 else "100+"] += 1
    for state in planning_rows:
        if state["run_id"] != run_id:
            continue
        if state.get("visits") is None:
            planning_missing += 1
        else:
            counts["planning_visits"][str(state["visits"])] += 1
    return {"run_id": run_id, **{key: dict(value) for key, value in counts.items()},
        "planning_visits_missing": planning_missing, "route_C1_claim": False,
        "counting_unit": "per-target native snapshots; shared contexts counted again across targets; ordinary component moves are not whole FLOW calls",
        "experimental_chemical_validity_claim": False}
