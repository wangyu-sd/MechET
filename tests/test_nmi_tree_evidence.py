"""Deterministic parser/metric contracts, not model experiments."""
import copy
import json

import pytest

from mechet.analysis.tree_records import normalize_native_tree, discover_inputs
from mechet.analysis.tree_metrics import candidate_metrics, morphology, route_closure
from mechet.analysis.chemical_evidence import validate_evidence, exposure_compatible


def native_fixture():
    # Equal display strings with distinct histories must stay separate nodes.
    contexts = [
        {"id": "r", "chemical_id": "chem", "state": "[CH3:1][OH:2]", "depth": 0, "actions": [], "terminal": False},
        {"id": "a", "chemical_id": "chem", "state": "[CH3:1][OH:2]", "depth": 1, "actions": [{"name": "import_fragments", "arguments": {"fragments": [{"smiles": "O", "count": 1, "purpose": "auxiliary"}]}}], "terminal": False},
        {"id": "b", "chemical_id": "other", "state": "[CH3:1].[OH:2]", "depth": 1, "actions": [{"name": "apply_electron_flow"}], "terminal": False},
        {"id": "t", "chemical_id": "other", "state": "[CH3:1].[OH:2]", "depth": 2, "actions": [{"name": "apply_electron_flow"}, {"name": "finish_trace"}], "terminal": True},
    ]
    def edge(i, parent, child, status, name, arguments=None):
        return {"id": i, "parent_context": parent, "child_context": child, "status": status, "error": "BAD" if status == "rejected" else "", "action": {"name": name, "arguments": arguments or {}, "tokens": 12, "logprob": -3.0}}
    edges = [edge("e0", "r", "a", "accepted", "import_fragments"),
             edge("e1", "r", "b", "accepted", "apply_electron_flow", {"electron_flow": [{"source": "s1", "destination": "d1"}, {"source": "s2", "destination": "d2"}]}),
             edge("e2", "r", None, "rejected", "apply_electron_flow"),
             edge("e3", "b", "t", "terminal", "finish_trace"),
             edge("e4", "r", "b", "duplicate_successor", "apply_electron_flow")]
    return {"case_id": "case:1", "target": "CO", "reaction_trees": {"CO": {"root_context": "r", "contexts": contexts, "chemical_nodes": [{"id": "chem", "state_unmapped": "CO"}, {"id": "other", "state_unmapped": "C.O"}], "edges": edges, "terminal_contexts": ["t"]}}, "reaction_edges": [], "mcts_graph": {}}


def normalized(tree=None):
    return normalize_native_tree(tree or native_fixture(), report={"id": "case:1"}, manifest={"full_denominator": 3}, source={"run_id": "run", "tree_sha256": "ab" * 32, "tree_path": "/original/tree.json"})


def test_equal_unmapped_states_keep_native_history_identity():
    data = normalized()
    nodes = data["nodes"]
    assert len(nodes) == 4
    assert nodes[0]["state_unmapped"] == nodes[1]["state_unmapped"] == "CO"
    assert nodes[0]["node_id"] != nodes[1]["node_id"]
    assert nodes[0]["history_sha256"] != nodes[1]["history_sha256"]
    assert nodes[0]["atom_maps"] == [1, 2]


def test_coupled_flow_is_one_edge_and_rejection_creates_no_node():
    data = normalized()
    assert len(data["attempted_edges"]) == 5
    coupled = next(e for e in data["attempted_edges"] if e["original_edge_id"] == "e1")
    rejected = next(e for e in data["attempted_edges"] if e["original_edge_id"] == "e2")
    assert coupled["component_move_count"] == 2
    assert rejected["child_node_id"] is None
    assert rejected["execution_accepted"] is False
    assert coupled["after_state"] == "[CH3:1].[OH:2]"
    assert len(data["terminals"]) == 1


def test_missing_child_is_partial_not_invented():
    t = native_fixture()
    t["reaction_trees"]["CO"]["contexts"] = t["reaction_trees"]["CO"]["contexts"][:-1]
    data = normalized(t)
    assert data["tree"]["has_complete_node_lineage"] is False
    assert len(data["nodes"]) == 3
    assert next(e for e in data["attempted_edges"] if e["original_edge_id"] == "e3")["child_node_id"] is None


def test_rejected_edge_with_child_is_rejected_as_corrupt():
    t = native_fixture()
    t["reaction_trees"]["CO"]["edges"][2]["child_context"] = "b"
    with pytest.raises(ValueError, match="rejected.*child"):
        normalized(t)


def test_morphology_counts_attempts_separately_from_successors():
    m = morphology(normalized())
    assert m["attempted_actions"] == 5
    assert m["rejected_attempts"] == 1
    assert m["duplicate_attempts"] == 1
    assert m["unique_context_transitions"] == 3
    assert m["expanded_contexts"] == 2
    assert m["branching_mean_expanded"] == 1.5
    assert m["terminal_contexts"] == 1


def test_candidate_pool_ranking_gap_same_pool_and_canonical_unique():
    rows = [{"structural_precursor": "C.O", "mean_nll": .1, "matches_reference": False},
            {"structural_precursor": "C.O", "mean_nll": .2, "matches_reference": False},
            {"structural_precursor": "CC", "mean_nll": .3, "matches_reference": True}]
    m = candidate_metrics(rows, has_reference=True)
    assert m["unique_candidates"] == 2
    assert m["reference_rank"] == 2
    assert m["top1"] is False and m["pool_recall"] is True
    assert m["ranking_gap"] == 1
    assert m["reciprocal_rank"] == .5


def test_empty_pool_with_and_without_reference():
    assert candidate_metrics([], has_reference=True)["pool_recall"] is False
    assert candidate_metrics([], has_reference=False)["pool_recall"] is None
    assert candidate_metrics([], has_reference=False)["ranking_gap"] is None


def test_ranker_never_uses_reference_to_reorder():
    a = [{"structural_precursor": "C", "mean_nll": .1, "matches_reference": False}, {"structural_precursor": "O", "mean_nll": .9, "matches_reference": True}]
    b = copy.deepcopy(a)
    b[0]["matches_reference"], b[1]["matches_reference"] = True, False
    assert candidate_metrics(a, has_reference=True)["ordered_precursors"] == candidate_metrics(b, has_reference=True)["ordered_precursors"] == ["C", "O"]


def test_stock_closure_requires_all_leaves_and_nontrivial_route():
    assert route_closure(["C", "O"], {"C"}, reaction_count=1)["all_leaves_in_stock"] is False
    assert route_closure(["C", "O"], {"C", "O"}, reaction_count=1)["stock_solved"] is True
    assert route_closure(["C"], {"C"}, reaction_count=0)["stock_solved"] is False
    assert route_closure(["C"], None, reaction_count=1)["stock_solved"] is None


def test_literature_is_for_complete_reactions_not_partial_flow():
    record = {"tier": "A", "evidence_status": "verified", "source_url": "https://example.org/real", "product_match": True, "precursor_match": True, "stereo_match": True, "source_overlap": "unknown"}
    with pytest.raises(ValueError, match="complete reaction"):
        validate_evidence(record, complete_reaction=False)
    assert validate_evidence(record, complete_reaction=True)["tier"] == "A"
    unreviewed = {**record, "evidence_status": "retrieved_unreviewed"}
    assert validate_evidence(unreviewed, complete_reaction=True)["verified_support"] is False


def test_no_source_found_is_not_conflicting_chemistry():
    with pytest.raises(ValueError, match="contradiction"):
        validate_evidence({"tier": "D", "evidence_status": "verified"}, complete_reaction=True)
    assert validate_evidence({"tier": "C", "evidence_status": "unresolved"}, complete_reaction=True)["verified_support"] is False


def test_h2_claim_requires_matching_training_exposure():
    assert exposure_compatible("H2", actor_exposure="FlowER_full") is False
    assert exposure_compatible("H2", actor_exposure="H2_train_only") is True


def test_discovery_uses_recorded_relative_resume_tree_path(tmp_path):
    old = tmp_path / "old" / "tree" / "rank00" / "trees"
    old.mkdir(parents=True)
    treepath = old / "case.json"
    treepath.write_text(json.dumps(native_fixture()))
    shard = tmp_path / "new" / "tree" / "rank00"
    (shard / "evaluations").mkdir(parents=True)
    (shard / "manifest.json").write_text(json.dumps({"full_denominator": 3}))
    (shard / "evaluations" / "case.json").write_text(json.dumps({"id": "case:1", "tree_artifact": "../../../old/tree/rank00/trees/case.json"}))
    found = discover_inputs([tmp_path / "new"])
    assert len(found) == 1
    assert found[0]["tree_path"] == str(treepath)
    assert found[0]["target_id"] == "case:1"


def test_cli_streams_provenance_tables_and_resumes_without_input_mutation(tmp_path):
    from scripts.analyze_nmi_tree_evidence import analyze
    import gzip
    root = tmp_path / "run"
    shard = root / "tree" / "rank00"
    (shard / "evaluations").mkdir(parents=True)
    (shard / "trees").mkdir()
    (shard / "manifest.json").write_text(json.dumps({"full_denominator": 3, "adapter_sha": "actor"}))
    tree_path = shard / "trees" / "one.json"
    tree_path.write_text(json.dumps(native_fixture()))
    original = tree_path.read_bytes()
    (shard / "evaluations" / "one.json").write_text(json.dumps({"id": "case:1", "tree_artifact": "trees/one.json", "expected_structural_precursor": "CC", "ranked_candidates": [{"context": "t", "structural_precursor": "CC", "mean_nll": .2, "matches_reference": True}]}))
    out = tmp_path / "out"
    m = analyze([root], out)
    assert m["runs"]["run"]["observed_targets"] == 1
    assert m["runs"]["run"]["benchmark_denominator"] == 3
    assert m["runs"]["run"]["candidate_ranking"]["top1_hits"] == 1
    with gzip.open(out / "nodes.jsonl.gz", "rt") as f:
        nodes = [json.loads(line) for line in f]
    assert len(nodes) == 4
    assert all(n["tree_id"] == nodes[0]["tree_id"] for n in nodes)
    inventory = json.loads((out / "input_inventory.json").read_text())
    assert inventory["inputs"][0]["tree_sha256"]
    assert tree_path.read_bytes() == original
    assert analyze([root], out, resume=True) == m
    with pytest.raises(FileExistsError):
        analyze([root], out)


def test_conflicting_reference_flags_are_not_silently_deduplicated():
    with pytest.raises(ValueError, match="contradictory"):
        candidate_metrics([{"structural_precursor": "C.O", "mean_nll": 1., "matches_reference": True}, {"structural_precursor": "O.C", "mean_nll": 2., "matches_reference": False}], has_reference=True)


def test_source_pointer_resolves_to_the_original_action():
    tree = native_fixture()
    edge = normalized(tree)["attempted_edges"][1]
    value = tree
    for part in edge["source_pointer"].strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    assert value["action"]["arguments"]["electron_flow"][1]["source"] == "s2"


def test_compressed_tables_are_byte_reproducible_across_clock_changes(tmp_path, monkeypatch):
    import gzip
    from scripts.analyze_nmi_tree_evidence import analyze
    root = tmp_path / "run"
    shard = root / "tree" / "rank00"
    (shard / "evaluations").mkdir(parents=True)
    (shard / "trees").mkdir()
    (shard / "trees" / "one.json").write_text(json.dumps(native_fixture()))
    (shard / "evaluations" / "one.json").write_text(json.dumps({"id": "case:1", "tree_artifact": "trees/one.json"}))
    monkeypatch.setattr(gzip.time, "time", lambda: 100)
    analyze([root], tmp_path / "out")
    first = (tmp_path / "out" / "nodes.jsonl.gz").read_bytes()
    monkeypatch.setattr(gzip.time, "time", lambda: 200)
    analyze([root], tmp_path / "out", resume=True)
    assert (tmp_path / "out" / "nodes.jsonl.gz").read_bytes() == first


def test_discovery_never_mixes_greedy_and_tree_conditions(tmp_path):
    for condition in ["greedy", "tree"]:
        shard = tmp_path / condition / "rank00"
        shard.mkdir(parents=True)
        (shard / "complete.json").write_text(json.dumps({"manifest": {"mode": condition}, "rows": [{"id": "same-target", "tree_artifact": "absent.json"}]}))
    records = discover_inputs([tmp_path])
    assert len(records) == 2
    assert len({r["run_id"] for r in records}) == 2


def test_rdkit_electron_arrows_are_grounded_in_recorded_private_maps():
    from scripts.render_nmi_tree_figures import molecule_svg
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import yaml
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    moves = [{"source": {"kind": "BOND", "atoms": [3, 9]}, "sink": {"kind": "ATOM", "atoms": [3]}, "electrons": 2}]
    svg, info = molecule_svg("[CH3:9][O:3][CH3:22]", moves, style, uid="unit", width=220, height=120)
    ET.fromstring(svg)
    assert info["source_maps"] == [[3, 9]]
    assert info["sink_maps"] == [[3]]
    assert info["electron_arrow_count"] == 1
    assert info["backend"] == "RDKit_MolDraw2DSVG"
    assert "data-electron-flow" in svg
    bad = [{"source": {"kind": "LP", "atoms": [999]}, "sink": {"kind": "ATOM", "atoms": [3]}}]
    with pytest.raises(ValueError, match="absent atom"):
        molecule_svg("[CH3:9][O:3][CH3:22]", bad, style, uid="unit", width=220, height=120)


def test_tree_layout_is_deterministic_and_retains_every_context():
    from scripts.render_nmi_tree_figures import layered_layout, select_flow_fork
    data = normalized()
    positions = layered_layout(data["nodes"], data["attempted_edges"])
    assert set(positions) == {n["node_id"] for n in data["nodes"]}
    assert positions == layered_layout(list(reversed(data["nodes"])), list(reversed(data["attempted_edges"])))
    assert positions[data["nodes"][0]["node_id"]][0] < positions[data["nodes"][3]["node_id"]][0]
    assert select_flow_fork(data) is None  # duplicate e4 is not a second choice.


def test_new_bond_arrow_targets_the_other_atom_not_empty_bond_midpoint():
    from scripts.render_nmi_tree_figures import molecule_svg
    from pathlib import Path
    import yaml
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    move = {"source": {"kind": "LP", "atoms": [3]}, "sink": {"kind": "BOND", "atoms": [3, 9]}}
    _, info = molecule_svg("[OH-:3].[CH3+:9]", [move], style, uid="newbond", width=200, height=120)
    assert info["arrow_geometry"][0]["end_anchor_atom"] == 9
    assert info["arrow_geometry"][0]["sink_bond_present"] is False


def test_observed_route_panel_keeps_pending_in_original_denominator():
    from mechet.analysis.tree_figure_data import route_panel_data
    entries = [{"run_id": "mcts", "target_id": "one", "report": {"stock_solved": True, "complete_reaction_edges": [{"id": "r1"}], "stock_leaves": ["C"]}},
               {"run_id": "mcts", "target_id": "two", "report": {"stock_solved": False, "already_in_stock": True}}]
    targets = [{"id": "one", "cohort": "c", "reference_depth": 2},
               {"id": "two", "cohort": "c", "reference_depth": 3},
               {"id": "three", "cohort": "c", "reference_depth": 3}]
    d = route_panel_data(entries, targets, run_id="mcts")
    assert d["cohorts"]["c"]["denominator"] == 3
    assert d["cohorts"]["c"]["observed_nonzero_stock_connected"] == 1
    assert d["cohorts"]["c"]["pending"] == 1
    assert d["cohorts"]["c"]["lower_bound_rate"] == 1 / 3
    assert d["cohorts"]["c"]["already_stock"] == 1
    assert d["depth"]["3"]["pending"] == 1


def test_dual_report_storage_exports_one_tree_and_one_target(tmp_path):
    from scripts.analyze_nmi_tree_evidence import analyze
    from mechet.analysis.tree_records import canonical_entries
    import gzip
    root = tmp_path / "run"
    shard = root / "tree" / "rank00"
    (shard / "trees").mkdir(parents=True)
    (shard / "evaluations").mkdir()
    (shard / "trees/one.json").write_text(json.dumps(native_fixture()))
    report = {"id": "case:1", "tree_artifact": "trees/one.json"}
    (shard / "evaluations/one.json").write_text(json.dumps(report))
    (shard / "complete.json").write_text(json.dumps({"manifest": {}, "rows": [report]}))
    analyze([root], tmp_path / "out")
    with gzip.open(tmp_path / "out/nodes.jsonl.gz", "rt") as f:
        nodes = [json.loads(line) for line in f]
    assert len(nodes) == 4
    inv = json.loads((tmp_path / "out/input_inventory.json").read_text())
    assert len(inv["inputs"]) == 2  # Retain both physical source records.
    assert len(canonical_entries(inv["inputs"], run_id="run")) == 1


def test_unreachable_context_marks_recorded_tree_partial():
    tree = native_fixture()
    tree["reaction_trees"]["CO"]["edges"] = [e for e in tree["reaction_trees"]["CO"]["edges"] if e["id"] != "e0"]
    assert normalized(tree)["tree"]["has_complete_node_lineage"] is False


def test_terminal_projection_preserves_structural_and_auxiliary_mixture():
    tree = native_fixture()
    c = tree["reaction_trees"]["CO"]["contexts"][-1]
    c["state"] = "[CH3:1][OH:2].[Na+:9]"
    data = normalized(tree)
    terminal = data["terminals"][0]
    assert terminal["structural_precursor"] == "CO"
    assert terminal["structural_precursor_mapped"] == "[CH3:1][OH:2]"
    assert terminal["auxiliary_fragments_mapped"] == ["[Na+:9]"]
    assert "[Na+:9]" in terminal["full_mixture"]


def test_empty_route_cannot_be_plotted_as_stock_solution():
    from mechet.analysis.tree_figure_data import route_panel_data
    with pytest.raises(ValueError, match="nontrivial.*leaves"):
        route_panel_data([{"run_id": "r", "target_id": "x", "report": {"stock_solved": True, "stock_leaves": [], "complete_reaction_edges": []}}], [{"id": "x", "cohort": "c"}], run_id="r")


def test_c1_needs_actual_exposure_and_annotation_evidence():
    assert exposure_compatible("C1", actor_exposure="FlowER_full") is False
    assert exposure_compatible("C1", actor_exposure="FlowER_full", verified_program_absent=True, minimum_primitive_frequency=5, reference_fully_annotated=True) is True
    assert exposure_compatible("invented_claim", actor_exposure="FlowER_full") is False


def test_same_basename_roots_are_distinct_runs(tmp_path):
    for parent in ["a", "b"]:
        shard = tmp_path / parent / "run/tree/rank00/evaluations"
        shard.mkdir(parents=True)
        (shard / "one.json").write_text(json.dumps({"id": "same", "tree_artifact": "absent.json"}))
    found = discover_inputs([tmp_path / "a/run", tmp_path / "b/run"])
    assert len({e["run_id"] for e in found}) == 2


def test_reported_route_must_cover_actual_nonstock_and_obligations():
    from mechet.analysis.tree_metrics import verify_reported_route
    r = {"target": "CCO", "stock_solved": True, "stock_leaves": ["C"], "complete_reaction_edges": [{"id": "r1", "product": "CCO", "precursors": ["C", "O"]}]}
    check = verify_reported_route(r, {"C"})
    assert check["stock_solved"] is False
    assert "O" in check["unresolved_precursors"]
    r["stock_leaves"] = ["C", "O"]
    check = verify_reported_route(r, {"C", "O"})
    assert check["stock_solved"] is True
    assert check["selected_edge_ids"] == ["r1"]


def test_parser_stores_independent_stock_recheck_when_supplied():
    tree = native_fixture()
    report = {"id": "case:1", "target": "CO", "stock_solved": True, "stock_leaves": ["C", "O"], "complete_reaction_edges": [{"id": "r", "product": "CO", "precursors": ["C", "O"]}]}
    data = normalize_native_tree(tree, report=report, manifest={}, source={"run_id": "r", "tree_sha256": "sha"}, stock={"C", "O"})
    assert data["routes"][0]["stock_recheck"]["stock_solved"] is True


def test_symbolic_bond_delta_is_descriptive_not_chemical_truth():
    from mechet.analysis.tree_chemistry import molecular_delta
    d = molecular_delta("[CH3:1][O:2][CH3:3]", "[CH3+:1].[O-:2][CH3:3]")
    assert d["changed_atom_maps"] == [1, 2]
    assert d["changed_bonds"] == [{"atoms": [1, 2], "before": 1.0, "after": 0.0}]
    assert d["fragment_count_delta"] == 1
    assert d["net_charge_delta"] == 0
    assert d["experimental_feasibility_established"] is False


def test_incomplete_or_nonordinary_program_has_no_whole_novelty_claim():
    from mechet.analysis.tree_chemistry import program_support
    exposure = {"primitive_counts": {}, "program_counts": {}}
    assert program_support([], exposure)["complete_signature"] is None
    nonordinary = [{"name": "apply_electron_flow", "state_before": "[CH3:1][OH:2]", "arguments": {"direction": "retrosynthetic", "bond_order_changes": [{"atoms": ["A01", "A02"], "delta": -1}]}}, {"name": "finish_trace"}]
    result = program_support(nonordinary, exposure)
    assert result["complete_signature"] is None
    assert result["ordinary_source_sink_program"] is False


def test_stock_version_must_match_the_saved_search_contract():
    from mechet.analysis.tree_records import assert_stock_identity
    with pytest.raises(ValueError, match="stock identity"):
        assert_stock_identity({"stock_canonical_sha_by_cohort": {"c": "expected"}}, "c", {"cohort_stock_sha256": {"c": "different"}})


def test_enrichment_preserves_real_visits_and_missing_planning_children():
    from scripts.enrich_nmi_tree_evidence import enrich_tree
    raw = native_fixture()
    raw["mcts_graph"] = {"s0": {"state": {"molecule": "CO", "active": "r", "pending": [], "ancestors": [], "solved": False}, "visits": 4, "value_sum": 2., "selections": 5, "edges": {"p": {"child": "s1", "n": 3, "w": 2., "count": 6}}}}
    data = normalized(raw)
    result = enrich_tree(raw, data)
    assert len(result["planning_states"]) == 1
    state = result["planning_states"][0]
    assert state["visits"] == 4
    assert state["edges"][0]["visits"] == 3
    assert state["edges"][0]["child_recorded_in_this_snapshot"] is False
    assert state["native_node_ids"] == [data["nodes"][0]["node_id"]]
    assert len(result["edge_chemistry"]) == 1  # e4 duplicate is not another successor.
    assert result["edge_chemistry"][0]["delta"]["fragment_count_delta"] == 1


def test_generated_program_support_does_not_invent_c1_labels():
    from scripts.enrich_nmi_tree_evidence import enrich_tree
    raw = native_fixture()
    result = enrich_tree(raw, normalized(raw), exposure={"model_sha256": "other", "actual_train_exposure": {"primitive_counts": {}, "program_counts": {}}})
    assert result["programs"][0]["actor_exposure_authenticated"] is False
    assert result["programs"][0]["route_C1_claim"] is False


def test_embedded_vector_panels_have_disjoint_ids_and_valid_local_references():
    # Without per-panel namespaces, clips and electron arrow markers collide.
    from scripts.render_nmi_tree_figures import embed
    import xml.etree.ElementTree as ET
    source = '<svg xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="clip"><rect width="1" height="1"/></clipPath></defs><path clip-path="url(#clip)"/><use href="#clip"/></svg>'
    first, second = [ET.fromstring(embed(source, x, 0, 100, 100)) for x in [0, 100]]
    ids = [{n.attrib["id"] for n in panel.iter() if "id" in n.attrib} for panel in [first, second]]
    assert not ids[0].intersection(ids[1])
    for panel, keys in zip([first, second], ids):
        path = next(n for n in panel.iter() if n.tag.endswith("path"))
        use = next(n for n in panel.iter() if n.tag.endswith("use"))
        assert path.attrib["clip-path"][5:-1] in keys
        assert use.attrib["href"][1:] in keys


def test_chart_svg_is_byte_stable_between_identical_renders():
    from pathlib import Path
    import yaml
    from mechet.analysis.tree_charts import chart_svg
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    def draw(ax):
        ax.plot([0, 1], [0, 1])
    assert chart_svg(style, 200, 120, draw) == chart_svg(style, 200, 120, draw)


def test_tree_highlight_is_a_real_root_to_terminal_path_not_rejected_edge():
    from pathlib import Path
    import yaml
    from scripts.render_nmi_tree_figures import tree_svg
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    svg, info = tree_svg(normalized(), style)
    assert len(info["selected_path_attempt_ids"]) == 2
    assert svg.count('data-selected-path="true"') == 2
    assert len(info["node_ids"]) == 4
    lookup = {edge["attempt_id"]: edge for edge in normalized()["attempted_edges"]}
    assert all(lookup[key]["execution_accepted"] and lookup[key]["child_node_id"] for key in info["selected_path_attempt_ids"])


def test_figure_qa_verifies_vectors_fonts_geometry_and_detects_changed_files(tmp_path):
    # Catch stale manifests, bitmap-only outputs and missing embedded fonts.
    from pathlib import Path
    import yaml
    from scripts.render_nmi_tree_figures import export, svg_document, label
    from mechet.analysis.figure_qa import inspect_figure
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    svg = svg_document(label(10, 20, "Recorded state") + '<path d="M10 30 L90 30" stroke="black"/>', style, height_pt=120)
    files = export(svg, tmp_path / "vector", style)
    report = inspect_figure(files, style)
    assert report["machine_checks_passed"] is True
    assert report["pdf"]["all_fonts_embedded"] is True
    assert report["svg"]["raster_images"] == 0
    assert report["png"]["effective_dpi"] >= 299
    assert report["visual_inspection_status"] == "not_established_by_machine_checks"
    (tmp_path / "vector.svg").write_text(svg.replace("Recorded state", "Changed"))
    assert inspect_figure(files, style)["machine_checks_passed"] is False


def test_flat_parent_records_preserve_real_links_without_rejected_child_states():
    from mechet.analysis.flat_tree_records import normalize_flat_tree
    flat = {"manifest": {"case_id": "historical_case", "scope": "single_case", "adapter_model_sha256": "actual"}, "nodes": [
        {"id": "r", "parent": None, "state": "[CH3:1][OH:2]", "status": "expanded", "depth": 0, "action": None},
        {"id": "t", "parent": "r", "state_before": "[CH3:1][OH:2]", "state": "[CH3:1][OH:2]", "status": "terminal", "depth": 1, "action": {"name": "finish_trace", "arguments": {}}},
        {"id": "bad", "parent": "r", "state_before": "[CH3:1][OH:2]", "state": "[CH3:1][OH:2]", "status": "rejected", "depth": 1, "error": "no bond", "action": {"name": "apply_electron_flow", "arguments": {}}}]}
    data = normalize_flat_tree(flat, source={"run_id": "old", "tree_sha256": "sha"})
    assert len(data["nodes"]) == 2
    assert len(data["attempted_edges"]) == 2
    rejected = next(e for e in data["attempted_edges"] if not e["execution_accepted"])
    assert rejected["child_node_id"] is None
    assert rejected["source_pointer"] == "/nodes/2/action"
    assert data["tree"]["checkpoint_sha"] == "actual"
    assert data["tree"]["source_protocol"] == "historical_single_case_not_full_benchmark"
    assert data["tree"]["has_complete_node_lineage"] is True


def test_negative_gallery_selects_a_real_same_pool_ranking_gap():
    from mechet.analysis.tree_figure_data import ranking_gap_case
    from mechet.analysis.tree_metrics import candidate_metrics
    rows = [{"target_id": "a", "report": {"ranked_candidates": []}, "summary": {"ranking": candidate_metrics([], has_reference=True)}},
        {"target_id": "b", "report": {"ranked_candidates": [
            {"structural_precursor": "O", "mean_nll": .1, "matches_reference": False},
            {"structural_precursor": "C", "mean_nll": .4, "matches_reference": True}]},
         "summary": {"ranking": candidate_metrics([{"structural_precursor": "O", "mean_nll": .1, "matches_reference": False}, {"structural_precursor": "C", "mean_nll": .4, "matches_reference": True}], has_reference=True)}}]
    selected = ranking_gap_case(rows)
    assert selected["target_id"] == "b"
    assert selected["top1"]["structural_precursor"] == "O"
    assert selected["source_reference_match"]["structural_precursor"] == "C"
    assert selected["reference_rank"] == 2
    assert selected["chemical_invalidity_claim"] is False


def test_extended_data_charts_keep_measured_counts_and_explicit_units():
    from pathlib import Path
    import yaml
    from mechet.analysis.tree_charts import morphology_chart, error_chart, quality_chart
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    rows = [{"summary": {"morphology": morphology(normalized())}}]
    import xml.etree.ElementTree as ET
    for renderer in [morphology_chart, error_chart, quality_chart]:
        root = ET.fromstring(renderer(rows, style))
        assert any(n.tag.endswith("path") for n in root.iter())
        assert not any(n.tag.endswith("image") for n in root.iter())
    text = " ".join(ET.fromstring(quality_chart(rows, style)).itertext())
    assert "different counting units" in text
    assert "5" in text and "3" in text and "1" in text


def test_single_reaction_archive_is_normalized_without_fabricating_outer_routes(tmp_path):
    from scripts.analyze_nmi_tree_evidence import analyze
    root = tmp_path / "old_greedy"
    shard = root / "greedy" / "rank00"
    (shard / "trees").mkdir(parents=True)
    (shard / "evaluations").mkdir()
    source = native_fixture()
    single = {"case_id": source["case_id"], "target": source["target"], **source["reaction_trees"]["CO"]}
    (shard / "trees/case.json").write_text(json.dumps(single))
    (shard / "manifest.json").write_text(json.dumps({"full_denominator": 1}))
    (shard / "evaluations/case.json").write_text(json.dumps({"id": "case:1", "tree_artifact": "trees/case.json"}))
    output = tmp_path / "analysis"
    result = analyze([root], output, tree_schema="single_reaction")
    inventory = json.loads((output / "input_inventory.json").read_text())
    assert inventory["status_counts"] == {"normalized": 1}
    assert result["runs"]["old_greedy:greedy"]["observed_targets"] == 1
    import gzip
    with gzip.open(output / "routes.jsonl.gz", "rt") as f:
        assert list(f) == []
    with gzip.open(output / "attempted_edges.jsonl.gz", "rt") as f:
        edges = [json.loads(line) for line in f]
    assert len(edges) == 5
    assert edges[0]["source_pointer"] == "/edges/0"


def test_delivery_catalog_rejects_stale_source_inventory(tmp_path):
    from scripts.finalize_nmi_tree_evidence import verified_analysis
    from mechet.analysis.tree_records import file_digest
    (tmp_path / "progress.json").write_text('{"phase":"complete"}')
    inventory = tmp_path / "input_inventory.json"
    inventory.write_text('{"inputs":[],"input_count":0}')
    (tmp_path / "source_manifest.json").write_text(json.dumps({"input_inventory_sha256": file_digest(inventory), "tables": {}}))
    assert verified_analysis(tmp_path)["input_count"] == 0
    inventory.write_text('{"inputs":[],"input_count":1}')
    with pytest.raises(ValueError, match="inventory hash"):
        verified_analysis(tmp_path)


def test_delivery_catalog_accounts_for_schema_fallback_without_double_counting():
    from scripts.finalize_nmi_tree_evidence import coverage_catalog
    a = {"run_id": "r", "target_id": "a", "tree_sha256": "a", "analysis_status": "normalized"}
    b = {"run_id": "s", "target_id": "b", "tree_sha256": "b", "analysis_status": "unsupported_or_invalid"}
    base = {"inputs": [a, b, dict(b)], "input_count": 3, "input_roots": ["/source"]}
    supplemental = {"inputs": [{**b, "analysis_status": "normalized"}], "input_count": 1}
    summary = coverage_catalog(base, supplemental)
    assert summary["original_input_records"] == 3
    assert summary["accounted_for_records"] == 3
    assert summary["canonical_run_target_graphs"] == 2
    assert summary["unhandled_original_records"] == 0
    assert summary["cross_run_metric_pooling"] is False


def test_chemistry_histograms_keep_missing_and_exposure_units_separate():
    from mechet.analysis.enrichment_summary import summarize_enrichment
    edges = [{"run_id": "r", "delta_status": "available", "delta": {"changed_atom_maps": [1, 2], "changed_bonds": [{"atoms": [1, 2]}], "cycle_rank_delta": 0, "net_charge_delta": 0}, "source_sink_types": ["BOND→ATOM", "LP→BOND"]},
        {"run_id": "r", "delta_status": "not_establishable", "delta": None},
        {"run_id": "other", "delta_status": "not_establishable", "delta": None}]
    programs = [{"run_id": "r", "actor_exposure_authenticated": True, "complete_signature": "sig", "complete_signature_training_count": 0, "minimum_primitive_training_frequency": 9},
        {"run_id": "r", "actor_exposure_authenticated": False, "complete_signature": "sig", "complete_signature_training_count": 0, "minimum_primitive_training_frequency": 9}]
    planning = [{"run_id": "r", "visits": 7}, {"run_id": "r", "visits": None}]
    result = summarize_enrichment(edges, programs, planning, run_id="r")
    assert result["flow_delta_status"] == {"available": 1, "not_establishable": 1}
    assert result["changed_atoms"] == {"2": 1}
    assert result["program_classes"] == {"unseen_ordered_signature": 1, "unavailable_or_unauthenticated": 1}
    assert result["planning_visits"] == {"7": 1}
    assert result["planning_visits_missing"] == 1
    assert result["route_C1_claim"] is False


def test_chemistry_extended_panels_do_not_turn_missing_visits_into_zeros():
    from pathlib import Path
    import yaml
    import xml.etree.ElementTree as ET
    from scripts.render_nmi_chemistry_extended import render_summary
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    summary = {"changed_atoms": {"2": 1}, "changed_bonds": {"1": 1}, "source_sink_types": {"BOND→ATOM": 1},
        "program_classes": {"unavailable_or_unauthenticated": 1}, "planning_visits": {}, "planning_visits_missing": 1}
    panels = render_summary(summary, style)
    assert set(panels) == {"extended_data_local_chemistry", "extended_data_program_support", "extended_data_mcts_visits"}
    assert "not recorded" in " ".join(ET.fromstring(panels["extended_data_mcts_visits"]).itertext())


def test_full_program_cells_retain_interleaved_imports_and_counts():
    from mechet.analysis.tree_figure_data import program_cells
    reaction = {"id": "r", "program": {"structural_terminal_mapped": "[CH3:1].[OH:2]", "actions": [
        {"name": "import_fragments", "state_before": "[CH3:1][OH:2]", "arguments": {"fragments": [{"smiles": "O", "count": 3, "purpose": "auxiliary"}]}},
        {"name": "apply_electron_flow", "state_before": "[CH3:1][OH:2].O.O.O", "arguments": {"electron_flow": []}},
        {"name": "finish_trace", "arguments": {}}]}}
    cells = program_cells(reaction)
    assert [c["kind"] for c in cells] == ["import_fragments", "apply_electron_flow", "finish_trace"]
    assert cells[0]["fragments"] == [{"smiles": "O", "count": 3, "purpose": "auxiliary"}]
    assert cells[1]["state"] == "[CH3:1][OH:2].O.O.O"
    assert cells[2]["state"] == "[CH3:1].[OH:2]"


def test_action_gallery_uses_the_target_reaction_not_a_shorter_downstream_step():
    from mechet.analysis.tree_figure_data import target_reaction_program
    report = {"target": "CCO", "complete_reaction_edges": [
        {"id": "shorter", "product": "CO", "program": {"actions": [1]}},
        {"id": "actual_target", "product": "CCO", "program": {"actions": list(range(13))}}]}
    assert target_reaction_program(report)["id"] == "actual_target"


def test_microscope_does_not_present_symmetric_context_choices_as_distinct_chemistry():
    from scripts.render_nmi_tree_figures import select_flow_fork
    raw = native_fixture()
    inner = raw["reaction_trees"]["CO"]
    sibling = copy.deepcopy(inner["contexts"][2])
    sibling["id"] = "b2"
    inner["contexts"].append(sibling)
    other = copy.deepcopy(inner["edges"][1])
    other.update(id="e5", child_context="b2")
    inner["edges"].append(other)
    assert select_flow_fork(normalized(raw)) is None
    sibling["chemical_id"] = "charged"
    sibling["state"] = "[CH3+:1].[OH-:2]"
    inner["chemical_nodes"].append({"id": "charged", "state_unmapped": "[CH3+].[OH-]"})
    fork = select_flow_fork(normalized(raw))
    assert len({child["state_unmapped"] for child in fork["children"]}) == 2
    import yaml
    from pathlib import Path
    from scripts.render_nmi_tree_figures import tree_svg
    style = yaml.safe_load((Path(__file__).parents[1] / "configs/figures/mechet_nmi_tree_style_v1.yaml").read_text())
    _, info = tree_svg(normalized(raw), style)
    assert info["microscope_parent_node_id"] == fork["parent"]["node_id"]
