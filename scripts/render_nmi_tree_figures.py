#!/usr/bin/env python3
"""Vector figures from source-bound existing-tree analysis, never new chemistry.

All atom addresses pass through the original current-state compiler. Display
coordinates never substitute for chemistry, and rejected proposals have no node.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import gzip
import hashlib
import html
import io
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.analysis.tree_records import file_digest, canonical_entries

NS = "http://www.w3.org/2000/svg"


def molecule_svg(state, moves, style, *, uid, width, height):
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(state, params)
    if mol is None:
        raise ValueError("source molecular state cannot be rendered")
    rdDepictor.Compute2DCoords(mol)
    maps = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    active, bonds = set(), set()
    arrows = []
    for move in moves:
        if move.get("mode") == "BE_DELTA":
            raise ValueError("BE_DELTA is not an ordinary source-sink arrow; render bond changes separately")
        for end in [move["source"], move["sink"]]:
            active.update(end["atoms"])
            if end["kind"] == "BOND":
                bonds.add(tuple(end["atoms"]))
        arrows.append(move)
    if active - maps.keys():
        raise ValueError("recorded flow refers to absent atom maps")
    for move in arrows:
        source = move["source"]
        if source["kind"] == "BOND" and mol.GetBondBetweenAtoms(*(maps[x] for x in source["atoms"])) is None:
            raise ValueError("recorded electron source bond is absent")
    display = Chem.Mol(mol)
    for atom in display.GetAtoms():
        atom.SetAtomMapNum(0)
    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = .10
    options.bondLineWidth = style["molecules"]["standard_bond_line_width_px"]
    # RDKit's C++ interface requires integer font-size bounds, unlike the YAML
    # and Matplotlib point-valued typography configuration.
    options.minFontSize = math.ceil(style["fonts"]["figure_text_min_pt"])
    options.maxFontSize = math.floor(style["fonts"]["figure_text_max_pt"])
    color = style["palette"]["flow_blue"].lstrip("#")
    tint = tuple(.82 + .18 * int(color[i:i+2], 16) / 255 for i in (0, 2, 4))
    highlight_bonds = []
    for pair in bonds:
        bond = mol.GetBondBetweenAtoms(*(maps[x] for x in pair))
        if bond is not None:
            highlight_bonds.append(bond.GetIdx())
    drawer.DrawMolecule(display, highlightAtoms=[maps[x] for x in active], highlightBonds=highlight_bonds,
        highlightAtomColors={maps[x]: tint for x in active}, highlightBondColors={x: tint for x in highlight_bonds})
    coords = {n: (drawer.GetDrawCoords(idx).x, drawer.GetDrawCoords(idx).y) for n, idx in maps.items()}
    drawer.FinishDrawing()
    drawing = ET.fromstring(drawer.GetDrawingText())
    contents = "".join(ET.tostring(child, encoding="unicode") for child in drawing)
    ink = style["palette"][style["molecules"]["electron_arrow_color"]]
    contents += f'<defs><marker id="arrow-{uid}" markerWidth="5" markerHeight="5" refX="4.4" refY="2.5" orient="auto"><path d="M0 0 L5 2.5 L0 5" fill="none" stroke="{ink}" stroke-width="0.8"/></marker></defs>'
    def point(end):
        xy = [coords[a] for a in end["atoms"]]
        x = sum(a[0] for a in xy) / len(xy)
        y = sum(a[1] for a in xy) / len(xy)
        return (x + 3, y - 4) if end["kind"] == "LP" else (x, y)
    geometry = []
    for index, move in enumerate(arrows):
        sx, sy = point(move["source"])
        ex, ey = point(move["sink"])
        end_anchor_atom = None
        sink_bond_present = None
        if move["sink"]["kind"] == "BOND":
            pair = move["sink"]["atoms"]
            sink_bond_present = mol.GetBondBetweenAtoms(*(maps[x] for x in pair)) is not None
            if not sink_bond_present:
                other = set(pair) - set(move["source"]["atoms"])
                if len(other) == 1:
                    end_anchor_atom = next(iter(other))
                    ex, ey = coords[end_anchor_atom]
        geometry.append({"end_anchor_atom": end_anchor_atom, "sink_bond_present": sink_bond_present,
            "start_xy": [sx, sy], "end_xy": [ex, ey]})
        dx, dy = ex - sx, ey - sy
        length = max(1., math.hypot(dx, dy))
        bend = min(16., max(7., length * .25))
        cx, cy = (sx + ex) / 2 - dy / length * bend, (sy + ey) / 2 + dx / length * bend
        source_maps = ",".join(map(str, move["source"]["atoms"]))
        sink_maps = ",".join(map(str, move["sink"]["atoms"]))
        contents += f'<path data-electron-flow="{index}" data-source-maps="{source_maps}" data-sink-maps="{sink_maps}" d="M{sx:.3f} {sy:.3f} Q{cx:.3f} {cy:.3f} {ex:.3f} {ey:.3f}" stroke="{ink}" stroke-width="{style["molecules"]["electron_arrow_line_pt"]}" fill="none" marker-end="url(#arrow-{uid})"/>'
    svg = f'<svg xmlns="{NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-backend="RDKit">{contents}</svg>'
    info = {"backend": "RDKit_MolDraw2DSVG", "rdkit_version": rdBase.rdkitVersion,
        "state_sha256": hashlib.sha256(state.encode()).hexdigest(), "source_maps": [a["source"]["atoms"] for a in arrows],
        "sink_maps": [a["sink"]["atoms"] for a in arrows], "electron_arrow_count": len(arrows),
        "changed_atom_maps": sorted(active), "element_colors": "RDKit_default", "stereochemistry_preserved": True,
        "arrow_geometry": geometry}
    return svg, info


def layered_layout(nodes, edges):
    """Stable depth layout of all native contexts, preserving transpositions."""
    layers = defaultdict(list)
    for n in nodes:
        layers[n.get("depth") or 0].append(n["node_id"])
    parents = defaultdict(set)
    for e in edges:
        if e.get("parent_node_id") and e.get("child_node_id"):
            parents[e["child_node_id"]].add(e["parent_node_id"])
    positions = {}
    for depth, keys in sorted(layers.items()):
        def key(node):
            prior = [positions[p][1] for p in sorted(parents[node]) if p in positions]
            return (sum(prior) / len(prior) if prior else .5, node)
        for i, node in enumerate(sorted(keys, key=key)):
            positions[node] = (float(depth), (i + .5) / len(keys))
    return positions


def select_flow_fork(data):
    nodes = {n["node_id"]: n for n in data["nodes"]}
    choices = defaultdict(dict)
    for e in data["attempted_edges"]:
        if e["action_family"] == "apply_electron_flow" and e["execution_accepted"] and e["child_node_id"] and e["parent_node_id"]:
            # Display chemistry-distinct alternatives, not merely private-map
            # or history-distinct contexts. The full atlas still retains both.
            projection = nodes[e["child_node_id"]]["state_unmapped"]
            if projection is not None:
                choices[e["parent_node_id"]].setdefault(projection, e)
    eligible = [p for p, v in choices.items() if len(v) > 1]
    if not eligible:
        return None
    parent = min(eligible, key=lambda p: (nodes[p].get("depth") or 0, p))
    edges = sorted(choices[parent].values(), key=lambda e: e["attempt_id"])[:2]
    return {"parent": nodes[parent], "edges": edges, "children": [nodes[e["child_node_id"]] for e in edges]}


def svg_document(body, style, *, height_pt):
    width = style["format"]["full_width_mm"] * 72 / 25.4
    height_mm = height_pt * 25.4 / 72
    family = ",".join([style["fonts"]["family"], *style["fonts"]["fallbacks"]])
    return f'<svg xmlns="{NS}" width="{style["format"]["full_width_mm"]}mm" height="{height_mm}mm" viewBox="0 0 {width} {height_pt}"><rect width="100%" height="100%" fill="{style["format"]["background"]}"/><style>text{{font-family:{family};font-size:{style["fonts"]["figure_text_pt"]}px;fill:{style["palette"]["ink"]}}}</style>{body}</svg>'


def label(x, y, value, *, size=None, weight=None):
    settings = (f'font-size:{size}px;' if size else "") + (f'font-weight:{weight};' if weight else "")
    props = f'style="{settings}"' if settings else ""
    return f'<text x="{x}" y="{y}" {props}>{html.escape(str(value))}</text>'


def embed(svg, x, y, width, height):
    root = ET.fromstring(svg)
    # Matplotlib/RDKit use local marker and clip IDs. A nested SVG shares the
    # outer document's namespace, so prefix IDs and references deterministically.
    prefix = "panel-" + hashlib.sha256((svg + repr((x, y, width, height))).encode()).hexdigest()[:16] + "-"
    replacements = {node.attrib["id"]: prefix + node.attrib["id"] for node in root.iter() if "id" in node.attrib}
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if key == "id":
                node.set(key, replacements[value])
                continue
            if key.endswith("href") and value.startswith("#") and value[1:] in replacements:
                node.set(key, "#" + replacements[value[1:]])
            else:
                for old, new in replacements.items():
                    value = value.replace("url(#" + old + ")", "url(#" + new + ")")
                node.set(key, value)
    root.set("x", str(x)); root.set("y", str(y))
    root.set("width", str(width)); root.set("height", str(height))
    return ET.tostring(root, encoding="unicode")


def tree_svg(data, style, *, width=500, height=260, uid="atlas"):
    positions = layered_layout(data["nodes"], data["attempted_edges"])
    # An illustrated path is a recorded accepted chain to a true terminal,
    # not necessarily the report's chosen all-leaf route or a physical mechanism.
    adjacency = defaultdict(list)
    children = set()
    for edge in data["attempted_edges"]:
        if edge["execution_accepted"] and edge["parent_node_id"] in positions and edge["child_node_id"] in positions:
            adjacency[edge["parent_node_id"]].append(edge)
            children.add(edge["child_node_id"])
    terminal_nodes = {n["node_id"] for n in data["nodes"] if n["terminal"]}
    roots = sorted(set(positions) - children)
    frontier = deque((root, []) for root in roots)
    seen, selected = set(), []
    while frontier:
        node_id, path = frontier.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id in terminal_nodes and path:
            selected = path
            break
        for edge in sorted(adjacency[node_id], key=lambda e: e["attempt_id"]):
            frontier.append((edge["child_node_id"], path + [edge]))
    selected_pairs = {(e["parent_node_id"], e["child_node_id"]) for e in selected}
    max_depth = max((xy[0] for xy in positions.values()), default=1) or 1
    def xy(key):
        depth, y = positions[key]
        return 8 + depth / max_depth * (width - 16), 12 + y * (height - 24)
    body = []
    families = {"import_fragments": "action_import", "apply_electron_flow": "action_flow", "finish_trace": "action_finish"}
    distinct = set()
    rejected = defaultdict(int)
    for e in data["attempted_edges"]:
        p, c = e["parent_node_id"], e["child_node_id"]
        if not p or p not in positions:
            continue
        sx, sy = xy(p)
        color = style["palette"][style["semantic"][families.get(e["action_family"], "action_executor_rejected")]]
        if c and c in positions and (p, c) not in distinct:
            distinct.add((p, c)); ex, ey = xy(c)
            chosen = (p, c) in selected_pairs
            line_width = style["tree"]["selected_path_line_pt"] if chosen else style["tree"]["context_edge_line_pt"]
            opacity = 1 if chosen else style["tree"]["non_selected_alpha"]
            body.append(f'<path data-selected-path="{str(chosen).lower()}" d="M{sx} {sy} L{ex} {ey}" stroke="{color}" stroke-width="{line_width}" stroke-opacity="{opacity}" fill="none"/>')
        elif not c and not e["execution_accepted"]:
            rejected[p] += 1
    for p, count in sorted(rejected.items()):
        x, y = xy(p)
        body.append(f'<path data-rejected-attempt-count="{count}" d="M{x} {y} l3 -3" fill="none" stroke="{style["palette"]["rejected_clay"]}" stroke-dasharray="1 1" stroke-width=".4"/>')
    for n in sorted(data["nodes"], key=lambda n: n["node_id"]):
        x, y = xy(n["node_id"])
        color = style["palette"]["terminal_teal_dark" if n["terminal"] else "neutral_mid"]
        radius = 1.5 if n["terminal"] else .75
        body.append(f'<circle data-node-id="{n["node_id"]}" cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>')
    fork = select_flow_fork(data)
    microscope_parent = fork["parent"]["node_id"] if fork else None
    if microscope_parent:
        x, y = xy(microscope_parent)
        body.append(f'<circle data-microscope-parent="{microscope_parent}" cx="{x}" cy="{y}" r="3.2" fill="none" stroke="{style["palette"]["search_blue_dark"]}" stroke-width=".7"/>')
        body.append(label(x + 5, max(7, y - 4), "Zoom"))
    body.append(label(8, height - 1, f'{len(data["nodes"]):,} contexts · {len(distinct):,} transitions · action depth {int(max_depth)}'))
    return f'<svg xmlns="{NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(body)}</svg>', {"node_ids": sorted(positions), "context_transitions": len(distinct), "max_action_depth": int(max_depth), "rejected_stubs_grouped_by_parent": dict(rejected), "node_size_encoding": "constant by terminal status, not synthetic visits", "selected_path_attempt_ids": [e["attempt_id"] for e in selected], "selected_path_rule": "shortest recorded accepted root-to-terminal chain, lexical tie break; not chosen AND route", "microscope_parent_node_id": microscope_parent}


def export(svg, destination, style):
    import cairosvg
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.fromstring(svg)
    destination.with_suffix(".svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(destination.with_suffix(".pdf")))
    width_px = round(style["format"]["full_width_mm"] / 25.4 * style["format"]["preview_dpi"])
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(destination.with_suffix(".png")), output_width=width_px)
    return {ext: {"path": str(destination.with_suffix("." + ext)), "sha256": file_digest(destination.with_suffix("." + ext))} for ext in ("svg", "pdf", "png")}


def render_microscope(data, style):
    from mechet.natural_language_electron_flow import compile_event_arguments
    fork = select_flow_fork(data)
    if fork is None:
        raise ValueError("no real distinct FLOW fork available")
    body = label(6, 12, "a", size=8, weight="bold") + label(18, 12, "Two recorded electron-flow choices from the same state")
    metadata = []
    for i, edge in enumerate(fork["edges"]):
        moves = compile_event_arguments(edge["before_state"], edge["tool_payload"]["arguments"])
        drawing, info = molecule_svg(edge["before_state"], moves, style, uid=f"fork{i}", width=250, height=175)
        after, after_info = molecule_svg(edge["after_state"], [], style, uid=f"child{i}", width=250, height=175)
        x = i * 259
        body += label(x + 6, 30, f'Choice {i + 1} · {len(moves)} recorded electron-pair moves')
        body += embed(drawing, x + 2, 35, 250, 175)
        body += label(x + 6, 228, "Accepted successor") + embed(after, x + 2, 235, 250, 175)
        metadata.append({"attempt_id": edge["attempt_id"], "source_pointer": edge["source_pointer"], "parent_node_id": edge["parent_node_id"], "child_node_id": edge["child_node_id"], "before": info, "after": after_info})
    body += label(6, 433, style["molecules"]["symbolic_state_caption"])
    body += label(6, 445, "Arrows are recorded symbolic operations; acceptance does not establish an experimental mechanism.")
    return svg_document(body, style, height_pt=453), metadata


def render_trajectory(report, style, *, normalized, width=508, height=250):
    """A complete observed reaction program; not an invented physical pathway."""
    from mechet.natural_language_electron_flow import compile_event_arguments
    reactions = report.get("complete_reaction_edges", [])
    eligible = [e for e in reactions if 1 <= sum(a["name"] == "apply_electron_flow" for a in e["program"]["actions"]) <= 3]
    if not eligible:
        raise ValueError("no complete short recorded reaction program fits the trajectory panel")
    reaction = eligible[0]
    actions = reaction["program"]["actions"]
    cells = []
    for index, action in enumerate(actions):
        if action["name"] == "apply_electron_flow":
            moves = compile_event_arguments(action["state_before"], action["arguments"])
            cells.append((f"FLOW {len(cells)+1}", action["state_before"], moves, index))
    cells.append(("Executed structural precursors", reaction["program"]["structural_terminal_mapped"], [], len(actions)-1))
    # Always two columns; four large molecular panels, not squeezed thumbnails.
    body = ""
    metadata = []
    row_height = height / 2
    for i, (title, state, moves, index) in enumerate(cells):
        x, y = (i % 2) * width / 2, (i // 2) * row_height
        drawing, info = molecule_svg(state, moves, style, uid=f"trajectory{i}", width=width / 2 - 8, height=row_height - 20)
        body += label(x + 5, y + 10, title) + embed(drawing, x + 2, y + 15, width / 2 - 8, row_height - 20)
        metadata.append({"reaction_id": reaction["id"], "action_index": index,
            "molecule": info, "kind": "flow_before_state" if moves else "structural_endpoint"})
        if moves:
            action = actions[index]
            matching = [edge for edge in normalized["attempted_edges"]
                if edge["execution_accepted"] and edge["before_state"] == state
                and edge["tool_payload"] == {"name": action["name"], "arguments": action["arguments"]}]
            if not matching:
                raise ValueError("illustrated trajectory action has no source attempted edge")
            metadata[-1].update({"attempt_id": matching[0]["attempt_id"], "parent_node_id": matching[0]["parent_node_id"],
                "child_node_id": matching[0]["child_node_id"], "source_pointer": matching[0]["source_pointer"],
                "equivalent_recorded_attempt_ids": [e["attempt_id"] for e in matching]})
        else:
            matching = [t for t in normalized["terminals"] if t["native_context_id"] == reaction["program"]["terminal_context"] and t["reaction_product"] == reaction["product"]]
            if len(matching) != 1:
                raise ValueError("illustrated endpoint has no unique recorded terminal")
            metadata[-1]["terminal_id"] = matching[0]["terminal_id"]
    return f'<svg xmlns="{NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{body}</svg>', metadata


def render_ranking_ambiguity(entries, style):
    from mechet.analysis.tree_figure_data import ranking_gap_case
    selection = ranking_gap_case(entries)
    if selection is None:
        raise ValueError("no recorded same-pool ranking-gap case")
    row = next(e for e in entries if e["target_id"] == selection["target_id"])
    states = [("Target product", row["report"]["target"]),
        (f'NLL rank 1 · mean NLL {selection["top1"]["mean_nll"]:.4f}', selection["top1"]["structural_precursor"]),
        (f'Source-reference match at rank {selection["reference_rank"]}', selection["source_reference_match"]["structural_precursor"])]
    body = label(6, 12, "Candidate ranking ambiguity · " + selection["target_id"])
    molecules = []
    for i, (title, state) in enumerate(states):
        drawing, info = molecule_svg(state, [], style, uid=f"ambiguity{i}", width=500, height=85)
        y = 26 + i * 105
        body += label(6, y, title) + embed(drawing, 6, y + 5, 500, 85)
        molecules.append(info)
    body += label(6, 350, "Reference mismatch is not a chemical-invalidity label; independent corroboration remains unresolved.")
    selection.update(tree_id=row["tree_id"], tree_sha256=row["tree_sha256"], report_path=row["report_path"],
        report_sha256=row.get("report_sha256"), molecule_sources=molecules,
        candidate_contexts=[selection[k].get("context") for k in ["top1", "source_reference_match"]])
    return svg_document(body, style, height_pt=365), selection


def render_full_action_program(report, style, *, normalized):
    from mechet.analysis.tree_figure_data import program_cells, target_reaction_program
    from mechet.natural_language_electron_flow import compile_event_arguments
    reaction = target_reaction_program(report)
    cells = program_cells(reaction)
    documents, metadata = [], []
    page_count = math.ceil(len(cells) / 8)
    for offset in range(0, len(cells), 8):
        chunk = cells[offset:offset + 8]
        rows = math.ceil(len(chunk) / 2)
        height = 32 + rows * 108 + 15
        page = offset // 8 + 1
        body = label(6, 12, f"Complete target-reaction action order · part {page}/{page_count}")
        for local, cell in enumerate(chunk):
            i = offset + local
            x, y = (local % 2) * 259, 24 + (local // 2) * 108
            if cell["kind"] == "import_fragments":
                fragments = cell["fragments"]
                # Each unique imported structure is drawn once with its actual
                # count. Pagination preserves every action, including imports.
                body += label(x + 5, y + 7, f'{i+1}. IMPORT · ' + ", ".join(f'×{f.get("count", 1)} {f.get("purpose", "unspecified")}' for f in fragments))
                state = ".".join(f["smiles"] for f in fragments)
                moves = []
            elif cell["kind"] == "apply_electron_flow":
                state = cell["state"]
                moves = compile_event_arguments(state, cell["arguments"])
                body += label(x + 5, y + 7, f'{i+1}. FLOW · {len(moves)} electron-pair moves')
            else:
                state, moves = cell["state"], []
                body += label(x + 5, y + 7, f'{i+1}. FINISH · structural precursors')
            drawing, info = molecule_svg(state, moves, style, uid=f'full-action-{i}', width=249, height=92)
            body += embed(drawing, x + 3, y + 12, 249, 92)
            action = reaction["program"]["actions"][cell["action_index"]]
            matches = [e for e in normalized["attempted_edges"] if e["execution_accepted"]
                and e["reaction_product"] == reaction["product"]
                and e["tool_payload"] == {"name": action["name"], "arguments": action["arguments"]}
                and (not action.get("state_before") or action["state_before"] == e["before_state"])]
            if not matches:
                raise ValueError("full action gallery lost its original attempted-edge binding")
            metadata.append({**cell, "page": page, "molecule": info, "attempt_ids": [e["attempt_id"] for e in matches],
                "source_pointers": [e["source_pointer"] for e in matches]})
        body += label(6, height - 5, "IMPORT: added species only · FLOW: exact pre-action mixture · FINISH: structural endpoint · symbolic operations, not observed intermediates.")
        documents.append(svg_document(body, style, height_pt=height))
    return documents, metadata


def render_observed(analysis_root, style_path, *, case_ids, run_id, assets_manifest=None, historical_report=None, endpoint_run=None):
    import yaml
    from mechet.analysis.tree_charts import route_chart, empirical_chart, depth_chart, growth_chart, ranking_chart, morphology_chart, error_chart, quality_chart
    from mechet.analysis.tree_figure_data import route_panel_data
    analysis_root = Path(analysis_root)
    style = yaml.safe_load(Path(style_path).read_text())
    inventory_path = analysis_root / "input_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    entries = inventory["inputs"]
    route_entries = canonical_entries(entries, run_id=run_id)
    if not route_entries:
        raise ValueError("no normalized primary route run")
    selected = []
    out = analysis_root / "figures"
    figures = {}
    case_meta = []
    for case_index, case_id in enumerate(case_ids):
        matches = [e for e in route_entries if e["target_id"] == case_id]
        if len(matches) != 1:
            raise ValueError("select exactly one recorded source for each case")
        row = matches[0]
        if file_digest(row["tree_path"]) != row["tree_sha256"]:
            raise ValueError("original tree changed since normalization")
        with gzip.open(row["cache_path"], "rt") as f:
            data = json.load(f)
        selected.append((row, data))
        tree, tree_info = tree_svg(data, style)
        prefix = f"case_{case_index+1}"
        atlas_doc = svg_document(label(6, 12, case_id) + embed(tree, 5, 20, 508, 290)
            + label(6, 315, "Dot: symbolic state · line: action · teal dot: terminal · bold: one recorded terminal path")
            + label(6, 326, "Ochre: import · blue: FLOW · teal: finish · dashed stub: rejection (no child state)"), style, height_pt=338)
        figures[prefix + "_atlas"] = export(atlas_doc, out / (prefix + "_atlas"), style)
        microscope, fork = render_microscope(data, style)
        figures[prefix + "_microscope"] = export(microscope, out / (prefix + "_microscope"), style)
        action_docs, action_meta = render_full_action_program(row["report"], style, normalized=data)
        for page, action_doc in enumerate(action_docs):
            key = prefix + "_action_program" + (f"_page_{page+1}" if page else "")
            figures[key] = export(action_doc, out / key, style)
        if case_index == 0:
            figures["tree_atlas"] = export(atlas_doc, out / "tree_atlas", style)
            figures["node_microscope"] = export(microscope, out / "node_microscope", style)
        case_meta.append({"case_id": case_id, "run_id": run_id, "tree_sha256": row["tree_sha256"],
            "tree_path": row["tree_path"], "tree": tree_info, "chemical_fork": fork, "full_action_program": action_meta,
            "case_selection": "fixed genuine case IDs; not a matched-baseline rescue claim"})
    row, data = selected[0]
    tree, _ = tree_svg(data, style, width=310, height=130)
    morph = empirical_chart(route_entries, style, width=190, height=130)
    fork = select_flow_fork(data)
    from mechet.natural_language_electron_flow import compile_event_arguments
    microscope_body = ""
    for i, edge in enumerate(fork["edges"]):
        moves = compile_event_arguments(edge["before_state"], edge["tool_payload"]["arguments"])
        drawing, _ = molecule_svg(edge["before_state"], moves, style, uid=f"fig2fork{i}", width=247, height=140)
        microscope_body += label(i * 255 + 5, 9, f"Recorded choice {i+1}") + embed(drawing, i * 255 + 3, 15, 247, 140)
    # Main Figure2 uses the whole frozen denominator but explicitly lower-bound
    # observations, NOT an interpolated final curve or equal-compute comparison.
    panel_sources = {}
    if assets_manifest:
        from mechet.analysis.tree_records import load_frozen_stocks, assert_stock_identity
        from mechet.analysis.tree_metrics import verify_reported_route
        assets_path = Path(assets_manifest)
        assets = json.loads(assets_path.read_text())
        stocks, stock_sources = load_frozen_stocks(assets_path)
        rechecks = {}
        for entry in route_entries:
            assert_stock_identity(entry["manifest"], entry["report"].get("cohort"), stock_sources)
            check = verify_reported_route(entry["report"], stocks.get(entry["report"].get("cohort")))
            if entry["report"].get("stock_solved") and check["stock_solved"] is not True:
                raise ValueError("cannot plot a reported route that fails frozen-stock AND recheck")
            rechecks[entry["target_id"]] = check
        targets_path = Path(assets["full_targets_path"])
        if file_digest(targets_path) != assets["full_targets_sha256"]:
            raise ValueError("frozen route targets hash mismatch")
        targets = [json.loads(line) for line in targets_path.read_text().splitlines() if line]
        panel = route_panel_data(entries, targets, run_id=run_id)
        upper = route_chart(panel, style)
        panel_sources["fig2a"] = {"data": panel, "assets_path": str(assets_path), "assets_sha256": file_digest(assets_path),
            "stock_sources": stock_sources, "AND_route_rechecks": rechecks}
    else:
        upper = empirical_chart(route_entries, style, width=510, height=125)
        panel_sources["fig2a"] = {"availability": "observed morphology only; route asset manifest not provided"}
    body = label(6, 12, "a", size=8, weight="bold") + label(18, 12, "Existing search output · MCTS remains incomplete")
    body += embed(upper, 5, 17, 510, 125)
    body += label(6, 158, "b", size=8, weight="bold") + label(18, 158, "Recorded electronic-state forest")
    body += embed(tree, 5, 166, 310, 130)
    body += label(325, 158, "c", size=8, weight="bold") + label(337, 158, f"Observed targets (n = {len(route_entries)})")
    body += embed(morph, 323, 166, 190, 130)
    body += label(6, 314, "d", size=8, weight="bold") + label(18, 314, "Two electronic choices at a shared state")
    body += f'<g transform="translate(5,323)">{microscope_body}</g>'
    figures["fig2_r1_observed_tree_analysis"] = export(svg_document(body, style, height_pt=480), out / "fig2_r1_observed_tree_analysis", style)
    panel_sources["fig2bc"] = {"case": case_meta[0], "target_ids": [e["target_id"] for e in route_entries], "tree_ids": [e["tree_id"] for e in route_entries]}
    panel_sources["fig2d"] = case_meta[0]["chemical_fork"]
    # Figure3 retains authentic historical accumulation and current route depth;
    # these are explicitly different populations and training exposures.
    body = label(6, 12, "a", size=8, weight="bold") + label(18, 12, "RetroCast · reference-route complexity")
    if assets_manifest:
        retro = [t for t in targets if t["cohort"] == "retrocast_mkt_cnv160"]
        retro_ids = {t["id"] for t in retro}
        retro_panel = route_panel_data([e for e in entries if e["target_id"] in retro_ids], retro, run_id=run_id)
        body += embed(depth_chart(retro_panel, style), 0, 20, 260, 155)
        panel_sources["fig3a"] = {"data": retro_panel, "composition_novelty_claim": False, "classification": "reference_depth_not_C1"}
    else:
        body += embed(empirical_chart(route_entries, style, width=260, height=155), 0, 20, 260, 155)
        panel_sources["fig3a"] = {"availability": "reference depth unavailable; observed morphology substituted with explicit label"}
    body += label(272, 12, "b", size=8, weight="bold") + label(284, 12, "Historical source-pool signatures")
    if historical_report:
        history_path = Path(historical_report)
        history = json.loads(history_path.read_text())
        if history.get("rows") != 257167:
            raise ValueError("historical accumulation row-count mismatch")
        body += embed(growth_chart(history, style), 272, 20, 240, 155)
        panel_sources["fig3b"] = {"path": str(history_path), "sha256": file_digest(history_path),
            "points": history["vocabulary_growth"], "current_actor_exposure": False,
            "source_hash_in_report": history.get("source_sha256"), "underlying_large_training_source_rehashed_in_this_render": False}
    else:
        # Do not produce a fake accumulation panel; use a real second observed
        # morphology view and name it accordingly.
        raise ValueError("Figure3 requires the real historical accumulation report")
    trajectory, trajectory_info = render_trajectory(row["report"], style, normalized=data)
    body += label(6, 190, "c", size=8, weight="bold") + label(18, 190, "One complete generated inverse-electron program")
    body += embed(trajectory, 5, 200, 508, 250)
    body += label(6, 467, "Symbolic electron movement; imported species are included in each recorded state.")
    figures["fig3_r2_observed_chemistry"] = export(svg_document(body, style, height_pt=480), out / "fig3_r2_observed_chemistry", style)
    panel_sources["fig3c"] = {"case_id": row["target_id"], "tree_sha256": row["tree_sha256"], "program": trajectory_info,
        "claim": "executed complex program, not H2 or C1 generalization"}
    source_ids = {"run_id": run_id, "target_ids": [e["target_id"] for e in route_entries], "tree_ids": [e["tree_id"] for e in route_entries]}
    for name, title, chart in [("extended_data_morphology", "Observed electronic-search morphology", morphology_chart),
            ("extended_data_executor_errors", "Recorded executor rejection taxonomy", error_chart),
            ("extended_data_execution_funnel", "Execution evidence at distinct counting units", quality_chart)]:
        doc = svg_document(label(6, 12, title) + embed(chart(route_entries, style), 5, 20, 510, 200), style, height_pt=235)
        figures[name] = export(doc, out / name, style)
        panel_sources[name] = source_ids
    if endpoint_run:
        endpoint_entries = canonical_entries(entries, run_id=endpoint_run)
        ranking = ranking_chart(endpoint_entries, style)
        doc = svg_document(label(6, 12, "Recorded candidate generation versus NLL selection") + embed(ranking, 5, 20, 510, 180), style, height_pt=220)
        figures["extended_data_endpoint_ranking"] = export(doc, out / "extended_data_endpoint_ranking", style)
        panel_sources["extended_data_endpoint_ranking"] = {"run_id": endpoint_run, "target_ids": [e["target_id"] for e in endpoint_entries],
            "source_protocol": "frozen_rxn_prod_min_proxy_DIAGNOSTIC", "independent_pass_at_k": False}
        ambiguous, info = render_ranking_ambiguity(endpoint_entries, style)
        figures["extended_data_chemical_ambiguity"] = export(ambiguous, out / "extended_data_chemical_ambiguity", style)
        panel_sources["extended_data_chemical_ambiguity"] = info
    manifest = {"schema": "nmi_observed_figure_manifest_v1", "style_sha256": file_digest(style_path),
        "producer_files": {str(p.relative_to(ROOT)): file_digest(p) for p in [Path(__file__), ROOT / "src/mechet/analysis/tree_charts.py", ROOT / "src/mechet/analysis/tree_figure_data.py", ROOT / "src/mechet/analysis/figure_qa.py", ROOT / "src/mechet/natural_language_electron_flow.py"]},
        "input_inventory_sha256": file_digest(inventory_path), "figures": figures, "panels": panel_sources,
        "cases": case_meta, "new_model_sampling": False, "paper_main_modified": False,
        "formal_final_benchmarks_complete": False}
    (analysis_root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from mechet.analysis.figure_qa import inspect_figure
    qa = {"schema": "nmi_figure_qa_v1", "figure_manifest_sha256": file_digest(analysis_root / "figure_manifest.json"),
        "figures": {key: inspect_figure(files, style) for key, files in figures.items()}}
    (analysis_root / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    selected_keys = {e["tree_id"] for e, _ in selected}
    with (analysis_root / "case_registry.csv").open("w", newline="") as f:
        fields = ["run_id", "target_id", "tree_id", "tree_sha256", "tree_path", "selected", "selection_rule"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in canonical_entries(entries):
            writer.writerow({**{key: entry[key] for key in fields[:5]}, "selected": entry["tree_id"] in selected_keys,
                "selection_rule": "fixed named actual FLOW-fork cases; no baseline rescue claim" if entry["tree_id"] in selected_keys else "candidate inventory"})
    (analysis_root / "captions.md").write_text(
        "# Observed-cohort figure legends\n\n"
        f"Figure 2. Executable inverse-electron-flow search. The primary run contains {len(route_entries)} recorded target trees. "
        "(a) Nonzero-step frozen-stock AND-closure lower bounds on all 460 declared targets; unfinished targets remain pending, and already-stock targets are separate. "
        "No equal-compute comparison is implied. (b) All native states of the named example; bold edges mark a recorded accepted root-to-terminal chain, not a verified physical mechanism. "
        "Action depth is not synthesis-route depth. (c) Empirical distributions on the observed target universe, with shared contexts counted per target. "
        "(d) Two distinct recorded FLOW successors from one shared state. Curved arrows use the actual current-state private atom references.\n\n"
        "Figure 3. Observed chemistry and source-pool composition. (a) Stock-closure lower bounds by reference-route depth on the same frozen RetroCast 160 IDs (40 at each depth). "
        "These are not C1 composition labels. (b) Actual nested hash-sample accumulation from the historical strict 257,167-reaction pool: 604 local and 6,553 ordered complete signatures. "
        "These historical signatures are distinct from the current actor's 603/6,491 exposure. (c) A complete generated inverse program from the first named case; "
        "each FLOW cell shows the recorded pre-action mixture and the final cell its structural precursor projection. This does not establish H2 generalization.\n\n"
        "Cabozantinib note. The benchmark label contains S-malate, but the recorded target is the free base. The exact acid-chloride/aniline structural coupling "
        "has patent precedent; neither the displayed proton-transfer choices nor the actor's full auxiliary mixture is thereby experimentally verified.\n\n"
        "Extended Data. Whole native-state forests and chemical microscopes retain all accepted nodes. Morphology, rejection taxonomy and execution counts are descriptive; "
        "different levels have different counting units. The 31k ranking plots use the saved rxn_prod_min proxy diagnostic, not a corrected complete-test result. "
        "The ambiguity example is the lexicographically first source-ID with a recorded-pool hit but NLL Top-1 miss; it is a ranking failure, not proof of chemically impossible alternatives. "
        "Independent full-reaction literature evidence, source overlap, H2-trained inference and matched-budget controls remain separately audited.\n")
    if not all(item["machine_checks_passed"] for item in qa["figures"].values()):
        raise ValueError("vector delivery checks failed; inspect figure_qa.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--case-id", required=True, action="append")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--route-assets-manifest", type=Path)
    parser.add_argument("--historical-report", type=Path)
    parser.add_argument("--endpoint-run")
    args = parser.parse_args()
    if args.historical_report:
        result = render_observed(args.analysis_root, args.style, case_ids=args.case_id, run_id=args.run_id,
            assets_manifest=args.route_assets_manifest, historical_report=args.historical_report, endpoint_run=args.endpoint_run)
        print(json.dumps({"phase": "observed-figures", "figures": list(result["figures"])}), flush=True)
        return
    args.case_id = args.case_id[0]
    import yaml
    style = yaml.safe_load(args.style.read_text())
    inventory = json.loads((args.analysis_root / "input_inventory.json").read_text())
    matches = [e for e in inventory["inputs"] if e.get("target_id") == args.case_id and e.get("run_id") == args.run_id and e.get("cache_path")]
    if len(matches) != 1:
        raise ValueError("select one recorded source case")
    row = matches[0]
    if file_digest(row["tree_path"]) != row["tree_sha256"]:
        raise ValueError("original tree changed since normalization")
    with gzip.open(row["cache_path"], "rt") as f:
        data = json.load(f)
    tree, tree_info = tree_svg(data, style)
    doc = svg_document(label(6, 12, args.case_id) + embed(tree, 5, 20, 508, 290), style, height_pt=330)
    out = args.analysis_root / "figures"
    atlas = export(doc, out / "tree_atlas", style)
    microscope, info = render_microscope(data, style)
    micro = export(microscope, out / "node_microscope", style)
    manifest = {"style_sha256": file_digest(args.style), "case_id": args.case_id,
        "tree_sha256": row["tree_sha256"], "source_path": row["tree_path"],
        "tree": tree_info, "chemical_fork": info, "exports": {"tree_atlas": atlas, "node_microscope": micro},
        "complete_figure2_figure3": False, "new_sampling": False}
    (args.analysis_root / "case_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"phase": "actual-case-vectors", "case": args.case_id, "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
