"""FlowER full mechanism graphs (chains/trees/DAGs) as CoT for retrosynthesis.

Follows the official FlowER sequence evaluator semantics in
``FlowER/sequence_evaluation.py``:

- same ``sequence_idx`` elementary steps form a directed graph;
- unique start = in-degree 0 node;
- terminals = nodes with a self-loop;
- evaluation enumerates simple paths from start to terminals.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


MECH_GRAPH_HEADER = "MECH_GRAPH v2"
MECH_GRAPH_HEADERS = frozenset({"MECH_GRAPH v1", "MECH_GRAPH v2"})


@dataclass
class GraphState:
    state_id: str
    mapped_smiles: str
    canonical_key: str


@dataclass
class FlowERMechanismGraph:
    """One FlowER reaction mechanism as a directed state graph."""

    trajectory_id: str
    states: dict[str, GraphState] = field(default_factory=dict)
    # Forward non-self edges: src_id -> list of dst_id
    forward_edges: list[tuple[str, str]] = field(default_factory=list)
    precursor_state_id: str = ""
    target_state_ids: list[str] = field(default_factory=list)
    source_path: str | None = None
    diagnostics: list[dict[str, str]] = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_edges(self) -> int:
        return len(self.forward_edges)

    @property
    def precursor_smiles(self) -> str:
        if not self.precursor_state_id:
            return ""
        return self.states[self.precursor_state_id].mapped_smiles

    @property
    def target_smiles_list(self) -> list[str]:
        return [self.states[sid].mapped_smiles for sid in self.target_state_ids if sid in self.states]

    @property
    def main_product(self) -> str:
        # Prefer heaviest fragment among all terminal states.
        best = ""
        best_heavy = -1
        for smiles in self.target_smiles_list:
            for frag in [p.strip() for p in smiles.split(".") if p.strip()]:
                heavy = _heavy_atom_count(frag)
                if heavy > best_heavy or (heavy < 0 and len(frag) > len(best)):
                    best_heavy = heavy
                    best = frag
        return best

    @property
    def topology(self) -> str:
        if self.n_states <= 1 and self.n_edges == 0:
            return "singleton"
        out_deg = defaultdict(int)
        in_deg = defaultdict(int)
        for src, dst in self.forward_edges:
            out_deg[src] += 1
            in_deg[dst] += 1
        branch = sum(1 for v in out_deg.values() if v > 1)
        join = sum(1 for v in in_deg.values() if v > 1)
        if branch == 0 and join == 0:
            return "linear"
        if branch > 0 and join == 0:
            return "tree"
        if branch > 0 and join > 0:
            return "dag_branch_join"
        if join > 0:
            return "dag_join"
        return "other"

    def reverse_edges(self) -> list[tuple[str, str]]:
        """Non-self forward edges reversed for retrosynthesis CoT."""
        return [(dst, src) for src, dst in self.forward_edges]

    def format_mechanism_body(self) -> str:
        """Serialize reverse mech-graph CoT (v2: strip-H + SHARED spectators)."""
        ordered_ids = _reverse_topo_order(self)
        compact = {sid: compact_mapped_smiles(self.states[sid].mapped_smiles) for sid in ordered_ids}
        frag_sets = {sid: set(_split_frags(compact[sid])) for sid in ordered_ids}
        shared: set[str] = set.intersection(*frag_sets.values()) if frag_sets else set()
        # Keep SHARED only when it actually shortens the body.
        if shared and self.n_states >= 2:
            active = {
                sid: _join_frags(f for f in _split_frags(compact[sid]) if f not in shared)
                for sid in ordered_ids
            }
            # Guard: never emit a completely empty active state.
            if any(not active[sid] for sid in ordered_ids):
                shared = set()
                active = compact
        else:
            shared = set()
            active = compact

        lines = [
            MECH_GRAPH_HEADER,
            "DIRECTION RETRO",
            f"N_STATES {self.n_states}",
            f"N_EDGES {len(self.reverse_edges())}",
        ]
        if shared:
            lines.append(f'SHARED "{_escape_quotes(_join_frags(shared))}"')
        for sid in ordered_ids:
            lines.append(f'STATE {sid} "{_escape_quotes(active[sid])}"')
        for tid in self.target_state_ids:
            lines.append(f"TARGET_STATE {tid}")
        lines.append(f"PRECURSOR_STATE {self.precursor_state_id}")
        id_rank = {sid: i for i, sid in enumerate(ordered_ids)}
        for src, dst in sorted(
            self.reverse_edges(),
            key=lambda e: (id_rank.get(e[0], 10**9), id_rank.get(e[1], 10**9), e[0], e[1]),
        ):
            lines.append(f"RETRO_EDGE {src} {dst}")
        return "\n".join(lines)

    def compact_precursor_smiles(self) -> str:
        """Strip-H precursor system SMILES used as the SFT <answer>."""
        return compact_mapped_smiles(self.precursor_smiles)

    def compact_main_product(self) -> str:
        return compact_mapped_smiles(self.main_product)


_STATE_KEY_CACHE: dict[str, str | None] = {}
_STATE_KEY_CACHE_MAX = 8192


def official_state_key(mapped_smiles: str) -> str | None:
    """Canonical state key matching FlowER sequence_evaluation.clean (approx).

    Official eval removes maps after RemoveHs; we use sanitize + ClearProp
    when possible. Results are cached because the same intermediate states
    recur many times within and across trajectories.
    """
    text = (mapped_smiles or "").strip()
    if not text:
        return None
    if text in _STATE_KEY_CACHE:
        return _STATE_KEY_CACHE[text]
    key: str | None
    if Chem is None:
        key = ".".join(sorted(p.strip() for p in text.split(".") if p.strip()))
    else:
        try:
            ps = Chem.SmilesParserParams()
            ps.removeHs = False
            ps.sanitize = True
            mol = Chem.MolFromSmiles(text, ps)
            if mol is None:
                key = None
            else:
                try:
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    pass
                for atom in mol.GetAtoms():
                    atom.ClearProp("molAtomMapNumber")
                key = Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            key = None
    if len(_STATE_KEY_CACHE) >= _STATE_KEY_CACHE_MAX:
        for drop_key in list(_STATE_KEY_CACHE.keys())[: max(1, _STATE_KEY_CACHE_MAX // 8)]:
            _STATE_KEY_CACHE.pop(drop_key, None)
    _STATE_KEY_CACHE[text] = key
    return key


def get_main_product(products_smiles: str) -> str:
    product_list = [p.strip() for p in (products_smiles or "").split(".") if p.strip()]
    if not product_list:
        return ""
    main_product = product_list[0]
    max_heavy = -1
    for smiles in product_list:
        heavy = _heavy_atom_count(smiles)
        if heavy > max_heavy:
            max_heavy = heavy
            main_product = smiles
        elif heavy < 0 and len(smiles) > len(main_product):
            main_product = smiles
    return main_product


def _heavy_atom_count(smiles: str) -> int:
    if Chem is None:
        return -1
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return -1
    if mol is None:
        return -1
    return int(mol.GetNumHeavyAtoms())


def _escape_quotes(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _unescape_quotes(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _split_frags(smiles: str) -> list[str]:
    return [p.strip() for p in (smiles or "").split(".") if p.strip()]


def _join_frags(frags: Iterable[str]) -> str:
    return ".".join(sorted(p.strip() for p in frags if p and p.strip()))


_COMPACT_CACHE: dict[str, str] = {}
_COMPACT_CACHE_MAX = 8192


def compact_mapped_smiles(smiles: str) -> str:
    """Keep atom maps, drop explicit hydrogens; fail soft to sorted frags."""
    text = (smiles or "").strip()
    if not text:
        return ""
    cached = _COMPACT_CACHE.get(text)
    if cached is not None:
        return cached
    if Chem is None:
        out = _join_frags(_split_frags(text))
    else:
        try:
            ps = Chem.SmilesParserParams()
            ps.removeHs = False
            ps.sanitize = True
            mol = Chem.MolFromSmiles(text, ps)
            if mol is None:
                out = _join_frags(_split_frags(text))
            else:
                try:
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    pass
                out = Chem.MolToSmiles(mol)
        except Exception:
            out = _join_frags(_split_frags(text))
    if len(_COMPACT_CACHE) >= _COMPACT_CACHE_MAX:
        for drop_key in list(_COMPACT_CACHE.keys())[: max(1, _COMPACT_CACHE_MAX // 8)]:
            _COMPACT_CACHE.pop(drop_key, None)
    _COMPACT_CACHE[text] = out
    return out


def expand_state_smiles(active: str, shared: str = "") -> str:
    """Reconstruct full system SMILES from SHARED + per-state ACTIVE frags."""
    return _join_frags(_split_frags(shared) + _split_frags(active))


def parse_flower_line(line: str) -> tuple[str, str, str] | None:
    text = (line or "").strip()
    if not text or ">>" not in text or "|" not in text:
        return None
    rxn, trajectory_id = text.rsplit("|", 1)
    trajectory_id = trajectory_id.strip()
    if not trajectory_id or ">>" not in rxn:
        return None
    reactants, products = rxn.split(">>", 1)
    reactants, products = reactants.strip(), products.strip()
    if not reactants or not products:
        return None
    return reactants, products, trajectory_id


def build_mechanism_graph(
    trajectory_id: str,
    steps: list[tuple[str, str]],
    *,
    source_path: str | None = None,
) -> FlowERMechanismGraph | None:
    """Build an official-semantic mechanism graph from elementary steps."""
    diagnostics: list[dict[str, str]] = []
    if not steps:
        return None

    # Map canonical key -> representative mapped smiles (first occurrence).
    key_to_mapped: dict[str, str] = {}
    key_order: list[str] = []
    forward_pairs: list[tuple[str, str]] = []  # canonical keys
    self_loop_keys: set[str] = set()

    for reactants, products in steps:
        rk = official_state_key(reactants)
        pk = official_state_key(products)
        if rk is None or pk is None:
            diagnostics.append({"code": "PARSE_BAD", "message": "unparseable SMILES in step"})
            return None
        if rk not in key_to_mapped:
            key_to_mapped[rk] = reactants
            key_order.append(rk)
        if pk not in key_to_mapped:
            key_to_mapped[pk] = products
            key_order.append(pk)
        if rk == pk:
            self_loop_keys.add(rk)
        else:
            forward_pairs.append((rk, pk))

    # Deduplicate forward edges while preserving order.
    seen_edges: set[tuple[str, str]] = set()
    unique_forward: list[tuple[str, str]] = []
    for edge in forward_pairs:
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        unique_forward.append(edge)

    # Official start: in-degree 0 (self-loops count toward in-degree in NX).
    # We compute NX-style in-degree including self-loops.
    nodes = set(key_to_mapped)
    indeg = {n: 0 for n in nodes}
    for a, b in unique_forward:
        indeg[b] += 1
    for loop in self_loop_keys:
        indeg[loop] += 1  # self-loop contributes 1 in-degree in NetworkX
    starts = [n for n in key_order if indeg.get(n, 0) == 0]
    # Keep only nodes that appear; preserve discovery order.
    starts = [n for n in starts if n in nodes]
    if len(starts) != 1:
        diagnostics.append({"code": "BAD_ROOT", "message": f"expected 1 start, got {len(starts)}"})
        return None
    if not self_loop_keys:
        diagnostics.append({"code": "NO_TERMINAL", "message": "no self-loop terminal"})
        return None

    root_key = starts[0]
    # Assign stable state IDs in reverse-topo preference after building ids.
    # First create temporary ids by discovery order, then renumber.
    tmp_ids = {key: f"t{i}" for i, key in enumerate(key_order)}
    tmp_states = {
        tmp_ids[k]: GraphState(state_id=tmp_ids[k], mapped_smiles=key_to_mapped[k], canonical_key=k)
        for k in key_order
    }
    tmp_edges = [(tmp_ids[a], tmp_ids[b]) for a, b in unique_forward]
    tmp_targets = [tmp_ids[k] for k in key_order if k in self_loop_keys]
    tmp_graph = FlowERMechanismGraph(
        trajectory_id=trajectory_id,
        states=tmp_states,
        forward_edges=tmp_edges,
        precursor_state_id=tmp_ids[root_key],
        target_state_ids=tmp_targets,
        source_path=source_path,
        diagnostics=diagnostics,
    )
    # Renumber by reverse topo: targets first, precursor last.
    ordered = _reverse_topo_order(tmp_graph)
    new_ids = {old: f"s{i}" for i, old in enumerate(ordered)}
    states = {
        new_ids[old]: GraphState(
            state_id=new_ids[old],
            mapped_smiles=tmp_states[old].mapped_smiles,
            canonical_key=tmp_states[old].canonical_key,
        )
        for old in ordered
    }
    edges = [(new_ids[a], new_ids[b]) for a, b in tmp_edges]
    targets = [new_ids[t] for t in tmp_targets]
    precursor = new_ids[tmp_graph.precursor_state_id]
    return FlowERMechanismGraph(
        trajectory_id=trajectory_id,
        states=states,
        forward_edges=edges,
        precursor_state_id=precursor,
        target_state_ids=targets,
        source_path=source_path,
        diagnostics=diagnostics,
    )


def _reverse_topo_order(graph: FlowERMechanismGraph) -> list[str]:
    """Order states for RETRO serialization: terminals early, precursor late."""
    ids = list(graph.states.keys())
    if not ids:
        return []
    # Build reverse adjacency (for retrosynthesis topo: from targets toward precursor).
    rev_adj: dict[str, list[str]] = defaultdict(list)
    indeg = {sid: 0 for sid in ids}
    for src, dst in graph.forward_edges:
        # reverse edge dst -> src
        rev_adj[dst].append(src)
        indeg[src] += 1
    # Start from targets (indegree 0 in reverse graph when they have no outgoing reverse? 
    # In reverse graph, targets have no incoming from forward edges' reverse if they only receive...
    # Forward: a->b means reverse b->a. Target T is usually a sink in forward (except self-loop removed).
    # So in reverse, T has out-edges to parents, and indegree 0 if nothing points to T in reverse.
    # Reverse indegree of X = number of forward edges Y->X... wait:
    # reverse edge is (dst, src) for forward (src,dst). So reverse adj[dst].append(src).
    # indegree_rev[src] += 1 for each forward edge.
    # Targets (sinks in forward) have forward out-degree 0, so reverse in-degree 0. Good.
    queue = deque([sid for sid in ids if indeg[sid] == 0])
    # Prefer target states first among zero-indegree.
    target_set = set(graph.target_state_ids)
    ordered: list[str] = []
    # Stable: sort initial queue with targets first, then id.
    queue = deque(sorted(queue, key=lambda s: (0 if s in target_set else 1, s)))
    seen = set()
    while queue:
        # pop leftmost; keep queue sorted by inserting in order
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        ordered.append(node)
        for nxt in sorted(rev_adj[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                # insert keeping targets-first preference among newly ready
                queue.append(nxt)
        # re-sort queue lightly for stability
        queue = deque(sorted(queue, key=lambda s: (0 if s in target_set else 1, s)))
    # Append any leftover (cycles) in id order.
    for sid in sorted(ids):
        if sid not in seen:
            ordered.append(sid)
    # Ensure precursor is last if present and unique.
    if graph.precursor_state_id in ordered and ordered[-1] != graph.precursor_state_id:
        ordered = [s for s in ordered if s != graph.precursor_state_id] + [graph.precursor_state_id]
    return ordered


def load_flower_graphs(
    split_path: Path | str,
    *,
    limit: int | None = None,
) -> tuple[list[FlowERMechanismGraph], dict[str, int]]:
    """Stream-group FlowER elementary steps and build valid mechanism graphs."""
    path = Path(split_path)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    order: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_flower_line(line)
            if parsed is None:
                continue
            reactants, products, tid = parsed
            if tid not in groups:
                if limit is not None and len(order) >= limit:
                    break
                order.append(tid)
            groups[tid].append((reactants, products))

    skip = defaultdict(int)
    graphs: list[FlowERMechanismGraph] = []
    for tid in order:
        graph = build_mechanism_graph(tid, groups[tid], source_path=str(path))
        if graph is None:
            skip["invalid_official"] += 1
            continue
        graphs.append(graph)
    return graphs, dict(skip)


def parse_mech_graph_body(text: str) -> dict[str, Any]:
    """Parse a MECH_GRAPH v1/v2 body.

    v2 may include a SHARED spectator block; returned ``states`` are always the
    fully expanded system SMILES (SHARED + ACTIVE).
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    result: dict[str, Any] = {
        "ok": False,
        "version": "",
        "shared": "",
        "active_states": {},
        "states": {},
        "target_state_ids": [],
        "precursor_state_id": "",
        "retro_edges": [],
        "diagnostics": [],
    }
    if not lines or lines[0] not in MECH_GRAPH_HEADERS:
        result["diagnostics"].append({"code": "BAD_HEADER", "message": f"expected one of {sorted(MECH_GRAPH_HEADERS)}"})
        return result
    result["version"] = lines[0]
    i = 1
    n_states = None
    n_edges = None
    shared = ""
    active_states: dict[str, str] = {}
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("DIRECTION"):
            continue
        if line.startswith("N_STATES"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                n_states = int(parts[1])
            else:
                result["diagnostics"].append({"code": "BAD_N_STATES", "message": line})
                return result
            continue
        if line.startswith("N_EDGES"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                n_edges = int(parts[1])
            else:
                result["diagnostics"].append({"code": "BAD_N_EDGES", "message": line})
                return result
            continue
        if line.startswith("SHARED "):
            payload = line[len("SHARED ") :].strip()
            if not (payload.startswith('"') and payload.endswith('"') and len(payload) >= 2):
                result["diagnostics"].append({"code": "BAD_SHARED", "message": line})
                return result
            shared = _unescape_quotes(payload[1:-1])
            result["shared"] = shared
            continue
        if line.startswith("STATE "):
            rest = line[len("STATE ") :].strip()
            if " " not in rest:
                result["diagnostics"].append({"code": "BAD_STATE", "message": line})
                return result
            sid, payload = rest.split(" ", 1)
            payload = payload.strip()
            if not (payload.startswith('"') and payload.endswith('"') and len(payload) >= 2):
                result["diagnostics"].append({"code": "BAD_STATE_SMILES", "message": line})
                return result
            smiles = _unescape_quotes(payload[1:-1])
            active_states[sid] = smiles
            continue
        if line.startswith("TARGET_STATE "):
            result["target_state_ids"].append(line.split(" ", 1)[1].strip())
            continue
        if line.startswith("PRECURSOR_STATE "):
            result["precursor_state_id"] = line.split(" ", 1)[1].strip()
            continue
        if line.startswith("RETRO_EDGE "):
            parts = line.split()
            if len(parts) != 3:
                result["diagnostics"].append({"code": "BAD_EDGE", "message": line})
                return result
            result["retro_edges"].append((parts[1], parts[2]))
            continue
        result["diagnostics"].append({"code": "UNKNOWN_LINE", "message": line})
        return result

    result["active_states"] = dict(active_states)
    result["states"] = {sid: expand_state_smiles(active, shared) for sid, active in active_states.items()}

    if n_states is not None and n_states != len(result["states"]):
        result["diagnostics"].append(
            {"code": "N_STATES_MISMATCH", "message": f"declared {n_states} got {len(result['states'])}"}
        )
        return result
    if n_edges is not None and n_edges != len(result["retro_edges"]):
        result["diagnostics"].append(
            {"code": "N_EDGES_MISMATCH", "message": f"declared {n_edges} got {len(result['retro_edges'])}"}
        )
        return result
    if not result["precursor_state_id"] or result["precursor_state_id"] not in result["states"]:
        result["diagnostics"].append({"code": "BAD_PRECURSOR", "message": "missing/invalid PRECURSOR_STATE"})
        return result
    if not result["target_state_ids"]:
        result["diagnostics"].append({"code": "NO_TARGET", "message": "missing TARGET_STATE"})
        return result
    for tid in result["target_state_ids"]:
        if tid not in result["states"]:
            result["diagnostics"].append({"code": "BAD_TARGET", "message": tid})
            return result
    for a, b in result["retro_edges"]:
        if a not in result["states"] or b not in result["states"]:
            result["diagnostics"].append({"code": "DANGLING_EDGE", "message": f"{a}->{b}"})
            return result
    result["ok"] = True
    return result


def verify_mech_graph(
    *,
    mechanism_body: str,
    answer: str,
    main_product: str | None = None,
    expected_precursor: str | None = None,
    expected_graph: FlowERMechanismGraph | None = None,
) -> dict[str, Any]:
    """Verify MECH_GRAPH format, reachability, and optional GT alignment."""
    parsed = parse_mech_graph_body(mechanism_body)
    out: dict[str, Any] = {
        "format_ok": bool(parsed.get("ok")),
        "reachability_ok": False,
        "answer_exact": False,
        "main_product_ok": False,
        "edge_f1": 0.0,
        "node_f1": 0.0,
        "graph_exact": False,
        "n_states": len(parsed.get("states") or {}),
        "n_edges": len(parsed.get("retro_edges") or []),
        "diagnostics": list(parsed.get("diagnostics") or []),
        "topology": None,
    }
    if not parsed.get("ok"):
        return out

    states: dict[str, str] = dict(parsed["states"])
    precursor = str(parsed["precursor_state_id"])
    targets = list(parsed["target_state_ids"])
    edges: list[tuple[str, str]] = list(parsed["retro_edges"])

    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)

    def can_reach(src: str, dst: str) -> bool:
        seen = {src}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                return True
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    reach_ok = all(can_reach(t, precursor) for t in targets)
    out["reachability_ok"] = reach_ok
    if not reach_ok:
        out["diagnostics"].append({"code": "UNREACHABLE", "message": "some TARGET cannot reach PRECURSOR"})

    answer_side = (answer or "").strip()
    precursor_smiles = states.get(precursor, "")
    expected = expected_precursor if expected_precursor is not None else precursor_smiles
    out["answer_exact"] = bool(answer_side) and _sides_equal(answer_side, expected)

    if main_product:
        product_key = official_state_key(main_product) or compact_mapped_smiles(main_product)
        found = False
        for tid in targets:
            for frag in _split_frags(states[tid]):
                frag_key = official_state_key(frag) or compact_mapped_smiles(frag)
                if frag_key == product_key or frag == main_product.strip() or compact_mapped_smiles(frag) == compact_mapped_smiles(main_product):
                    found = True
                    break
            if found:
                break
        out["main_product_ok"] = found
        if not found:
            out["diagnostics"].append({"code": "MAIN_PRODUCT_MISSING", "message": "main product not in TARGET states"})
    else:
        out["main_product_ok"] = True

    if expected_graph is not None:
        pred_nodes = {official_state_key(s) for s in states.values()}
        gt_nodes = {st.canonical_key for st in expected_graph.states.values()}
        pred_nodes.discard(None)
        out["node_f1"] = _f1(pred_nodes, gt_nodes)

        pred_edges = set()
        for a, b in edges:
            ka = official_state_key(states[a])
            kb = official_state_key(states[b])
            if ka and kb:
                pred_edges.add((ka, kb))
        gt_edges = set()
        for src, dst in expected_graph.reverse_edges():
            gt_edges.add((expected_graph.states[src].canonical_key, expected_graph.states[dst].canonical_key))
        out["edge_f1"] = _f1(pred_edges, gt_edges)
        out["graph_exact"] = pred_nodes == gt_nodes and pred_edges == gt_edges
        out["topology"] = expected_graph.topology
    return out


def _sides_equal(a: str, b: str) -> bool:
    """Equality under strip-H / official demap when possible."""
    if _side_multiset(a) == _side_multiset(b):
        return True
    if _side_multiset(compact_mapped_smiles(a)) == _side_multiset(compact_mapped_smiles(b)):
        return True
    ka = official_state_key(a)
    kb = official_state_key(b)
    return bool(ka and kb and ka == kb)


def _side_multiset(side: str) -> tuple[str, ...]:
    return tuple(sorted(p.strip() for p in (side or "").split(".") if p.strip()))


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def format_mech_graph_cot(graph: FlowERMechanismGraph) -> str:
    return graph.format_mechanism_body()
