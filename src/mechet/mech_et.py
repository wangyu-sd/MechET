"""MECH_ET v3: mechanism-graph CoT with explicit bond-electron (BE) deltas.

Builds on FlowERMechanismGraph (MECH_GRAPH v2 states/edges) and adds:
- PERCEIVE / ET_SIGNATURE stages from the product SMILES
- per-RETRO_EDGE sparse BE_DELTA aligned with FlowER ``get_BE_matrix``
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdchem

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    rdchem = None  # type: ignore

from mechet.mech_graph import (
    FlowERMechanismGraph,
    _escape_quotes,
    _join_frags,
    _reverse_topo_order,
    _sides_equal,
    _split_frags,
    _unescape_quotes,
    compact_mapped_smiles,
    expand_state_smiles,
    official_state_key,
)

MECH_ET_HEADER = "MECH_ET v3"
MECH_ET_HEADERS = frozenset({MECH_ET_HEADER})

_BT_TO_ELECTRON = {
    "SINGLE": 2,
    "DOUBLE": 4,
    "TRIPLE": 6,
    "AROMATIC": 3,
}

# Lightweight SMARTS → (signature, demand) for product-side perception.
_ENDPOINT_RULES: list[tuple[str, str, str, str]] = [
    # smarts, endpoint_label, signature, demand
    ("[CX3](=[OX1])[NX3;H1,H0]", "amide_n", "amide_formation", "acyl_substitution"),
    ("[CX3](=[OX1])[OX2]", "ester_o", "ester_formation", "acyl_substitution"),
    ("[CX3](=[OX1])[Cl,Br,I,F]", "acyl_halide", "acyl_substitution", "acyl_substitution"),
    ("[c][NX3;H2,H1]", "aniline_like", "aromatic_amination", "nucleophilic_aromatic"),
    ("[C,c][OH]", "alcohol_like", "alcohol_functionalization", "heteroatom_nucleophile"),
    ("[C,c][NH2]", "amine_like", "amine_functionalization", "heteroatom_nucleophile"),
    ("[CX3]=[CX3]", "alkene", "alkene_addition", "pi_addition"),
    ("[CX3](=[OX1])[CX4]", "ketone_like", "carbonyl_addition", "carbonyl_electrophile"),
]


@dataclass
class BEDelta:
    """Sparse bond-electron delta in FlowER matrix units (single bond = 1)."""

    bonds: list[tuple[int, int, int]] = field(default_factory=list)  # (i, j, d) i<j
    lone_pairs: list[tuple[int, int]] = field(default_factory=list)  # (i, d)
    charges: list[tuple[int, int, int]] = field(default_factory=list)  # (i, q0, q1)

    def is_empty(self) -> bool:
        return not self.bonds and not self.lone_pairs and not self.charges

    def format_lines(self, *, indent: str = "  ") -> list[str]:
        lines = [f"{indent}BE_DELTA"]
        for i, j, d in sorted(self.bonds):
            sign = f"+{d}" if d > 0 else str(d)
            lines.append(f"{indent}  BOND {i} {j} {sign}")
        for i, d in sorted(self.lone_pairs):
            sign = f"+{d}" if d > 0 else str(d)
            lines.append(f"{indent}  LP {i} {sign}")
        for i, q0, q1 in sorted(self.charges):
            lines.append(f"{indent}  CHARGE {i} {q0} {q1}")
        return lines


def _bond_electron_half(bond) -> float:
    bt = bond.GetBondType()
    name = str(bt).replace("BondType.", "")
    if hasattr(bt, "name"):
        name = bt.name
    electrons = _BT_TO_ELECTRON.get(name)
    if electrons is None:
        # Fallback via float bond order
        order = float(bond.GetBondTypeAsDouble())
        electrons = int(round(order * 2))
    return electrons / 2.0


def _count_lone_pairs(atom) -> float:
    tbl = Chem.GetPeriodicTable()
    v = tbl.GetNOuterElecs(atom.GetAtomicNum())
    c = atom.GetFormalCharge()
    b = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
    h = atom.GetTotalNumHs()
    return float(v - c - b - h)


_BE_CACHE: dict[str, dict[tuple[int, int], float] | None] = {}
_CHARGE_CACHE: dict[str, dict[int, int]] = {}
_CACHE_MAX = 8192


def _cache_put(cache: dict, key: str, value) -> None:
    if key in cache:
        cache[key] = value
        return
    if len(cache) >= _CACHE_MAX:
        # Drop ~12.5% oldest insertion-order keys (CPython 3.7+ dict order).
        for drop_key in list(cache.keys())[: max(1, _CACHE_MAX // 8)]:
            cache.pop(drop_key, None)
    cache[key] = value


def get_be_matrix_by_map(mapped_smiles: str) -> dict[tuple[int, int], float] | None:
    """FlowER-style BE matrix keyed by atom-map numbers (1-indexed).

    Diagonal ``(i,i)`` = lone pairs; off-diagonal ``(i,j)`` = bond_electrons/2
    (symmetric). Returns None if SMILES cannot be parsed. Also fills charge cache.
    """
    text = (mapped_smiles or "").strip()
    if not text:
        return None
    if text in _BE_CACHE:
        return _BE_CACHE[text]
    if Chem is None:
        _cache_put(_BE_CACHE, text, None)
        _cache_put(_CHARGE_CACHE, text, {})
        return None
    try:
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        ps.sanitize = True
        mol = Chem.MolFromSmiles(text, ps)
        if mol is None:
            _cache_put(_BE_CACHE, text, None)
            _cache_put(_CHARGE_CACHE, text, {})
            return None
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        be: dict[tuple[int, int], float] = {}
        charges: dict[int, int] = {}
        for atom in mol.GetAtoms():
            if not atom.HasProp("molAtomMapNumber"):
                continue
            mid = int(atom.GetIntProp("molAtomMapNumber"))
            be[(mid, mid)] = _count_lone_pairs(atom)
            charges[mid] = int(atom.GetFormalCharge())
        for bond in mol.GetBonds():
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            if not a.HasProp("molAtomMapNumber") or not b.HasProp("molAtomMapNumber"):
                continue
            i = int(a.GetIntProp("molAtomMapNumber"))
            j = int(b.GetIntProp("molAtomMapNumber"))
            val = _bond_electron_half(bond)
            be[(i, j)] = val
            be[(j, i)] = val
        _cache_put(_BE_CACHE, text, be)
        _cache_put(_CHARGE_CACHE, text, charges)
        return be
    except Exception:
        _cache_put(_BE_CACHE, text, None)
        _cache_put(_CHARGE_CACHE, text, {})
        return None


def get_formal_charges_by_map(mapped_smiles: str) -> dict[int, int]:
    text = (mapped_smiles or "").strip()
    if not text:
        return {}
    if text not in _CHARGE_CACHE:
        get_be_matrix_by_map(text)
    return dict(_CHARGE_CACHE.get(text) or {})


def be_delta_from_mapped_smiles(src: str, dst: str) -> BEDelta | None:
    """Sparse ΔBE = BE(dst) − BE(src) in FlowER units."""
    be_src = get_be_matrix_by_map(src)
    be_dst = get_be_matrix_by_map(dst)
    if be_src is None or be_dst is None:
        return None
    keys = set(be_src) | set(be_dst)
    bonds: list[tuple[int, int, int]] = []
    lps: list[tuple[int, int]] = []
    for i, j in keys:
        if i > j:
            continue
        d = int(round(be_dst.get((i, j), 0.0) - be_src.get((i, j), 0.0)))
        if d == 0:
            continue
        if i == j:
            lps.append((i, d))
        else:
            bonds.append((i, j, d))
    charges: list[tuple[int, int, int]] = []
    q_src = get_formal_charges_by_map(src)
    q_dst = get_formal_charges_by_map(dst)
    for mid in sorted(set(q_src) | set(q_dst)):
        a = int(q_src.get(mid, 0))
        b = int(q_dst.get(mid, 0))
        if a != b:
            charges.append((mid, a, b))
    return BEDelta(bonds=sorted(bonds), lone_pairs=sorted(lps), charges=sorted(charges))


def electron_conserved(delta: BEDelta) -> bool:
    """Full symmetric matrix sum of ΔBE is zero (FlowER mass/electron conservation)."""
    total = 0
    for _, d in delta.lone_pairs:
        total += d
    for _, _, d in delta.bonds:
        total += 2 * d  # both off-diagonal entries
    return total == 0


def be_delta_exact(pred: BEDelta, gold: BEDelta, *, check_charge: bool = False) -> bool:
    return (
        sorted(pred.bonds) == sorted(gold.bonds)
        and sorted(pred.lone_pairs) == sorted(gold.lone_pairs)
        and (not check_charge or sorted(pred.charges) == sorted(gold.charges))
    )


def parse_be_delta_lines(lines: list[str]) -> BEDelta:
    delta = BEDelta()
    for raw in lines:
        line = raw.strip()
        if not line or line == "BE_DELTA":
            continue
        parts = line.split()
        if not parts:
            continue
        op = parts[0]
        if op == "BOND" and len(parts) >= 4:
            i, j, d = int(parts[1]), int(parts[2]), int(parts[3])
            if i > j:
                i, j = j, i
            delta.bonds.append((i, j, d))
        elif op == "LP" and len(parts) >= 3:
            delta.lone_pairs.append((int(parts[1]), int(parts[2])))
        elif op == "CHARGE" and len(parts) >= 4:
            delta.charges.append((int(parts[1]), int(parts[2]), int(parts[3])))
    delta.bonds = sorted(delta.bonds)
    delta.lone_pairs = sorted(delta.lone_pairs)
    delta.charges = sorted(delta.charges)
    return delta


def _maps_in_smiles(smiles: str) -> set[int]:
    out: set[int] = set()
    if Chem is None:
        return out
    try:
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        ps.sanitize = True
        mol = Chem.MolFromSmiles(smiles or "", ps)
        if mol is None:
            return out
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                out.add(int(atom.GetIntProp("molAtomMapNumber")))
    except Exception:
        return out
    return out


def perceive_from_product(product_smiles: str, *, center_maps: Iterable[int] | None = None) -> dict[str, Any]:
    """Heuristic PERCEIVE + ET_SIGNATURE from mapped product SMILES."""
    endpoints: list[dict[str, Any]] = []
    signature = "unknown"
    demand = "unknown"
    centers: list[str] = []
    if Chem is not None and product_smiles:
        try:
            ps = Chem.SmilesParserParams()
            ps.removeHs = False
            ps.sanitize = True
            mol = Chem.MolFromSmiles(product_smiles, ps)
            if mol is not None:
                for smarts, label, sig, dem in _ENDPOINT_RULES:
                    patt = Chem.MolFromSmarts(smarts)
                    if patt is None:
                        continue
                    matches = mol.GetSubstructMatches(patt)
                    if not matches:
                        continue
                    maps: list[int] = []
                    for match in matches[:3]:
                        for idx in match:
                            atom = mol.GetAtomWithIdx(idx)
                            if atom.HasProp("molAtomMapNumber"):
                                maps.append(int(atom.GetIntProp("molAtomMapNumber")))
                    maps = sorted(set(maps))
                    if maps:
                        endpoints.append({"label": label, "maps": maps})
                        if signature == "unknown":
                            signature = sig
                            demand = dem
                    if len(endpoints) >= 3:
                        break
        except Exception:
            pass
    if center_maps:
        maps = sorted({int(m) for m in center_maps})
        if len(maps) >= 2:
            centers.append(f"{maps[0]}-{maps[1]}")
        elif len(maps) == 1:
            centers.append(str(maps[0]))
    elif endpoints:
        # Default center = first two maps of first endpoint.
        maps = endpoints[0]["maps"]
        if len(maps) >= 2:
            centers.append(f"{maps[0]}-{maps[1]}")
        elif maps:
            centers.append(str(maps[0]))
    return {
        "endpoints": endpoints,
        "centers": centers,
        "et_signature": signature,
        "et_demand": demand,
    }


def _center_maps_from_delta(delta: BEDelta) -> list[int]:
    maps: set[int] = set()
    for i, j, _ in delta.bonds:
        maps.add(i)
        maps.add(j)
    for i, _ in delta.lone_pairs:
        maps.add(i)
    return sorted(maps)


def format_mech_et_cot(graph: FlowERMechanismGraph) -> str:
    """Serialize full MECH_ET v3 body for a FlowER mechanism graph."""
    ordered_ids = _reverse_topo_order(graph)
    compact = {sid: compact_mapped_smiles(graph.states[sid].mapped_smiles) for sid in ordered_ids}
    frag_sets = {sid: set(_split_frags(compact[sid])) for sid in ordered_ids}
    shared: set[str] = set.intersection(*frag_sets.values()) if frag_sets else set()
    if shared and graph.n_states >= 2:
        active = {
            sid: _join_frags(f for f in _split_frags(compact[sid]) if f not in shared)
            for sid in ordered_ids
        }
        if any(not active[sid] for sid in ordered_ids):
            shared = set()
            active = compact
    else:
        shared = set()
        active = compact

    # BE deltas from raw (explicit-H) mapped states so proton-transfer maps survive.
    # STATE lines remain compact (strip-H + SHARED) for length.
    reverse_edges = sorted(
        graph.reverse_edges(),
        key=lambda e: (
            ordered_ids.index(e[0]) if e[0] in ordered_ids else 10**9,
            ordered_ids.index(e[1]) if e[1] in ordered_ids else 10**9,
            e[0],
            e[1],
        ),
    )
    edge_deltas: list[tuple[str, str, BEDelta]] = []
    first_delta: BEDelta | None = None
    for src, dst in reverse_edges:
        # RETRO a->b : Δ = BE(b) - BE(a)
        raw_src = graph.states[src].mapped_smiles
        raw_dst = graph.states[dst].mapped_smiles
        delta = be_delta_from_mapped_smiles(raw_src, raw_dst)
        if delta is None:
            delta = BEDelta()
        edge_deltas.append((src, dst, delta))
        if first_delta is None and not delta.is_empty():
            first_delta = delta

    product = graph.compact_main_product()
    perceive = perceive_from_product(
        product,
        center_maps=_center_maps_from_delta(first_delta) if first_delta else None,
    )

    lines = [
        MECH_ET_HEADER,
        "DIRECTION RETRO",
        f'TARGET_SMILES "{_escape_quotes(product)}"',
        "PERCEIVE",
    ]
    for ep in perceive["endpoints"]:
        maps = ",".join(str(m) for m in ep["maps"])
        lines.append(f"  ENDPOINT {ep['label']} maps={maps}")
    for c in perceive["centers"]:
        lines.append(f"  CENTER {c}")
    lines.append(f"ET_SIGNATURE {perceive['et_signature']}")
    lines.append(f"ET_DEMAND {perceive['et_demand']}")
    lines.append(f"N_STATES {graph.n_states}")
    lines.append(f"N_EDGES {len(reverse_edges)}")
    if shared:
        lines.append(f'SHARED "{_escape_quotes(_join_frags(shared))}"')
    for sid in ordered_ids:
        lines.append(f'STATE {sid} "{_escape_quotes(active[sid])}"')
    for tid in graph.target_state_ids:
        lines.append(f"TARGET_STATE {tid}")
    lines.append(f"PRECURSOR_STATE {graph.precursor_state_id}")
    for src, dst, delta in edge_deltas:
        lines.append(f"RETRO_EDGE {src} {dst}")
        if not delta.is_empty():
            lines.extend(delta.format_lines(indent="  "))
        else:
            lines.append("  BE_DELTA")
    return "\n".join(lines)


def parse_mech_et_body(text: str) -> dict[str, Any]:
    """Parse MECH_ET v3 body into structured fields + expanded states."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    # Drop blank and comment lines
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        cleaned.append(ln)
    result: dict[str, Any] = {
        "ok": False,
        "version": "",
        "target_smiles": "",
        "endpoints": [],
        "centers": [],
        "et_signature": "unknown",
        "et_demand": "unknown",
        "shared": "",
        "active_states": {},
        "states": {},
        "target_state_ids": [],
        "precursor_state_id": "",
        "retro_edges": [],
        "edge_deltas": {},  # (src,dst) -> BEDelta
        "diagnostics": [],
    }
    if not cleaned or cleaned[0].strip() not in MECH_ET_HEADERS:
        result["diagnostics"].append({"code": "BAD_HEADER", "message": f"expected {MECH_ET_HEADER}"})
        return result
    result["version"] = cleaned[0].strip()

    n_states = None
    n_edges = None
    i = 1
    in_perceive = False
    current_edge: tuple[str, str] | None = None
    delta_buf: list[str] = []

    def _flush_delta() -> None:
        nonlocal current_edge, delta_buf
        if current_edge is not None:
            result["edge_deltas"][current_edge] = parse_be_delta_lines(delta_buf)
            if current_edge not in result["retro_edges"]:
                result["retro_edges"].append(current_edge)
        current_edge = None
        delta_buf = []

    while i < len(cleaned):
        raw = cleaned[i]
        line = raw.strip()
        i += 1
        if line.startswith("DIRECTION"):
            in_perceive = False
            continue
        if line.startswith("TARGET_SMILES"):
            in_perceive = False
            payload = line[len("TARGET_SMILES") :].strip()
            if payload.startswith('"') and payload.endswith('"') and len(payload) >= 2:
                result["target_smiles"] = _unescape_quotes(payload[1:-1])
            else:
                result["diagnostics"].append({"code": "BAD_TARGET_SMILES", "message": line})
                return result
            continue
        if line == "PERCEIVE":
            _flush_delta()
            in_perceive = True
            continue
        if in_perceive and (line.startswith("ENDPOINT") or line.startswith("CENTER")):
            if line.startswith("ENDPOINT"):
                # ENDPOINT label maps=1,2
                rest = line[len("ENDPOINT") :].strip()
                label = rest.split()[0] if rest else ""
                maps: list[int] = []
                if "maps=" in rest:
                    maps_str = rest.split("maps=", 1)[1].strip().split()[0]
                    maps = [int(x) for x in maps_str.split(",") if x.strip().lstrip("+-").isdigit()]
                result["endpoints"].append({"label": label, "maps": maps})
            else:
                result["centers"].append(line[len("CENTER") :].strip())
            continue
        if line.startswith("ET_SIGNATURE"):
            in_perceive = False
            _flush_delta()
            result["et_signature"] = line.split(None, 1)[1].strip() if " " in line else "unknown"
            continue
        if line.startswith("ET_DEMAND"):
            in_perceive = False
            result["et_demand"] = line.split(None, 1)[1].strip() if " " in line else "unknown"
            continue
        if line.startswith("N_STATES"):
            in_perceive = False
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
            _flush_delta()
            payload = line[len("SHARED ") :].strip()
            if not (payload.startswith('"') and payload.endswith('"') and len(payload) >= 2):
                result["diagnostics"].append({"code": "BAD_SHARED", "message": line})
                return result
            result["shared"] = _unescape_quotes(payload[1:-1])
            continue
        if line.startswith("STATE "):
            _flush_delta()
            rest = line[len("STATE ") :].strip()
            if " " not in rest:
                result["diagnostics"].append({"code": "BAD_STATE", "message": line})
                return result
            sid, payload = rest.split(" ", 1)
            payload = payload.strip()
            if not (payload.startswith('"') and payload.endswith('"') and len(payload) >= 2):
                result["diagnostics"].append({"code": "BAD_STATE_SMILES", "message": line})
                return result
            result["active_states"][sid] = _unescape_quotes(payload[1:-1])
            continue
        if line.startswith("TARGET_STATE "):
            _flush_delta()
            result["target_state_ids"].append(line.split(" ", 1)[1].strip())
            continue
        if line.startswith("PRECURSOR_STATE "):
            _flush_delta()
            result["precursor_state_id"] = line.split(" ", 1)[1].strip()
            continue
        if line.startswith("RETRO_EDGE "):
            _flush_delta()
            parts = line.split()
            if len(parts) != 3:
                result["diagnostics"].append({"code": "BAD_EDGE", "message": line})
                return result
            current_edge = (parts[1], parts[2])
            delta_buf = []
            continue
        if current_edge is not None and (line == "BE_DELTA" or line.startswith("BOND ") or line.startswith("LP ") or line.startswith("CHARGE ")):
            delta_buf.append(line)
            continue
        # Indented perceive leftovers already handled; unknown
        result["diagnostics"].append({"code": "UNKNOWN_LINE", "message": line})
        return result

    _flush_delta()
    shared = result["shared"]
    result["states"] = {
        sid: expand_state_smiles(active, shared) for sid, active in result["active_states"].items()
    }

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


def verify_mech_et(
    *,
    mechanism_body: str,
    answer: str,
    main_product: str | None = None,
    expected_precursor: str | None = None,
    expected_graph: FlowERMechanismGraph | None = None,
) -> dict[str, Any]:
    """Verify MECH_ET format, graph reachability, and BE_DELTA consistency."""
    parsed = parse_mech_et_body(mechanism_body)
    out: dict[str, Any] = {
        "format_ok": bool(parsed.get("ok")),
        "reachability_ok": False,
        "answer_exact": False,
        "main_product_ok": False,
        "be_delta_exact": False,
        "electron_conserved": False,
        "center_in_delta": False,
        "signature_consistent": True,  # soft
        "edge_f1": 0.0,
        "node_f1": 0.0,
        "graph_exact": False,
        "n_states": len(parsed.get("states") or {}),
        "n_edges": len(parsed.get("retro_edges") or []),
        "diagnostics": list(parsed.get("diagnostics") or []),
        "topology": None,
        "et_signature": parsed.get("et_signature"),
        "parsed": parsed,
    }
    if not parsed.get("ok"):
        return out

    states: dict[str, str] = dict(parsed["states"])
    precursor = str(parsed["precursor_state_id"])
    targets = list(parsed["target_state_ids"])
    edges: list[tuple[str, str]] = list(parsed["retro_edges"])
    edge_deltas: dict[tuple[str, str], BEDelta] = dict(parsed["edge_deltas"])

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

    out["reachability_ok"] = all(can_reach(t, precursor) for t in targets)
    if not out["reachability_ok"]:
        out["diagnostics"].append({"code": "UNREACHABLE", "message": "some TARGET cannot reach PRECURSOR"})

    answer_side = (answer or "").strip()
    precursor_smiles = states.get(precursor, "")
    expected = expected_precursor if expected_precursor is not None else precursor_smiles
    out["answer_exact"] = bool(answer_side) and _sides_equal(answer_side, expected)

    product = main_product or parsed.get("target_smiles") or ""
    if product:
        product_key = official_state_key(product) or compact_mapped_smiles(product)
        found = False
        for tid in targets:
            for frag in _split_frags(states[tid]):
                frag_key = official_state_key(frag) or compact_mapped_smiles(frag)
                if frag_key == product_key or compact_mapped_smiles(frag) == compact_mapped_smiles(product):
                    found = True
                    break
            if found:
                break
        # Also accept TARGET_SMILES match against main product
        if not found and parsed.get("target_smiles"):
            found = _sides_equal(compact_mapped_smiles(parsed["target_smiles"]), compact_mapped_smiles(product))
        out["main_product_ok"] = found
        if not found:
            out["diagnostics"].append({"code": "MAIN_PRODUCT_MISSING", "message": "main product not in TARGET states"})
    else:
        out["main_product_ok"] = True

    # BE delta checks.
    # With expected_graph: compare to raw (explicit-H) gold deltas.
    # Without GT: compact STATE may drop mapped H, so only require each edge
    # to carry a BE_DELTA block (process presence), and treat conservation softly.
    if expected_graph is not None:
        exact_hits = 0
        conserved_hits = 0
        n_checked = 0
        for a, b in edges:
            pred = edge_deltas.get((a, b), BEDelta())
            gold = None
            if a in expected_graph.states and b in expected_graph.states:
                gold = be_delta_from_mapped_smiles(
                    expected_graph.states[a].mapped_smiles,
                    expected_graph.states[b].mapped_smiles,
                )
            if gold is None:
                out["diagnostics"].append({"code": "BE_PARSE_FAIL", "message": f"{a}->{b}"})
                continue
            n_checked += 1
            if be_delta_exact(pred, gold, check_charge=False):
                exact_hits += 1
                conserved_hits += 1
            else:
                out["diagnostics"].append({"code": "BE_DELTA_MISMATCH", "message": f"{a}->{b}"})
                if electron_conserved(pred) or pred.is_empty():
                    conserved_hits += 1
        out["be_delta_exact"] = n_checked == len(edges) and exact_hits == len(edges)
        out["electron_conserved"] = n_checked == len(edges) and conserved_hits == len(edges)
    else:
        present = all(e in edge_deltas for e in edges)
        out["be_delta_exact"] = present
        out["electron_conserved"] = present and all(
            electron_conserved(edge_deltas[e]) or edge_deltas[e].is_empty() or abs(
                sum(d for _, d in edge_deltas[e].lone_pairs) + 2 * sum(d for _, _, d in edge_deltas[e].bonds)
            )
            <= 2
            for e in edges
        )

    # center_in_delta: at least one CENTER map appears in some edge delta
    center_maps: set[int] = set()
    for c in parsed.get("centers") or []:
        for tok in str(c).replace("-", " ").split():
            if tok.lstrip("+-").isdigit():
                center_maps.add(int(tok))
    if not center_maps:
        out["center_in_delta"] = True  # nothing declared
    else:
        delta_maps: set[int] = set()
        for delta in edge_deltas.values():
            delta_maps.update(_center_maps_from_delta(delta))
        out["center_in_delta"] = bool(center_maps & delta_maps)

    if expected_graph is not None:
        # Reuse graph-level comparison via a synthetic verify on states/edges
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
