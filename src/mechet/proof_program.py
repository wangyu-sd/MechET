"""Executable proof-carrying retrosynthesis programs.

``MECH_PROOF v1`` removes model-authored intermediate states and the free-form
answer channel. A model emits only root imports and sparse electron-flow
operations. The executor reconstructs every state and derives the precursor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from rdkit import Chem

PROOF_HEADER = "MECH_PROOF v1"
PROOF_OPEN = "<proof>"
PROOF_CLOSE = "</proof>"

_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


class ProofProgramError(ValueError):
    """Raised when a proof cannot be parsed, compiled, or executed."""


@dataclass(frozen=True)
class ChargeAction:
    atom_map: int
    q0: int
    q1: int


@dataclass
class ProofEdge:
    src: str
    dst: str
    imports: list[str] = field(default_factory=list)
    bonds: list[tuple[int, int, int]] = field(default_factory=list)
    lone_pairs: list[tuple[int, int]] = field(default_factory=list)
    charges: list[ChargeAction] = field(default_factory=list)


@dataclass
class ProofProgram:
    target_smiles: str
    roots: dict[str, list[str]]
    precursor_state_id: str
    edges: list[ProofEdge]


@dataclass
class ProofExecutionResult:
    ok: bool
    precursor_smiles: str = ""
    states: dict[str, str] = field(default_factory=dict)
    diagnostics: list[dict[str, str]] = field(default_factory=list)


def _escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


def _unescape(text: str) -> str:
    out: list[str] = []
    escaped = False
    for char in text:
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            out.append(char)
    if escaped:
        out.append("\\")
    return "".join(out)


def _quoted_payload(line: str, prefix: str) -> str:
    payload = line[len(prefix) :].strip()
    if len(payload) < 2 or not payload.startswith('"') or not payload.endswith('"'):
        raise ProofProgramError(f"expected quoted payload: {line}")
    return _unescape(payload[1:-1])


def format_proof_program(program: ProofProgram) -> str:
    lines = [PROOF_HEADER, f'TARGET_SMILES "{_escape(program.target_smiles)}"']
    for root_id, imports in program.roots.items():
        lines.append(f"ROOT {root_id}")
        for fragment in imports:
            lines.append(f'  IMPORT "{_escape(fragment)}"')
    lines.append(f"PRECURSOR_STATE {program.precursor_state_id}")
    for edge in program.edges:
        lines.append(f"EDGE {edge.src} {edge.dst}")
        for fragment in edge.imports:
            lines.append(f'  IMPORT "{_escape(fragment)}"')
        for i, j, delta in edge.bonds:
            sign = f"+{delta}" if delta > 0 else str(delta)
            lines.append(f"  BOND {min(i, j)} {max(i, j)} {sign}")
        for atom_map, delta in edge.lone_pairs:
            sign = f"+{delta}" if delta > 0 else str(delta)
            lines.append(f"  LP {atom_map} {sign}")
        for charge in edge.charges:
            lines.append(f"  CHARGE {charge.atom_map} {charge.q0} {charge.q1}")
    return "\n".join(lines)


def format_proof_output(program: ProofProgram) -> str:
    return f"{PROOF_OPEN}\n{format_proof_program(program)}\n{PROOF_CLOSE}"


def extract_proof_body(text: str) -> str:
    raw = (text or "").strip()
    lower = raw.lower()
    start = lower.find(PROOF_OPEN)
    if start < 0:
        return raw if raw.startswith(PROOF_HEADER) else ""
    content_start = start + len(PROOF_OPEN)
    end = lower.find(PROOF_CLOSE, content_start)
    if end < 0:
        return ""
    return raw[content_start:end].strip()


def parse_proof_program(text: str) -> ProofProgram:
    body = extract_proof_body(text)
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines or lines[0] != PROOF_HEADER:
        raise ProofProgramError(f"expected {PROOF_HEADER}")

    target = ""
    roots: dict[str, list[str]] = {}
    precursor = ""
    edges: list[ProofEdge] = []
    current_root: str | None = None
    current_edge: ProofEdge | None = None

    for line in lines[1:]:
        if line.startswith("TARGET_SMILES "):
            target = _quoted_payload(line, "TARGET_SMILES ")
            current_root = None
            current_edge = None
        elif line.startswith("ROOT "):
            parts = line.split()
            if len(parts) != 2 or parts[1] in roots:
                raise ProofProgramError(f"invalid ROOT: {line}")
            current_root = parts[1]
            roots[current_root] = []
            current_edge = None
        elif line.startswith("PRECURSOR_STATE "):
            parts = line.split()
            if len(parts) != 2:
                raise ProofProgramError(f"invalid PRECURSOR_STATE: {line}")
            precursor = parts[1]
            current_root = None
            current_edge = None
        elif line.startswith("EDGE "):
            parts = line.split()
            if len(parts) != 3:
                raise ProofProgramError(f"invalid EDGE: {line}")
            current_edge = ProofEdge(parts[1], parts[2])
            edges.append(current_edge)
            current_root = None
        elif line.startswith("IMPORT "):
            fragment = _quoted_payload(line, "IMPORT ")
            if current_edge is not None:
                current_edge.imports.append(fragment)
            elif current_root is not None:
                roots[current_root].append(fragment)
            else:
                raise ProofProgramError(f"IMPORT outside ROOT/EDGE: {line}")
        elif line.startswith("BOND "):
            if current_edge is None:
                raise ProofProgramError(f"BOND outside EDGE: {line}")
            parts = line.split()
            if len(parts) != 4:
                raise ProofProgramError(f"invalid BOND: {line}")
            i, j, delta = int(parts[1]), int(parts[2]), int(parts[3])
            if i == j or delta == 0:
                raise ProofProgramError(f"invalid BOND action: {line}")
            current_edge.bonds.append((min(i, j), max(i, j), delta))
        elif line.startswith("LP "):
            if current_edge is None:
                raise ProofProgramError(f"LP outside EDGE: {line}")
            parts = line.split()
            if len(parts) != 3:
                raise ProofProgramError(f"invalid LP: {line}")
            current_edge.lone_pairs.append((int(parts[1]), int(parts[2])))
        elif line.startswith("CHARGE "):
            if current_edge is None:
                raise ProofProgramError(f"CHARGE outside EDGE: {line}")
            parts = line.split()
            if len(parts) != 4:
                raise ProofProgramError(f"invalid CHARGE: {line}")
            current_edge.charges.append(
                ChargeAction(int(parts[1]), int(parts[2]), int(parts[3]))
            )
        else:
            raise ProofProgramError(f"unknown proof line: {line}")

    if not target or not roots or not precursor or not edges:
        raise ProofProgramError(
            "proof requires TARGET_SMILES, ROOT, PRECURSOR_STATE, and EDGE"
        )
    if precursor in roots and not any(edge.dst == precursor for edge in edges):
        raise ProofProgramError("precursor cannot be an unchanged root")
    return ProofProgram(target, roots, precursor, edges)


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = True
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ProofProgramError(f"unparseable mapped SMILES: {smiles[:120]}")
    return mol


def _canonical_mapped(smiles_or_mol: str | Chem.Mol) -> str:
    mol = (
        _mol_from_smiles(smiles_or_mol)
        if isinstance(smiles_or_mol, str)
        else Chem.Mol(smiles_or_mol)
    )
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
        allHsExplicit=False,
    )


def _canonical_unmapped(smiles: str) -> str:
    mol = _mol_from_smiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def sides_equal(left: str, right: str, *, ignore_maps: bool = True) -> bool:
    def parts(value: str) -> list[str]:
        out: list[str] = []
        for item in (value or "").split("."):
            item = item.strip()
            if not item:
                continue
            out.append(
                _canonical_unmapped(item)
                if ignore_maps
                else _canonical_mapped(item)
            )
        return sorted(out)

    try:
        return bool(left and right) and parts(left) == parts(right)
    except ProofProgramError:
        return False


def _combine_smiles(base: str, imports: Iterable[str]) -> Chem.Mol:
    molecules = [_mol_from_smiles(base)] + [
        _mol_from_smiles(fragment) for fragment in imports if fragment
    ]
    combined = molecules[0]
    for mol in molecules[1:]:
        combined = Chem.CombineMols(combined, mol)
    combined = Chem.Mol(combined)
    try:
        Chem.Kekulize(combined, clearAromaticFlags=True)
    except Exception:
        pass
    return combined


def _atom_map_index(mol: Chem.Mol) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for atom in mol.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map <= 0:
            raise ProofProgramError("all proof atoms must carry positive atom maps")
        if atom_map in mapping:
            raise ProofProgramError(f"duplicate atom map {atom_map}")
        mapping[atom_map] = atom.GetIdx()
    return mapping


def _set_bond_order(
    rw: Chem.RWMol,
    map_index: dict[int, int],
    i: int,
    j: int,
    delta: int,
) -> None:
    if i not in map_index or j not in map_index:
        raise ProofProgramError(f"BOND references missing map: {i},{j}")
    ai, aj = map_index[i], map_index[j]
    bond = rw.GetBondBetweenAtoms(ai, aj)
    current = int(round(bond.GetBondTypeAsDouble())) if bond is not None else 0
    new_order = current + delta
    if new_order < 0 or new_order > 3:
        raise ProofProgramError(
            f"invalid bond order transition {i}-{j}: {current}->{new_order}"
        )
    if bond is not None:
        rw.RemoveBond(ai, aj)
    if new_order:
        rw.AddBond(ai, aj, _BOND_TYPES[new_order])


def _get_be_delta(
    src: str,
    dst: str,
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int]],
    list[ChargeAction],
]:
    def be(smiles: str) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
        mol = _mol_from_smiles(smiles)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        matrix: dict[tuple[int, int], int] = {}
        charges: dict[int, int] = {}
        table = Chem.GetPeriodicTable()
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map <= 0:
                continue
            valence = table.GetNOuterElecs(atom.GetAtomicNum())
            charge = atom.GetFormalCharge()
            bond_order = sum(
                int(round(bond.GetBondTypeAsDouble())) for bond in atom.GetBonds()
            )
            hydrogens = atom.GetTotalNumHs()
            matrix[(atom_map, atom_map)] = int(
                round(valence - charge - bond_order - hydrogens)
            )
            charges[atom_map] = charge
        for bond in mol.GetBonds():
            i = bond.GetBeginAtom().GetAtomMapNum()
            j = bond.GetEndAtom().GetAtomMapNum()
            if i > 0 and j > 0:
                value = int(round(bond.GetBondTypeAsDouble()))
                matrix[(i, j)] = value
                matrix[(j, i)] = value
        return matrix, charges

    src_be, src_q = be(src)
    dst_be, dst_q = be(dst)
    bonds: list[tuple[int, int, int]] = []
    lone_pairs: list[tuple[int, int]] = []
    for i, j in sorted(set(src_be) | set(dst_be)):
        if i > j:
            continue
        delta = dst_be.get((i, j), 0) - src_be.get((i, j), 0)
        if not delta:
            continue
        if i == j:
            lone_pairs.append((i, delta))
        else:
            bonds.append((i, j, delta))
    charges = [
        ChargeAction(atom_map, src_q.get(atom_map, 0), dst_q.get(atom_map, 0))
        for atom_map in sorted(set(src_q) | set(dst_q))
        if src_q.get(atom_map, 0) != dst_q.get(atom_map, 0)
    ]
    return bonds, lone_pairs, charges


def _execute_edge(source: str, edge: ProofEdge) -> str:
    source_augmented_mol = _combine_smiles(source, edge.imports)
    source_augmented = _canonical_mapped(source_augmented_mol)
    rw = Chem.RWMol(source_augmented_mol)
    map_index = _atom_map_index(rw)

    for i, j, delta in edge.bonds:
        _set_bond_order(rw, map_index, i, j, delta)
    for action in edge.charges:
        if action.atom_map not in map_index:
            raise ProofProgramError(
                f"CHARGE references missing map {action.atom_map}"
            )
        atom = rw.GetAtomWithIdx(map_index[action.atom_map])
        if atom.GetFormalCharge() != action.q0:
            raise ProofProgramError(
                f"CHARGE precondition failed for {action.atom_map}: "
                f"expected {action.q0}, got {atom.GetFormalCharge()}"
            )
        atom.SetFormalCharge(action.q1)

    result = rw.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception as exc:
        raise ProofProgramError(
            f"RDKit sanitization failed on {edge.src}->{edge.dst}: {exc}"
        ) from exc
    destination = _canonical_mapped(result)

    derived_bonds, derived_lps, derived_charges = _get_be_delta(
        source_augmented,
        destination,
    )
    if sorted(derived_bonds) != sorted(edge.bonds):
        raise ProofProgramError(
            f"BOND execution mismatch on {edge.src}->{edge.dst}: "
            f"written={sorted(edge.bonds)} derived={sorted(derived_bonds)}"
        )
    if sorted(derived_lps) != sorted(edge.lone_pairs):
        raise ProofProgramError(
            f"LP execution mismatch on {edge.src}->{edge.dst}: "
            f"written={sorted(edge.lone_pairs)} derived={sorted(derived_lps)}"
        )
    electron_delta = sum(delta for _, delta in derived_lps) + 2 * sum(
        delta for _, _, delta in derived_bonds
    )
    if electron_delta != 0:
        raise ProofProgramError(
            f"electron conservation failed on {edge.src}->{edge.dst}: "
            f"delta={electron_delta}"
        )
    key = lambda action: (action.atom_map, action.q0, action.q1)
    if sorted(derived_charges, key=key) != sorted(edge.charges, key=key):
        raise ProofProgramError(
            f"CHARGE execution mismatch on {edge.src}->{edge.dst}"
        )
    return destination


def execute_proof(program_or_text: ProofProgram | str) -> ProofExecutionResult:
    try:
        program = (
            parse_proof_program(program_or_text)
            if isinstance(program_or_text, str)
            else program_or_text
        )
        states = {
            root_id: _canonical_mapped(
                _combine_smiles(program.target_smiles, imports)
            )
            for root_id, imports in program.roots.items()
        }
        pending = list(program.edges)
        while pending:
            progressed = False
            remaining: list[ProofEdge] = []
            for edge in pending:
                if edge.src not in states:
                    remaining.append(edge)
                    continue
                destination = _execute_edge(states[edge.src], edge)
                if edge.dst in states and not sides_equal(
                    states[edge.dst],
                    destination,
                    ignore_maps=False,
                ):
                    raise ProofProgramError(
                        f"DAG join mismatch at state {edge.dst}"
                    )
                states[edge.dst] = destination
                progressed = True
            if not progressed:
                unresolved = ",".join(
                    f"{edge.src}->{edge.dst}" for edge in remaining
                )
                raise ProofProgramError(
                    f"unreachable proof edges: {unresolved}"
                )
            pending = remaining
        if program.precursor_state_id not in states:
            raise ProofProgramError(
                f"precursor state {program.precursor_state_id} was not derived"
            )
        return ProofExecutionResult(
            True,
            states[program.precursor_state_id],
            states,
            [],
        )
    except ProofProgramError as exc:
        return ProofExecutionResult(
            False,
            diagnostics=[
                {
                    "code": "PROOF_EXECUTION_FAILED",
                    "message": str(exc),
                }
            ],
        )


def verify_proof(
    text: str,
    *,
    expected_precursor: str | None = None,
) -> dict[str, Any]:
    result = execute_proof(text)
    endpoint_exact = bool(
        result.ok
        and expected_precursor
        and sides_equal(result.precursor_smiles, expected_precursor)
    )
    return {
        "format_ok": bool(extract_proof_body(text)),
        "execute_ok": result.ok,
        "derived_precursor": result.precursor_smiles,
        "endpoint_exact": endpoint_exact,
        "diagnostics": result.diagnostics,
        "n_states": len(result.states),
    }


def _fragment_smiles(smiles: str) -> list[str]:
    mol = _mol_from_smiles(smiles)
    return [
        Chem.MolToSmiles(
            fragment,
            canonical=True,
            isomericSmiles=True,
        )
        for fragment in Chem.GetMolFrags(
            mol,
            asMols=True,
            sanitizeFrags=True,
        )
    ]


def _map_set(smiles: str) -> set[int]:
    return {
        atom.GetAtomMapNum()
        for atom in _mol_from_smiles(smiles).GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def _imports_for_root(target_smiles: str, root_state: str) -> list[str]:
    target_maps = _map_set(target_smiles)
    imports: list[str] = []
    found_target = False
    for fragment in _fragment_smiles(root_state):
        maps = _map_set(fragment)
        if maps == target_maps and sides_equal(
            fragment,
            target_smiles,
            ignore_maps=False,
        ):
            found_target = True
        else:
            imports.append(fragment)
    if not found_target:
        raise ProofProgramError(
            "TARGET_SMILES is not an exact fragment of a target state"
        )
    return imports


def _extract_new_atom_fragments(
    destination: str,
    new_maps: set[int],
) -> list[str]:
    if not new_maps:
        return []
    mol = _mol_from_smiles(destination)
    indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() in new_maps
    ]
    if not indices:
        return []
    atom_set = set(indices)
    bonds = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in atom_set
        and bond.GetEndAtomIdx() in atom_set
    ]
    fragment = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=indices,
        bondsToUse=bonds,
        canonical=True,
        isomericSmiles=True,
        allBondsExplicit=True,
    )
    return [part for part in fragment.split(".") if part]


def compile_from_states(
    *,
    target_smiles: str,
    target_state_ids: list[str],
    precursor_state_id: str,
    states: dict[str, str],
    edges: list[tuple[str, str]],
) -> ProofProgram:
    """Compile a state-annotated mechanism into an action-only proof."""
    roots = {
        state_id: _imports_for_root(target_smiles, states[state_id])
        for state_id in target_state_ids
    }
    proof_edges: list[ProofEdge] = []
    for src, dst in edges:
        if src not in states or dst not in states:
            raise ProofProgramError(
                f"edge references missing state {src}->{dst}"
            )
        src_maps = _map_set(states[src])
        dst_maps = _map_set(states[dst])
        removed = src_maps - dst_maps
        if removed:
            raise ProofProgramError(
                f"proof edge removes mapped atoms {sorted(removed)}"
            )
        imports = _extract_new_atom_fragments(
            states[dst],
            dst_maps - src_maps,
        )
        source_augmented = _canonical_mapped(
            _combine_smiles(states[src], imports)
        )
        bonds, lone_pairs, charges = _get_be_delta(
            source_augmented,
            states[dst],
        )
        proof_edges.append(
            ProofEdge(
                src,
                dst,
                imports,
                bonds,
                lone_pairs,
                charges,
            )
        )
    program = ProofProgram(
        target_smiles,
        roots,
        precursor_state_id,
        proof_edges,
    )
    executed = execute_proof(program)
    if not executed.ok:
        raise ProofProgramError(executed.diagnostics[0]["message"])
    if not sides_equal(
        executed.precursor_smiles,
        states[precursor_state_id],
        ignore_maps=False,
    ):
        raise ProofProgramError(
            "compiled proof does not reconstruct precursor state"
        )
    return program


def compile_mech_et_body(mechanism_body: str) -> ProofProgram:
    """Compile a ``MECH_ET v3`` body into ``MECH_PROOF v1``."""
    from mechet.mech_et import parse_mech_et_body

    parsed = parse_mech_et_body(mechanism_body)
    if not parsed.get("ok"):
        raise ProofProgramError(
            f"invalid MECH_ET v3 body: {parsed.get('diagnostics')}"
        )
    return compile_from_states(
        target_smiles=str(parsed.get("target_smiles") or ""),
        target_state_ids=list(parsed.get("target_state_ids") or []),
        precursor_state_id=str(parsed.get("precursor_state_id") or ""),
        states=dict(parsed.get("states") or {}),
        edges=list(parsed.get("retro_edges") or []),
    )
