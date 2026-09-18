"""Template-conditioned Python slots for inverse electron-flow programs.

The host owns ``mechanism(steps=STEPS)`` and the requested target.  The model
predicts only the ``STEPS`` expression.  Output is parsed as a restricted AST;
it is never evaluated or executed as Python.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping

from .proof_program import ProofProgramError
from .python_program import (
    PythonElectronProgram,
    PythonProgramExecutionResult,
    execute_python_program,
    extract_python_program,
    parse_python_program,
)

PYTHON_TEMPLATE_SLOTS_VERSION = "python_template_slots_v1"


def _container_code(value: Mapping[str, Any]) -> str:
    kind = str(value.get("kind") or "").upper().replace("LONE_PAIR", "LP")
    names = {
        "ATOM": "atom",
        "BOND": "bond",
        "LP": "lp",
        "RADICAL_PAIR": "radical_pair",
    }
    if kind not in names:
        raise ProofProgramError(f"unsupported electron container: {kind}")
    atoms = [int(item) for item in value.get("atoms") or ()]
    expected = 2 if kind in {"BOND", "RADICAL_PAIR"} else 1
    if len(atoms) != expected or any(item <= 0 for item in atoms):
        raise ProofProgramError(f"invalid {kind} container atoms: {atoms}")
    if expected == 2:
        atoms.sort()
    return f"{names[kind]}({','.join(map(str, atoms))})"


def _compact_list(values: list[Any]) -> str:
    return "[" + ",".join(repr(item) for item in values) + "]"


def _move_code(value: Mapping[str, Any]) -> str:
    if value.get("mode") == "BE_DELTA":
        bonds = [
            (*sorted(int(atom) for atom in item.get("atoms") or ()), int(item["delta"]))
            for item in value.get("bond_deltas") or ()
        ]
        charges = [
            (int(item["atom_map"]), int(item["q0"]), int(item["q1"]))
            for item in value.get("charge_actions") or ()
        ]
        fields = [f"bonds={_compact_list(bonds)}"]
        if charges:
            fields.append(f"charges={_compact_list(charges)}")
        return f"delta({','.join(fields)})"
    if int(value.get("electrons", 2)) != 2:
        raise ProofProgramError("v1 supports two-electron moves only")
    return f"move({_container_code(value['source'])},{_container_code(value['sink'])})"


def format_template_slots(program: PythonElectronProgram) -> str:
    """Serialize only the ``STEPS`` hole in a deterministic compact form."""

    steps: list[str] = []
    for item in program.steps:
        arguments = [_move_code(move) for move in item.moves]
        if item.imports:
            arguments.append(f"imports={_compact_list(list(item.imports))}")
        steps.append(f"step({','.join(arguments)})")
    if not steps:
        raise ProofProgramError("program requires at least one step")
    return "[" + ",".join(steps) + "]"


def parse_template_slots(text: str, *, target_smiles: str = "") -> PythonElectronProgram:
    """Bind a generated ``STEPS`` expression into the fixed safe template."""

    body = extract_python_program(text)
    if not body:
        raise ProofProgramError("missing STEPS expression")
    try:
        expression = ast.parse(body, mode="eval").body
    except SyntaxError as exc:
        raise ProofProgramError(f"invalid STEPS syntax: {exc.msg}") from exc
    if not isinstance(expression, (ast.List, ast.Tuple)) or not expression.elts:
        raise ProofProgramError("STEPS must be a non-empty list")
    wrapper = ast.Expression(
        body=ast.Call(
            func=ast.Name(id="mechanism", ctx=ast.Load()),
            args=[],
            keywords=[ast.keyword(arg="steps", value=expression)],
        )
    )
    normalized = ast.unparse(ast.fix_missing_locations(wrapper))
    parsed = parse_python_program(normalized)
    return PythonElectronProgram(target_smiles=target_smiles, steps=parsed.steps)


def execute_template_slots(
    text: str, *, target_smiles: str
) -> PythonProgramExecutionResult:
    try:
        return execute_python_program(
            parse_template_slots(text, target_smiles=target_smiles),
            target_smiles=target_smiles,
        )
    except (ProofProgramError, ValueError, KeyError, TypeError) as exc:
        return PythonProgramExecutionResult(
            False,
            diagnostics=[
                {"code": "TEMPLATE_SLOTS_EXECUTION_FAILED", "message": str(exc)}
            ],
        )
