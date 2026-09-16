"""Safe Python-shaped language for inverse electron-flow programs.

The syntax is familiar Python, but model output is never executed.  A strict
AST parser accepts only ``mechanism``, ``step``, ``move`` and electron-container
constructors.  Each accepted program is replayed by the existing chemical
executor, which alone derives the precursor endpoint.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .electron_flow_trace import ElectronFlowTrace, compile_trace_to_proof
from .forward_expert import verify_electron_step
from .proof_program import ProofProgramError

PYTHON_PROGRAM_VERSION = "python_electron_program_v1"
_CONTAINER_NAMES = {
    "atom": "ATOM",
    "bond": "BOND",
    "lp": "LP",
    "radical_pair": "RADICAL_PAIR",
}


@dataclass(frozen=True)
class PythonElectronStep:
    moves: tuple[dict[str, Any], ...]
    imports: tuple[str, ...] = ()


@dataclass(frozen=True)
class PythonElectronProgram:
    target_smiles: str
    steps: tuple[PythonElectronStep, ...]


@dataclass
class PythonProgramExecutionResult:
    ok: bool
    precursor_smiles: str = ""
    compiled_proof: str = ""
    flow_trace: ElectronFlowTrace | None = None
    diagnostics: list[dict[str, str]] = field(default_factory=list)


def _container_code(value: Mapping[str, Any]) -> str:
    kind = str(value.get("kind") or "").upper().replace("LONE_PAIR", "LP")
    reverse = {name: call for call, name in _CONTAINER_NAMES.items()}
    if kind not in reverse:
        raise ProofProgramError(f"unsupported electron container: {kind}")
    atoms = tuple(int(item) for item in value.get("atoms") or ())
    return f"{reverse[kind]}({', '.join(map(str, atoms))})"


def _move_code(value: Mapping[str, Any]) -> str:
    if value.get("mode") == "BE_DELTA":
        bonds = [
            (*tuple(int(atom) for atom in item.get("atoms") or ()), int(item["delta"]))
            for item in value.get("bond_deltas") or ()
        ]
        charges = [
            (int(item["atom_map"]), int(item["q0"]), int(item["q1"]))
            for item in value.get("charge_actions") or ()
        ]
        fields = [f"bonds={bonds!r}"]
        if charges:
            fields.append(f"charges={charges!r}")
        return f"delta({', '.join(fields)})"
    electrons = int(value.get("electrons", 2))
    if electrons != 2:
        raise ProofProgramError("v1 supports two-electron moves only")
    return (
        f"move({_container_code(value['source'])}, "
        f"{_container_code(value['sink'])})"
    )


def _coerce_program(
    program: PythonElectronProgram | ElectronFlowTrace,
) -> PythonElectronProgram:
    if isinstance(program, PythonElectronProgram):
        return program
    return PythonElectronProgram(
        target_smiles=program.target_smiles,
        steps=tuple(
            PythonElectronStep(tuple(item.moves), tuple(item.imports))
            for item in program.transitions
        ),
    )


def format_python_program(
    program: PythonElectronProgram | ElectronFlowTrace,
    *,
    fenced: bool = True,
    include_target: bool = False,
) -> str:
    """Serialize actual source-to-sink arrows as one deterministic expression."""

    value = _coerce_program(program)
    lines = ["mechanism("]
    if include_target:
        lines.append(f"  target={value.target_smiles!r},")
    lines.append("  steps=[")
    for item in value.steps:
        arguments = [_move_code(move) for move in item.moves]
        if item.imports:
            arguments.append(f"imports={list(item.imports)!r}")
        lines.append(f"    step({', '.join(arguments)}),")
    lines.extend(["  ],", ")"])
    body = "\n".join(lines)
    return f"```python\n{body}\n```" if fenced else body


def extract_python_program(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline < 0:
            return ""
        language = raw[3:first_newline].strip().lower()
        if language not in {"", "py", "python"}:
            return ""
        end = raw.find("```", first_newline + 1)
        if end < 0 or raw[end + 3 :].strip():
            return ""
        return raw[first_newline + 1 : end].strip()
    return raw


def _literal(node: ast.AST, label: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ProofProgramError(f"{label} must contain literals only") from exc


def _string(node: ast.AST, label: str) -> str:
    value = _literal(node, label)
    if not isinstance(value, str) or not value:
        raise ProofProgramError(f"{label} must be a non-empty string")
    return value


def _integer_tuple_list(
    node: ast.AST | None, label: str, width: int
) -> list[tuple[int, ...]]:
    if node is None:
        return []
    value = _literal(node, label)
    if not isinstance(value, list):
        raise ProofProgramError(f"{label} must be a list")
    output: list[tuple[int, ...]] = []
    for item in value:
        if not isinstance(item, (tuple, list)) or len(item) != width:
            raise ProofProgramError(f"{label} entries must have width {width}")
        if any(not isinstance(part, int) or isinstance(part, bool) for part in item):
            raise ProofProgramError(f"{label} entries must contain integers")
        output.append(tuple(item))
    return output


def _keywords(
    call: ast.Call, name: str, allowed: set[str]
) -> dict[str, ast.AST]:
    output: dict[str, ast.AST] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ProofProgramError(f"{name} does not allow ** expansion")
        if keyword.arg not in allowed:
            raise ProofProgramError(f"unknown {name} argument: {keyword.arg}")
        if keyword.arg in output:
            raise ProofProgramError(f"duplicate {name} argument: {keyword.arg}")
        output[keyword.arg] = keyword.value
    return output


def _parse_container(node: ast.AST) -> dict[str, Any]:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ProofProgramError("move endpoints must be electron-container calls")
    if node.func.id not in _CONTAINER_NAMES:
        raise ProofProgramError(f"forbidden electron container: {node.func.id}")
    if node.keywords:
        raise ProofProgramError("electron containers accept positional atoms only")
    atoms = [_literal(item, f"{node.func.id} atom") for item in node.args]
    if any(not isinstance(item, int) or isinstance(item, bool) for item in atoms):
        raise ProofProgramError("electron-container atoms must be integers")
    expected = 2 if node.func.id in {"bond", "radical_pair"} else 1
    if len(atoms) != expected or any(item <= 0 for item in atoms):
        raise ProofProgramError(
            f"{node.func.id} requires {expected} positive atom-map arguments"
        )
    if expected == 2:
        atoms = sorted(atoms)
    return {"kind": _CONTAINER_NAMES[node.func.id], "atoms": atoms}


def _parse_move(node: ast.AST) -> dict[str, Any]:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ProofProgramError("step entries must be move(...) or delta(...) calls")
    if node.func.id == "move":
        if len(node.args) != 2 or node.keywords:
            raise ProofProgramError("move requires exactly two positional containers")
        return {
            "source": _parse_container(node.args[0]),
            "sink": _parse_container(node.args[1]),
            "electrons": 2,
        }
    if node.func.id == "delta":
        if node.args:
            raise ProofProgramError("delta accepts keyword arguments only")
        values = _keywords(node, "delta", {"bonds", "charges"})
        bonds = _integer_tuple_list(values.get("bonds"), "delta.bonds", 3)
        charges = _integer_tuple_list(values.get("charges"), "delta.charges", 3)
        if not bonds:
            raise ProofProgramError("delta requires at least one bond change")
        return {
            "mode": "BE_DELTA",
            "bond_deltas": [
                {"atoms": sorted((left, right)), "delta": change}
                for left, right, change in bonds
            ],
            "charge_actions": [
                {"atom_map": atom_map, "q0": q0, "q1": q1}
                for atom_map, q0, q1 in charges
            ],
        }
    raise ProofProgramError(f"forbidden step operation: {node.func.id}")


def _parse_step(node: ast.AST, index: int) -> PythonElectronStep:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ProofProgramError(f"steps[{index}] must be a step(...) call")
    if node.func.id != "step":
        raise ProofProgramError(f"steps[{index}] calls forbidden function {node.func.id}")
    values = _keywords(node, "step", {"imports"})
    imports = _literal(values["imports"], "step.imports") if "imports" in values else []
    if not isinstance(imports, list) or any(
        not isinstance(item, str) or not item for item in imports
    ):
        raise ProofProgramError("step.imports must be a list of non-empty strings")
    if not node.args:
        raise ProofProgramError("step requires at least one electron move")
    if len(node.args) > 32:
        raise ProofProgramError("step exceeds the 32-move safety limit")
    moves = tuple(_parse_move(item) for item in node.args)
    if any(move.get("mode") == "BE_DELTA" for move in moves) and len(moves) != 1:
        raise ProofProgramError("delta must be the only operation in a step")
    return PythonElectronStep(moves=moves, imports=tuple(imports))


def parse_python_program(text: str) -> PythonElectronProgram:
    """Parse the restricted Python DSL without ``eval`` or ``exec``."""

    body = extract_python_program(text)
    if not body:
        raise ProofProgramError("missing Python electron program")
    try:
        expression = ast.parse(body, mode="eval").body
    except SyntaxError as exc:
        raise ProofProgramError(f"invalid Python syntax: {exc.msg}") from exc
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ProofProgramError("program must be one mechanism(...) expression")
    if expression.func.id != "mechanism" or expression.args:
        raise ProofProgramError("program must call mechanism with keyword arguments")
    values = _keywords(expression, "mechanism", {"target", "steps"})
    if "steps" not in values:
        raise ProofProgramError("mechanism requires steps")
    steps_node = values["steps"]
    if not isinstance(steps_node, (ast.List, ast.Tuple)) or not steps_node.elts:
        raise ProofProgramError("steps must be a non-empty list")
    if len(steps_node.elts) > 64:
        raise ProofProgramError("program exceeds the 64-step safety limit")
    return PythonElectronProgram(
        target_smiles=(
            _string(values["target"], "target") if "target" in values else ""
        ),
        steps=tuple(
            _parse_step(item, index) for index, item in enumerate(steps_node.elts)
        ),
    )


def execute_python_program(
    program_or_text: PythonElectronProgram | str,
    *,
    target_smiles: str = "",
) -> PythonProgramExecutionResult:
    """Replay arrows and derive the endpoint with the existing executor."""

    try:
        program = (
            parse_python_program(program_or_text)
            if isinstance(program_or_text, str)
            else program_or_text
        )
        if target_smiles and program.target_smiles and target_smiles != program.target_smiles:
            raise ProofProgramError("model-authored target does not match bound target")
        bound_target = target_smiles or program.target_smiles
        if not bound_target:
            raise ProofProgramError("executor requires a bound target_smiles")
        trace = ElectronFlowTrace(bound_target)
        current_state = bound_target
        for index, item in enumerate(program.steps):
            augmented = ".".join((current_state, *item.imports))
            result = verify_electron_step(augmented, item.moves)
            if not result.get("ok"):
                raise ProofProgramError(
                    f"electron step {index} failed: {result.get('code')} "
                    f"{result.get('message', '')}".strip()
                )
            next_state = str(result.get("state_smiles") or "")
            trace.append(
                state_before=current_state,
                state_after=next_state,
                moves=item.moves,
                imports=item.imports,
            )
            current_state = next_state
        compiled = compile_trace_to_proof(
            trace, declared_moves_already_verified=True
        )
        return PythonProgramExecutionResult(
            True,
            precursor_smiles=compiled.precursor_smiles,
            compiled_proof=compiled.proof,
            flow_trace=trace,
        )
    except (ProofProgramError, ValueError, KeyError, TypeError) as exc:
        return PythonProgramExecutionResult(
            False,
            diagnostics=[
                {"code": "PYTHON_PROGRAM_EXECUTION_FAILED", "message": str(exc)}
            ],
        )
