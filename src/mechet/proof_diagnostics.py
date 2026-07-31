"""Structured failure certificates and deterministic local repair for MECH_PROOF v1."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import re
from typing import Any

from mechet.proof_program import (
    ProofProgramError,
    execute_proof,
    format_proof_output,
    parse_proof_program,
)


_EDGE_RE = re.compile(r"(?P<src>[^\s:]+)->(?P<dst>[^\s:]+)")
_MISMATCH_RE = re.compile(
    r"(?P<kind>BOND|LP) execution mismatch on "
    r"(?P<src>[^\s:]+)->(?P<dst>[^\s:]+): "
    r"written=(?P<written>\[[^\n]*\]) "
    r"derived=(?P<derived>\[[^\n]*\])"
)


@dataclass(frozen=True)
class FailureCertificate:
    code: str
    stage: str
    message: str
    edge_src: str = ""
    edge_dst: str = ""
    written: tuple[tuple[int, ...], ...] = ()
    expected: tuple[tuple[int, ...], ...] = ()
    repairable: bool = False
    repair_lines: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def edge(self) -> str:
        return (
            f"{self.edge_src}->{self.edge_dst}"
            if self.edge_src and self.edge_dst
            else ""
        )


@dataclass(frozen=True)
class ProofRepairResult:
    changed: bool
    repaired_text: str
    execute_ok: bool
    certificate: FailureCertificate | None = None
    diagnostics: tuple[dict[str, str], ...] = ()


def _literal_tuples(text: str) -> tuple[tuple[int, ...], ...]:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ()
    if not isinstance(value, list):
        return ()
    out: list[tuple[int, ...]] = []
    for item in value:
        if not isinstance(item, (tuple, list)):
            return ()
        try:
            out.append(tuple(int(part) for part in item))
        except (TypeError, ValueError):
            return ()
    return tuple(out)


def _repair_lines(
    kind: str,
    expected: tuple[tuple[int, ...], ...],
) -> tuple[str, ...]:
    lines: list[str] = []
    if kind == "LP":
        for atom_map, delta in expected:
            sign = f"+{delta}" if delta > 0 else str(delta)
            lines.append(f"LP {atom_map} {sign}")
    elif kind == "BOND":
        for i, j, delta in expected:
            sign = f"+{delta}" if delta > 0 else str(delta)
            lines.append(f"BOND {i} {j} {sign}")
    return tuple(lines)


def diagnose_proof(text: str) -> FailureCertificate | None:
    """Return a structured certificate for the first deterministic failure."""
    result = execute_proof(text)
    if result.ok:
        return None
    message = (
        result.diagnostics[0].get("message", "proof execution failed")
        if result.diagnostics
        else "proof execution failed"
    )

    mismatch = _MISMATCH_RE.search(message)
    if mismatch:
        kind = mismatch.group("kind")
        written = _literal_tuples(mismatch.group("written"))
        expected = _literal_tuples(mismatch.group("derived"))
        # LP declarations certify a transition but do not mutate the graph.
        # Replacing them is therefore semantics-preserving. BOND operations
        # change the state and are returned only as model repair feedback.
        repairable = kind == "LP" and bool(expected)
        return FailureCertificate(
            code=f"{kind}_EXECUTION_MISMATCH",
            stage="edge_execution",
            message=message,
            edge_src=mismatch.group("src"),
            edge_dst=mismatch.group("dst"),
            written=written,
            expected=expected,
            repairable=repairable,
            repair_lines=_repair_lines(kind, expected),
        )

    edge_match = _EDGE_RE.search(message)
    src = edge_match.group("src") if edge_match else ""
    dst = edge_match.group("dst") if edge_match else ""
    lowered = message.lower()
    if "electron conservation failed" in lowered:
        code, stage = "ELECTRON_NOT_CONSERVED", "conservation"
    elif "charge precondition failed" in lowered:
        code, stage = "CHARGE_PRECONDITION_FAILED", "edge_execution"
    elif "charge execution mismatch" in lowered:
        code, stage = "CHARGE_EXECUTION_MISMATCH", "edge_execution"
    elif "sanitization failed" in lowered:
        code, stage = "CHEMICAL_STATE_INVALID", "edge_execution"
    elif "dag join mismatch" in lowered:
        code, stage = "DAG_JOIN_MISMATCH", "graph_execution"
    elif "unreachable proof edges" in lowered:
        code, stage = "UNREACHABLE_EDGE", "graph_execution"
    elif "precursor state" in lowered and "not derived" in lowered:
        code, stage = "PRECURSOR_NOT_DERIVED", "graph_execution"
    elif "expected mech_proof" in lowered or "unknown proof line" in lowered:
        code, stage = "PROOF_PARSE_FAILED", "parse"
    elif "missing map" in lowered or "duplicate atom map" in lowered:
        code, stage = "ATOM_MAP_ERROR", "edge_execution"
    else:
        code, stage = "PROOF_EXECUTION_FAILED", "execution"
    return FailureCertificate(
        code=code,
        stage=stage,
        message=message,
        edge_src=src,
        edge_dst=dst,
    )


def repair_proof_once(
    text: str,
    certificate: FailureCertificate | None = None,
) -> ProofRepairResult:
    """Apply one semantics-preserving deterministic repair when available."""
    certificate = certificate or diagnose_proof(text)
    if certificate is None:
        return ProofRepairResult(False, text, True, None, ())
    if (
        not certificate.repairable
        or certificate.code != "LP_EXECUTION_MISMATCH"
    ):
        return ProofRepairResult(
            False,
            text,
            False,
            certificate,
            ({"code": certificate.code, "message": certificate.message},),
        )
    try:
        program = parse_proof_program(text)
    except ProofProgramError as exc:
        return ProofRepairResult(
            False,
            text,
            False,
            certificate,
            ({"code": "PROOF_PARSE_FAILED", "message": str(exc)},),
        )

    target_edge = next(
        (
            edge
            for edge in program.edges
            if edge.src == certificate.edge_src
            and edge.dst == certificate.edge_dst
        ),
        None,
    )
    if target_edge is None:
        return ProofRepairResult(
            False,
            text,
            False,
            certificate,
            (
                {
                    "code": "REPAIR_EDGE_NOT_FOUND",
                    "message": certificate.edge,
                },
            ),
        )
    target_edge.lone_pairs = [
        (int(atom_map), int(delta))
        for atom_map, delta in certificate.expected
    ]
    repaired_text = format_proof_output(program)
    execution = execute_proof(repaired_text)
    return ProofRepairResult(
        changed=True,
        repaired_text=repaired_text,
        execute_ok=execution.ok,
        certificate=certificate,
        diagnostics=tuple(execution.diagnostics),
    )


def format_repair_feedback(certificate: FailureCertificate) -> str:
    """Create a compact verifier message for model-guided local repair."""
    lines = [f"FAIL {certificate.code}"]
    if certificate.edge:
        lines.append(f"EDGE {certificate.edge}")
    lines.append(certificate.message)
    if certificate.repair_lines:
        lines.append("EXPECTED")
        lines.extend(f"  {line}" for line in certificate.repair_lines)
    return "\n".join(lines)
