#!/usr/bin/env python3
"""Independently replay and validate mech-USPTO inverse Tool-SFT rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mechet.agent_env import AgentEnvConfig
from mechet.proof_program import sides_equal
from mechet.trace_agent_env import TraceOwnedAgentEnv
from train_tool_sft import validate_conversation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_result(name: str, expected: dict, observed: dict) -> None:
    if bool(expected.get("ok")) != bool(observed.get("ok")):
        raise ValueError(f"REPLAY_OK_MISMATCH:{name}")
    if "state_smiles" in expected and not sides_equal(
        str(expected.get("state_smiles") or ""),
        str(observed.get("state_smiles") or ""),
        ignore_maps=False,
    ):
        raise ValueError(f"REPLAY_STATE_MISMATCH:{name}")
    for key in (
        "endpoint_exact",
        "trace_digest",
        "move_sequence_digest",
        "compiled_proof",
        "n_trace_transitions",
        "endpoint_source",
    ):
        if key in expected and expected.get(key) != observed.get(key):
            raise ValueError(f"REPLAY_RESULT_MISMATCH:{name}:{key}")


def validate_row(row: dict) -> dict[str, int]:
    counts = validate_conversation(row, require_trace_owned=True)
    messages = list(row.get("messages") or [])
    calls = [
        call
        for message in messages
        for call in message.get("tool_calls") or []
    ]
    metadata = dict(row.get("metadata") or {})
    observation_mode = str(metadata.get("observation_mode") or "full_state")
    if observation_mode.endswith("_v1"):
        observation_mode = observation_mode[:-3]
    env = TraceOwnedAgentEnv(
        config=AgentEnvConfig(
            max_tool_calls=max(12, len(calls) + 2),
            observation_mode=observation_mode,
        )
    )
    env.reset(
        target_smiles=str(row.get("target_smiles") or ""),
        expected_precursor=str(row.get("expected_precursor") or ""),
    )
    replayed: dict[str, tuple[str, dict]] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            call_id = str(call["id"])
            function = dict(call["function"])
            name = str(function["name"])
            arguments = dict(function.get("arguments") or {})
            if name == "inspect_state":
                result = json.loads(env.inspect_state())
            elif name == "import_fragment":
                result = json.loads(env.import_fragment(**arguments))
            elif name == "apply_coupled_electron_moves":
                moves = list(arguments.get("moves") or [])
                for move in moves:
                    if set(move) != {"source", "sink", "electrons"}:
                        raise ValueError("MOVE_SCHEMA_EXTRA_FIELDS")
                    if set(move["source"]) != {"kind", "atoms"}:
                        raise ValueError("MOVE_SOURCE_SCHEMA_EXTRA_FIELDS")
                    if set(move["sink"]) != {"kind", "atoms"}:
                        raise ValueError("MOVE_SINK_SCHEMA_EXTRA_FIELDS")
                result = json.loads(
                    env.apply_coupled_electron_moves(
                        json.dumps(moves, ensure_ascii=False)
                    )
                )
            elif name == "finish_trace":
                result = json.loads(env.finish_trace())
            else:
                raise ValueError(f"UNEXPECTED_TOOL:{name}")
            replayed[call_id] = (name, result)
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in replayed:
                raise ValueError(f"TOOL_RESULT_WITHOUT_REPLAY:{call_id}")
            name, result = replayed.pop(call_id)
            expected = json.loads(str(message.get("content") or "{}"))
            _compare_result(name, expected, result)
    if replayed:
        raise ValueError("REPLAY_RESULTS_NOT_CONSUMED")
    terminal = env.final_result
    if not terminal.get("formal_execute") or not terminal.get("endpoint_exact"):
        raise ValueError("REPLAY_FINISH_NOT_EXACT")
    if terminal.get("compiled_proof") != metadata.get("compiled_proof"):
        raise ValueError("METADATA_COMPILED_PROOF_MISMATCH")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = replayed = tool_calls = 0
    ids: set[str] = set()
    failures: list[dict] = []
    with args.input.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            identifier = str(row.get("id") or "")
            if not identifier or identifier in ids:
                failures.append(
                    {"source_row": index, "id": identifier, "error": "DUPLICATE_ID"}
                )
                continue
            ids.add(identifier)
            try:
                counts = validate_row(row)
                tool_calls += counts["tool_calls"]
                replayed += 1
            except Exception as exc:
                failures.append(
                    {"source_row": index, "id": identifier, "error": str(exc)}
                )
    report = {
        "artifact_type": "inverse_tool_sft_replay_validation",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "rows": rows,
        "unique_ids": len(ids),
        "replayed": replayed,
        "tool_calls": tool_calls,
        "failures": len(failures),
        "failure_examples": failures[:20],
        "passed": not failures and replayed == rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
