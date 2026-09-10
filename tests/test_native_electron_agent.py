import importlib.util
from pathlib import Path

from mechet.native_electron_agent import NativeElectronAgent


_SPEC = importlib.util.spec_from_file_location(
    "infer_native_electron_agent",
    Path(__file__).parents[1] / "scripts" / "infer_native_electron_agent.py",
)
_INFERENCE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_INFERENCE)


def call(agent, name, **arguments):
    if name != "inspect":
        arguments["state_id"] = agent.state_id
    return agent.call(name, arguments)


def substitution_moves(bromine="p3"):
    return [
        {
            "source": {"kind": "bond", "positions": ["p1", "p2"]},
            "sink": {"kind": "atom", "positions": ["p2"]},
        },
        {
            "source": {"kind": "lp", "positions": [bromine]},
            "sink": {"kind": "bond", "positions": ["p1", bromine]},
        },
    ]


def test_inspect_covers_all_atoms_and_marks_one_position():
    agent = NativeElectronAgent("C" * 30)
    first = agent.call("inspect", {})
    second = agent.call("inspect", {"offset": 24})
    assert first["next_offset"] == 24
    assert len(first["positions"]) == 24
    assert len(second["positions"]) == 6
    assert agent.call("inspect", {"position": "p27"})[
        "marked_component_smiles"
    ].count(":1]") == 1


def test_import_execute_finish_and_compile_endpoint():
    agent = NativeElectronAgent("CO")
    imported = call(
        agent,
        "import_fragment",
        smiles="[Br-]",
        role="nucleophile",
        reason="displacement participant",
    )
    assert imported["ok"] and imported["new_positions"] == ["p3"]
    inspected = agent.call("inspect", {})
    bromine = next(row for row in inspected["positions"] if row["position"] == "p3")
    assert bromine["lone_pair_source"]
    assert call(
        agent,
        "apply_step",
        moves=substitution_moves(),
        reason="break C-O while forming C-Br",
    )["ok"]
    result = call(agent, "finish", reason="nontrivial endpoint")
    assert result["ok"] and result["finished"]
    assert "CBr" in result["precursor_smiles"]
    assert "[OH-]" in result["precursor_smiles"]


def test_failed_step_is_atomic_and_reports_actionable_error():
    agent = NativeElectronAgent("CO")
    before = agent.observation()
    result = call(
        agent,
        "apply_step",
        moves=substitution_moves(),
        reason="references absent p3",
    )
    assert not result["ok"] and "UNKNOWN_POSITION p3" in result["error"]
    assert agent.observation() == before


def test_nested_positions_are_rejected_without_mutation():
    agent = NativeElectronAgent("CO")
    moves = [
        {
            "source": {"kind": "bond", "positions": [["p1", "p2"]]},
            "sink": {"kind": "atom", "positions": ["p2"]},
        }
    ]
    result = call(agent, "apply_step", moves=moves, reason="bad JSON shape")
    assert not result["ok"] and "flat string list" in result["error"]
    assert not agent.trace.transitions


def test_undo_drops_imports_and_never_reuses_position_ids():
    agent = NativeElectronAgent("CO")
    call(agent, "import_fragment", smiles="[Br-]", role="nucleophile", reason="one")
    assert call(agent, "undo_step")["ok"]
    second = call(
        agent,
        "import_fragment",
        smiles="[Br-]",
        role="nucleophile",
        reason="two",
    )
    assert second["new_positions"] == ["p4"]
    moves = substitution_moves("p4")
    assert call(agent, "apply_step", moves=moves, reason="substitution")["ok"]
    assert call(agent, "undo_step")["ok"]
    assert not agent.trace.transitions
    assert agent.call("inspect", {})["total_positions"] == 2


def test_stale_state_and_empty_finish_are_rejected():
    agent = NativeElectronAgent("CO")
    assert not agent.call("undo_step", {"state_id": "s99"})["ok"]
    assert not call(agent, "finish", reason="empty")["ok"]


def test_unknown_tool_role_and_non_array_moves_are_rejected_atomically():
    agent = NativeElectronAgent("CO")
    assert "UNKNOWN_TOOL" in agent.call("invent", {})["error"]
    bad_role = call(
        agent,
        "import_fragment",
        smiles="[Br-]",
        role="spectator",
        reason="invalid role",
    )
    assert bad_role["error"] == "INVALID_IMPORT_ROLE"
    assert agent.call("inspect", {})["total_positions"] == 2
    bad_moves = call(agent, "apply_step", moves="not-an-array", reason="invalid")
    assert bad_moves["error"] == "moves must be an array"


def test_native_tool_parser_and_reference_separation(tmp_path):
    value = _INFERENCE.parse_tool_call(
        '<think>brief</think><tool_call>{"name":"inspect","arguments":{}}</tool_call>'
    )
    assert value == {"name": "inspect", "arguments": {}}
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id":"case-1","target_smiles":"CO","expected_precursor":"CBr.[OH-]"}\n'
    )
    actor_cases, references = _INFERENCE.load_cases(cases, 1)
    assert actor_cases == [{"id": "case-1", "target_smiles": "CO"}]
    assert "expected_precursor" not in actor_cases[0]
    assert references == {"case-1": "CBr.[OH-]"}
