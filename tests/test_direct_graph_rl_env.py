from mechet.direct_graph_rl_env import DirectGraphElectronEnv
from mechet.graph_fragment_actions import (
    allocate_reactive_fragment_maps,
    decompose_reactive_fragment,
)


def test_map_free_fragment_program_receives_executor_private_maps():
    program = decompose_reactive_fragment(
        "[O-:7][CH3:8]",
        participating_maps=(7,),
        role="NUCLEOPHILE",
    )
    mapped, assigned = allocate_reactive_fragment_maps(program, first_map=20)
    assert assigned == (20, 21)
    assert ":20" in mapped and ":21" in mapped


def test_direct_graph_environment_imports_without_candidate_inventory():
    env = DirectGraphElectronEnv(max_steps=4)
    initial = env.reset(target="[CH3:1][Br:2]")
    program = decompose_reactive_fragment(
        "[Na+:3]", participating_maps=(), role="ENVIRONMENT"
    )
    transition = env.step(
        {"kind": "IMPORT_ENV", "program": program.to_dict()}
    )
    assert initial.history.events == ()
    assert transition.accepted
    assert transition.next_observation.history.family_path == ("IMPORT_ENV",)
    assert "sources" not in transition.result
    assert "sinks" not in transition.result
    assert ":3" in transition.next_observation.current
