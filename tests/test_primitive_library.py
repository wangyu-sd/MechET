import json
from pathlib import Path

from mechet.primitive_library import PrimitiveLibrary

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "knowledge" / "primitives" / "core_polar_primitives.yaml"
REGISTRY = REPO / "knowledge" / "source_registry.yaml"


def load_library():
    return PrimitiveLibrary.load(LIBRARY, source_registry=REGISTRY)


def test_primitive_library_validates_and_has_provenance():
    library = load_library()
    manifest = library.manifest()
    assert manifest["n_primitives"] >= 8
    assert "nucleophilic_substitution_sp3" in manifest["primitive_ids"]
    primitive = library.by_id["nucleophilic_substitution_sp3"]
    assert primitive.sources
    assert all(item.source_id for item in primitive.sources)


def test_retrieves_atom_bound_sn2_moves():
    library = load_library()
    matches = library.retrieve("[CH3:1][Br:2].[OH-:3]", top_k=10)
    sn2 = next(item for item in matches if item.primitive_id == "nucleophilic_substitution_sp3")
    assert sn2.role_bindings == {"electrophile": 1, "leaving_group": 2, "nucleophile": 3}
    assert len(sn2.moves) == 2
    assert sn2.moves[0]["source"]["kind"] == "LP"


def test_move_support_is_soft_and_exact():
    library = load_library()
    moves = [
        {"source": {"kind": "LP", "atoms": [3]}, "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
        {"source": {"kind": "BOND", "atoms": [1, 2]}, "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
    ]
    evidence = library.support_moves("[CH3:1][Br:2].[OH-:3]", moves)
    assert evidence["supported"] and evidence["soft_evidence_only"]
    assert "nucleophilic_substitution_sp3" in evidence["primitive_ids"]


def test_reaction_annotation_recovers_primitive_bond_deltas():
    evidence = load_library().annotate_reaction("[CH3:1][Br:2].[OH-:3]", "[CH3:1][OH:3].[Br-:2]")
    assert evidence["best_support"] == 1.0
    assert "nucleophilic_substitution_sp3" in evidence["complete_primitive_ids"]


def test_unmatched_state_is_not_rejected():
    library = load_library()
    assert library.retrieve("[CH4:1]", top_k=5) == []
    support = library.support_moves("[CH4:1]", [{"source": {"kind": "LP", "atoms": [1]}, "sink": {"kind": "ATOM", "atoms": [1]}}])
    assert not support["supported"]
    assert support["soft_evidence_only"]


def test_rendered_context_marks_guidance_as_soft():
    text = load_library().render_context("[CH3:1][Br:2].[OH-:3]", top_k=2)
    assert "nucleophilic_substitution_sp3" in text
    assert "do not override the executor" in text
