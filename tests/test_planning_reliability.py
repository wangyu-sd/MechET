from datetime import datetime, timedelta, timezone
import json
import sys

from mechet.planning_reliability import edge_audit, summarize_planning_graph
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram, format_proof_output


def proof():
    return format_proof_output(ProofProgram(
        target_smiles="[CH3:1][OH:2]",
        roots={"s0": ["[Br-:3]"]},
        precursor_state_id="s1",
        edges=[ProofEdge(
            "s0", "s1", bonds=[(1, 2, -1), (1, 3, +1)],
            lone_pairs=[(2, +2), (3, -2)],
            charges=[ChargeAction(2, 0, -1), ChargeAction(3, -1, 0)],
        )],
    ))


class Molecule:
    def __init__(self, smiles):
        self.smiles = smiles


class Reaction:
    def __init__(self, proof_text, reactants=None, target="CO", metadata=None):
        self.product = Molecule(target)
        self.reactants = [Molecule(s) for s in (
            reactants or ("CBr", "[OH-]"))]
        self.metadata = {"proof": proof_text, **(metadata or {})}


class Node:
    def __init__(self, *, reaction=None, mol=None, expanded=False, seconds=0, calls=0):
        if reaction is not None:
            self.reaction = reaction
        if mol is not None:
            self.mol = Molecule(mol)
        self.is_expanded = expanded
        self.creation_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
        self.data = {"num_calls_rxn_model": calls}


class Graph:
    def __init__(self, root, children):
        self.root_node = root
        self.children = children

    def nodes(self):
        return list(self.children)

    def successors(self, node):
        return self.children.get(node, [])


def test_proof_certification_checks_both_product_and_structural_precursors():
    assert edge_audit(Reaction(proof()))["status"] == "certified"
    wrong = edge_audit(Reaction(proof(), reactants=("CCl", "[OH-]")))
    assert wrong["executable"] is True
    assert wrong["certified"] is False
    assert wrong["reason"] == "precursor_mismatch"
    assert edge_audit(Reaction("not a proof"))["status"] == "failed"
    assert edge_audit(Reaction(""))["status"] == "unverified"


def test_failed_admitted_edge_counts_downstream_waste_without_shared_path_double_count():
    root = Node(mol="CO", expanded=True)
    good = Node(reaction=Reaction(proof()), seconds=2, calls=1)
    bad = Node(reaction=Reaction("not a proof"), seconds=3, calls=1)
    leaf_good = Node(mol="CBr", seconds=2, calls=1)
    leaf_other = Node(mol="[OH-]", seconds=2, calls=1)
    leaf_wasted = Node(mol="CCl", expanded=True, seconds=4, calls=2)
    graph = Graph(root, {
        root: [good, bad], good: [leaf_good, leaf_other],
        bad: [leaf_wasted], leaf_good: [], leaf_other: [], leaf_wasted: [],
    })
    report = summarize_planning_graph(
        graph, [{root, good, leaf_good, leaf_other}],
        model_calls=3, wall_seconds=5.0,
    )
    assert report["hallucinated_admitted_edge_rate"] == 0.5
    assert report["n_certified_routes"] == 1
    assert report["n_wasted_expansions_below_failed_edges"] == 1
    assert report["wasted_expansion_ratio"] == 0.5
    assert report["calls_to_first_solution"] == 1
    assert report["seconds_to_first_solution"] == 2.0


def test_missing_proof_never_becomes_a_zero_hallucination_claim():
    root = Node(mol="CO", expanded=True)
    unknown = Node(reaction=Reaction(""))
    leaf = Node(mol="CBr", expanded=True)
    graph = Graph(root, {root: [unknown], unknown: [leaf], leaf: []})
    report = summarize_planning_graph(
        graph, [{root, unknown, leaf}], model_calls=1, wall_seconds=1.0,
    )
    assert report["n_unverified_edges"] == 1
    assert report["hallucinated_admitted_edge_rate"] is None
    assert report["wasted_expansion_ratio"] is None
    assert report["n_certified_routes"] == 0
    assert report["n_unverified_routes"] == 1
    assert report["certified_route_rate"] is None


def test_pool_preserves_source_proposal_denominator_before_search_filter(tmp_path):
    from mechet.syntheseus_adapter import MechETCandidatePool

    source = tmp_path / "pool.jsonl"
    source.write_text(json.dumps({
        "target": "CO", "hypotheses": [
            {"precursor": "CBr.[OH-]", "proof": proof(), "execute_ok": True},
            {"precursor": "CCl", "proof": "invalid", "execute_ok": False},
        ],
    }) + "\n")
    pool = MechETCandidatePool.from_jsonl(source)
    assert pool.n_candidates == 1
    assert pool.source_audit["n_source_proposals"] == 2
    assert pool.source_audit["n_source_reported_execution_fail"] == 1
    assert pool.source_audit["source_reported_execution_failure_rate"] == 0.5

    source.write_text(json.dumps({
        "target": "CO", "hypotheses": [{"precursor": "CBr.[OH-]", "proof": proof()}],
    }) + "\n")
    missing = MechETCandidatePool.from_jsonl(source)
    assert missing.source_audit["n_source_execution_label_missing"] == 1
    assert missing.source_audit["source_reported_execution_failure_rate"] is None


def test_natural_language_trace_certificate_replays_without_reference_precursor():
    from rdkit import Chem
    from mechet.endpoints import split_precursor_endpoints
    from mechet.natural_language_electron_flow import render_event_arguments
    from scripts.run_natural_language_value_search import (
        Action, Node as ElectronNode, execute, mapped_atom_numbers,
        normal_smiles, product_only_private_state, visible,
    )

    target = "[O:1]=[C:2]([OH:3])[CH3:4].[O-:5][CH2:6][CH3:7]"
    state = product_only_private_state(target)
    mol = Chem.MolFromSmiles(state)
    carbonyl_o = carbonyl_c = anion_o = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O":
            continue
        if atom.GetFormalCharge() == -1:
            anion_o = atom.GetAtomMapNum()
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetSymbol() == "C":
                carbonyl_o = atom.GetAtomMapNum()
                carbonyl_c = bond.GetOtherAtom(atom).GetAtomMapNum()
    assert carbonyl_o and carbonyl_c and anion_o
    moves = [
        {"source": {"kind": "BOND", "atoms": [carbonyl_o, carbonyl_c]},
         "sink": {"kind": "ATOM", "atoms": [carbonyl_o]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [anion_o]},
         "sink": {"kind": "BOND", "atoms": [carbonyl_c, anion_o]}, "electrons": 2},
    ]
    event = render_event_arguments(state, moves)
    node = ElectronNode(
        target=visible(state), state=state,
        next_map=max(mapped_atom_numbers(state)) + 1, visited={visible(state)},
    )
    child, error = execute(node, Action("apply_electron_flow", event, "", 0.0, 1), max_imports=32)
    assert child is not None, error
    final, error = execute(
        child, Action("finish_trace", {}, "", 0.0, 1),
        max_imports=32, reject_target_retained_finish=True,
    )
    assert final is not None and final.terminal, error
    structural = split_precursor_endpoints(final.state, state).structural
    certificate = {
        "protocol": "mechet_nl_reverse_et_v2", "observation": "current_state",
        "product": visible(state), "max_imports": 32,
        "reject_target_retained_finish": True,
        "actions": [
            {"name": "apply_electron_flow", "arguments": event},
            {"name": "finish_trace", "arguments": {}},
        ],
    }
    reaction = Reaction(
        "", target=normal_smiles(target), reactants=structural.split("."),
        metadata={"mechet": {"trace_certificate": certificate}},
    )
    assert edge_audit(reaction)["status"] == "certified"
    reaction.reactants = [Molecule("CCl")]
    assert edge_audit(reaction)["reason"] == "precursor_mismatch"


def test_real_syntheseus_retrostar_route_certificate_when_extra_installed():
    import pytest

    pytest.importorskip("syntheseus")
    from syntheseus import Molecule as SyntheseusMolecule
    from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
    from syntheseus.search.analysis.route_extraction import iter_routes_time_order
    from syntheseus.search.mol_inventory import SmilesListInventory
    from syntheseus.search.node_evaluation.common import (
        ConstantNodeEvaluator, ReactionModelLogProbCost,
    )

    from mechet.syntheseus_adapter import (
        MechETBackwardReactionModel, MechETCandidatePool, PoolCandidate,
    )

    pool = MechETCandidatePool([PoolCandidate(
        target="CO", precursor="CBr.[OH-]", score=-0.1, proof=proof(),
    )])
    model = MechETBackwardReactionModel(pool, default_num_results=1, use_cache=True)
    search = RetroStarSearch(
        reaction_model=model,
        mol_inventory=SmilesListInventory(smiles_list=["CBr", "[OH-]"]),
        limit_iterations=10,
        limit_reaction_model_calls=10,
        time_limit_s=10,
        value_function=ConstantNodeEvaluator(0.0),
        and_node_cost_fn=ReactionModelLogProbCost(),
    )
    graph, _ = search.run_from_mol(SyntheseusMolecule("CO"))
    routes = list(iter_routes_time_order(graph, max_routes=5))
    report = summarize_planning_graph(
        graph, routes, model_calls=model.num_calls(), wall_seconds=1.0,
    )
    assert report["solved"] is True
    assert report["n_certified_routes"] >= 1
    assert report["n_unverified_edges"] == 0
    assert report["hallucinated_admitted_edge_rate"] == 0.0


def test_syntheseus_cli_records_frozen_budget_and_reliability_when_extra_installed(
    tmp_path, monkeypatch,
):
    import pytest

    pytest.importorskip("syntheseus")
    from scripts.run_syntheseus_search import main

    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({
        "target": "CO", "hypotheses": [{
            "precursor": "CBr.[OH-]", "score": -0.1,
            "execute_ok": True, "proof": proof(),
        }],
    }) + "\n")
    targets = tmp_path / "targets.smi"
    targets.write_text("CO\n")
    inventory = tmp_path / "inventory.smi"
    inventory.write_text("CBr\n[OH-]\n")
    output = tmp_path / "planning"
    monkeypatch.setattr(sys, "argv", [
        "run_syntheseus_search.py", "--candidate-pool", str(pool),
        "--targets", str(targets), "--inventory", str(inventory),
        "--output-dir", str(output), "--num-results", "1",
        "--max-routes", "5", "--iterations", "10",
        "--reaction-model-calls", "10", "--time-limit-s", "10",
    ])
    assert main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_targets"] == 1
    assert summary["certified_solved_observed"] == 1
    assert summary["hallucinated_admitted_edge_rate"] == 0.0
    assert summary["candidate_pool_audit"]["n_source_proposals"] == 1
    assert summary["candidate_pool_audit"]["n_retained_candidates"] == 1
    assert summary["budgets"]["num_results"] == 1
    assert summary["candidate_pool_sha256"]
