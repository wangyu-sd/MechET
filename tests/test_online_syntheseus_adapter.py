import json
from types import SimpleNamespace

import pytest

from mechet.online_syntheseus_adapter import _seed


def planning_args(**overrides):
    values = dict(
        product_only_remap=True, matched_v2=True,
        reject_target_retained_finish=True,
        max_decisions=40, max_imports=32, branching=1,
        early_beam=1, late_beam=1, value_weight=0.0,
        compact_history=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_online_plan_seed_is_query_order_independent_and_map_free():
    assert _seed(17, "CO", 0) == _seed(17, "[CH3:7][OH:9]", 0)
    assert _seed(17, "CO", 0) != _seed(17, "CO", 1)


def test_online_bridge_requires_frozen_product_only_contract():
    pytest.importorskip("syntheseus")
    from mechet.online_syntheseus_adapter import OnlineMechETBackwardReactionModel

    runtime = SimpleNamespace(torch=SimpleNamespace(manual_seed=lambda _: None))
    with pytest.raises(ValueError, match="product-only private remapping"):
        OnlineMechETBackwardReactionModel(runtime, planning_args(product_only_remap=False))
    with pytest.raises(ValueError, match="SFT-aligned"):
        OnlineMechETBackwardReactionModel(runtime, planning_args(matched_v2=False))
    with pytest.raises(ValueError, match="single policy trajectory"):
        OnlineMechETBackwardReactionModel(runtime, planning_args(branching=2))


def test_online_bridge_queries_arbitrary_planner_molecules_without_gold(monkeypatch):
    syntheseus = pytest.importorskip("syntheseus")
    from mechet.online_syntheseus_adapter import OnlineMechETBackwardReactionModel
    from scripts import run_natural_language_value_search as search

    queries = []
    seeds = []

    def fake_search(_runtime, product, args):
        queries.append((product, args.planning_sample))
        top = search.Node(
            target="CO", state="[CH3:1][Br:3].[OH-:2]",
            next_map=4, terminal=True,
            logprob=-1.0, tokens=10,
            actions=[
                {"name": "apply_electron_flow", "arguments": {}, "result": {}},
                {"name": "finish_trace", "arguments": {}, "result": {}},
            ],
        )
        return {
            "target": "CO", "target_mapped": "[CH3:1][OH:2]",
            "top": top, "attempts": [{"accepted": True}],
        }

    monkeypatch.setattr(search, "search_unlabeled", fake_search)
    runtime = SimpleNamespace(torch=SimpleNamespace(manual_seed=seeds.append))
    model = OnlineMechETBackwardReactionModel(
        runtime, planning_args(), seed=17, max_candidates=3,
    )
    product = syntheseus.Molecule("CO")
    [reactions] = model([product], num_results=2)
    assert queries == [("CO", True), ("CO", True)]
    assert len(seeds) == 2 and seeds[0] != seeds[1]
    assert model.episode_count == 2
    assert model.policy_decision_count == 2
    assert model.rejected_decision_count == 0
    assert model.nonterminal_count == 0
    assert model.unadmitted_episode_count == 0
    assert len(reactions) == 1  # equivalent precursor episodes are deduplicated
    assert reactions[0].metadata["mechet"]["trace_certificate"]["product"] == "CO"
    with pytest.raises(ValueError, match="exceeds frozen maximum"):
        model([product], num_results=4)


def test_online_cli_dry_run_checks_frozen_adapter_without_loading_model(tmp_path, capsys):
    from scripts.run_syntheseus_online import main

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"frozen test weight")
    (adapter / "adapter_manifest.json").write_text(json.dumps({
        "environment_revision": "natural_language_electron_event_history_v2",
        "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
        "base_model": "Qwen/Qwen3-0.6B",
        "base_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    }))
    targets = tmp_path / "targets.smi"
    targets.write_text("CO\n")
    inventory = tmp_path / "inventory.smi"
    inventory.write_text("CBr\n[OH-]\n")
    common = [
        "--adapter", str(adapter), "--targets", str(targets),
        "--inventory", str(inventory), "--output-dir", str(tmp_path / "out"),
        "--dry-run",
    ]
    assert main(["--stage", "trajectory", *common]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_targets"] == 1
    assert report["reference_precursor_visible_to_policy"] is False
    assert report["adapter_sha256"]
    with pytest.raises(ValueError, match="environment_revision"):
        main(["--stage", "state", *common])


def test_online_bridge_real_syntheseus_route_replays_electron_certificate(
    tmp_path, monkeypatch,
):
    pytest.importorskip("syntheseus")
    from mechet.natural_language_electron_flow import render_event_arguments
    from mechet.online_syntheseus_adapter import OnlineMechETBackwardReactionModel
    from scripts import run_natural_language_value_search as search
    from scripts.run_syntheseus_search import run_planner

    def fake_search(_runtime, product, _args):
        assert product == "CO"  # planner requests the current molecule, not a GT row
        state = search.product_only_private_state(product)
        root = search.Node(
            target=search.visible(state), state=state, next_map=3,
            visited={search.visible(state)},
        )
        imported, error = search.execute(
            root, search.Action("import_fragments", {
                "fragments": [{"smiles": "[Br-]", "count": 1}],
            }, "", -1.0, 1), max_imports=32,
        )
        assert imported is not None, error
        moves = [
            {"source": {"kind": "BOND", "atoms": [1, 2]},
             "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
            {"source": {"kind": "LP", "atoms": [3]},
             "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
        ]
        event = render_event_arguments(imported.state, moves)
        moved, error = search.execute(
            imported, search.Action("apply_electron_flow", event, "", -1.0, 1),
            max_imports=32,
        )
        assert moved is not None, error
        finished, error = search.execute(
            moved, search.Action("finish_trace", {}, "", -1.0, 1),
            max_imports=32, reject_target_retained_finish=True,
        )
        assert finished is not None and finished.terminal, error
        return {
            "target": search.visible(state), "target_mapped": state,
            "top": finished, "attempts": [{"accepted": True}] * 3,
        }

    monkeypatch.setattr(search, "search_unlabeled", fake_search)
    runtime = SimpleNamespace(torch=SimpleNamespace(manual_seed=lambda _: None))
    model = OnlineMechETBackwardReactionModel(
        runtime, planning_args(), max_candidates=1,
    )
    result = run_planner(
        model=model, targets=["CO"], inventory=["CBr", "[OH-]"],
        args=SimpleNamespace(
            algorithm="retro_star", num_results=1, max_routes=5,
            reaction_model_calls=10, iterations=10, time_limit_s=10.0,
            output_dir=tmp_path,
        ),
        provenance={"candidate_provider": "online_test"},
    )
    assert result["solved"] == 1
    assert result["certified_solved_observed"] == 1
    assert result["hallucinated_admitted_edge_rate"] == 0.0
    assert result["online_generation"]["episode_admission_rate"] == 1.0


def test_online_planner_resets_episode_counters_for_each_target(tmp_path, monkeypatch):
    pytest.importorskip("syntheseus")
    from mechet.online_syntheseus_adapter import OnlineMechETBackwardReactionModel
    from scripts import run_natural_language_value_search as search
    from scripts.run_syntheseus_search import run_planner

    queried = []

    def fake_search(_runtime, product, _args):
        queried.append(product)
        state = search.product_only_private_state(product)
        target = search.visible(state)
        return {
            "target": target, "target_mapped": state,
            "top": search.Node(
                target=target, state=state, next_map=4, visited={target},
            ),
            "attempts": [{"accepted": False}],
        }

    monkeypatch.setattr(search, "search_unlabeled", fake_search)
    runtime = SimpleNamespace(torch=SimpleNamespace(manual_seed=lambda _: None))
    model = OnlineMechETBackwardReactionModel(
        runtime, planning_args(), max_candidates=1,
    )
    report = run_planner(
        model=model, targets=["CO", "CCO"], inventory=["N"],
        args=SimpleNamespace(
            algorithm="retro_star", num_results=1, max_routes=5,
            reaction_model_calls=10, iterations=10, time_limit_s=10.0,
            output_dir=tmp_path,
        ),
        provenance={"candidate_provider": "online_test"},
    )
    assert queried == ["CO", "CCO"]
    assert [item["online_generation"]["episodes"] for item in report["targets"]] == [1, 1]
    assert report["online_generation"]["episodes"] == 2
    assert report["online_generation"]["unadmitted_episodes"] == 2
