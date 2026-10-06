import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_reliable_independent_episodes as ev
from scripts.run_natural_language_value_search import Action, generation_sampling_policy
from mechet.natural_language_electron_flow import render_event_arguments
from mechet.endpoints import structural_exact


def _episode(index: int, target: str, *, precursor: str = "", score=None, event=False):
    return {
        "index": index, "seed": ev.episode_seed(17, target, index),
        "terminal": bool(precursor), "precursor": precursor,
        "mean_token_logprob": score, "has_electron_event": event,
    }


def test_independent_episode_seeds_and_gold_independent_nll_ranking():
    row = {"id": "r", "source_id": "r", "target_smiles": "CO",
           "structural_precursor": "CN"}
    episodes = [
        _episode(0, "CO", precursor="CC", score=-0.1, event=True),
        _episode(1, "CO", precursor="CN", score=-0.8, event=True),
        _episode(2, "CO", precursor="CC", score=-0.05, event=True),
        *[_episode(index, "CO") for index in range(3, 10)],
    ]
    scored = ev.score_reaction(row, {
        "source_id": "r", "target": "CO", "seed": 17, "episodes": episodes,
    })
    assert scored["terminal_episodes"] == 3
    assert scored["unique_terminal_candidates"] == 2
    assert not scored["generation_pass_at_1"]
    assert scored["generation_pass_at_5"]
    assert not scored["nll_ranked_top_1"]
    assert scored["nll_ranked_top_5"]
    assert scored["process_reliable_pass_at_5"]
    assert ev.episode_seed(17, "[CH3:71][OH:90]", 1) == ev.episode_seed(17, "CO", 1)
    episodes[1]["seed"] += 1
    with pytest.raises(ValueError, match="seed mismatch"):
        ev.score_reaction(row, {
            "source_id": "r", "target": "CO", "seed": 17, "episodes": episodes,
        })


def test_k_one_is_greedy_and_k_many_samples_independent_paths(monkeypatch):
    calls = []

    def fake_episode(_runtime, target, _args, *, index, seed, stochastic):
        calls.append((target, index, seed, stochastic))
        return _episode(index, target)

    monkeypatch.setattr(ev, "sample_episode", fake_episode)
    row = {"id": "r", "source_id": "r", "target_smiles": "CO",
           "expected_precursor": "SECRET_GOLD_NEVER_USED"}
    runtime = SimpleNamespace()
    actor = SimpleNamespace()
    greedy = ev.sample_reaction(runtime, row, actor, episodes=1, seed=17)
    assert calls == [("CO", 0, ev.episode_seed(17, "CO", 0), False)]
    assert "SECRET_GOLD_NEVER_USED" not in json.dumps(greedy)
    calls.clear()
    ev.sample_reaction(runtime, row, actor, episodes=5, seed=17)
    assert len(calls) == 5
    assert all(item[3] is True for item in calls)
    assert len({item[2] for item in calls}) == 5
    assert generation_sampling_policy(
        matched_v2=True, vnext_v2_prefix=False, candidates=1,
        planning_sample=False,
    ) == {"do_sample": False}
    assert generation_sampling_policy(
        matched_v2=True, vnext_v2_prefix=False, candidates=1,
        planning_sample=True,
    )["do_sample"] is True


def test_sample_episode_uses_fresh_product_only_runtime_path(monkeypatch):
    observed = {}

    def fake_search(_runtime, target, args):
        observed["target"] = target
        observed["sampling"] = args.planning_sample
        return {
            "target_mapped": "[CH3:1][OH:2]",
            "top": SimpleNamespace(
                terminal=True, state="[CH3:1][OH:2]", policy_score=-0.3,
                actions=[{"name": "apply_electron_flow", "arguments": {}}],
            ),
            "attempts": [], "rejected": {},
        }

    monkeypatch.setattr(ev, "search_unlabeled", fake_search)
    runtime = SimpleNamespace(torch=SimpleNamespace(
        manual_seed=lambda value: observed.update(seed=value)
    ))
    episode = ev.sample_episode(
        runtime, "[CH3:77][OH:42]", SimpleNamespace(product_only_remap=True),
        index=2, seed=123, stochastic=True,
    )
    assert observed == {
        "target": "[CH3:77][OH:42]", "sampling": True, "seed": 123,
    }
    assert episode["precursor"] == "CO"
    assert episode["has_electron_event"] is True
    assert episode["mean_token_logprob"] == -0.3


def test_one_episode_executes_real_import_electron_flow_and_finish():
    class FakeRuntime:
        torch = SimpleNamespace(manual_seed=lambda _seed: None)
        pointer_invalid_handles = 0

        def proposals(self, node, **_kwargs):
            depth = len(node.actions)
            if depth == 0:
                return [Action("import_fragments", {
                    "fragments": [{"smiles": "[Br-]", "count": 1}],
                }, "", -1.0, 1)]
            if depth == 1:
                moves = [
                    {"source": {"kind": "BOND", "atoms": [1, 2]},
                     "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
                    {"source": {"kind": "LP", "atoms": [3]},
                     "sink": {"kind": "BOND", "atoms": [1, 3]}, "electrons": 2},
                ]
                return [Action("apply_electron_flow", render_event_arguments(
                    node.state, moves,
                ), "", -1.0, 1)]
            return [Action("finish_trace", {}, "", -1.0, 1)]

        def values(self, _target, states, **_kwargs):
            return [0.0] * len(states)

    actor = SimpleNamespace(
        product_only_remap=True, vnext_v2_prefix=False,
        reject_target_retained_finish=True, compact_history=False,
        max_decisions=4, max_imports=32, branching=1,
        max_new_tokens=64, early_depth=2, early_beam=1, late_beam=1,
        value_weight=0.0, pointer_weight=0.0,
    )
    episode = ev.sample_episode(
        FakeRuntime(), "CO", actor, index=0,
        seed=ev.episode_seed(17, "CO", 0), stochastic=False,
    )
    assert episode["terminal"] is True
    assert episode["has_electron_event"] is True
    assert structural_exact(episode["precursor"], "CBr.[OH-]")
    assert [item["name"] for item in episode["actions"]] == [
        "import_fragments", "apply_electron_flow", "finish_trace",
    ]


def test_missing_reactions_remain_in_k_denominator_and_lineage_is_checked(tmp_path: Path):
    data = tmp_path / "valid.jsonl"
    rows = [
        {"id": f"r{i}", "source_id": f"r{i}", "target_smiles": "CO",
         "structural_precursor": "CN"}
        for i in range(2)
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_manifest.json").write_text(json.dumps({
        "base_model": ev.MODEL, "base_model_revision": ev.REVISION,
        "environment_revision": "natural_language_electron_event_v2",
        "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
    }))
    output = tmp_path / "predictions"
    output.mkdir()
    args = Namespace(
        data=data, expected_source_sha256=ev.sha256(data), adapter=adapter,
        expected_adapter_sha256=ev.sha256(adapter / "adapter_model.safetensors"),
        stage="state", output=output, episodes=5, sample_reactions=2,
        seed=17, max_new_tokens=512, max_context=4096, no_4bit=True,
    )
    fingerprint = ev.run_fingerprint(args)
    prediction = {
        "id": "r0", "source_id": "r0", "target": "CO", "seed": 17,
        "run_fingerprint": fingerprint,
        "episodes": [
            _episode(0, "CO", precursor="CN", score=-0.2, event=True),
            *[_episode(i, "CO") for i in range(1, 5)],
        ],
    }
    shard = output / "episodes.shard-00-of-01.jsonl"
    shard.write_text(json.dumps(prediction) + "\n")
    report = ev.aggregate(args)
    assert report["denominator"] == 2
    assert report["observed_reactions"] == 1
    assert report["missing_reactions"] == 1
    assert report["rates"]["generation_pass_at_1"] == 0.5
    assert report["rates"]["nll_ranked_top_5"] == 0.5
    assert report["rates"]["generation_pass_at_10"] is None
    prediction["run_fingerprint"] = "0" * 64
    shard.write_text(json.dumps(prediction) + "\n")
    with pytest.raises(ValueError, match="lineage mismatch"):
        ev.aggregate(args)


def test_resume_rejects_partial_episode_record_before_loading_model(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(ev, "checked_inputs", lambda _args: ([{"id": "r"}], "frozen"))
    shard = tmp_path / "episodes.shard-00-of-01.jsonl"
    shard.write_text(json.dumps({
        "id": "r", "seed": 17, "run_fingerprint": "frozen",
        "episodes": [{"index": i} for i in range(4)],
    }) + "\n")
    with pytest.raises(ValueError, match="incomplete episode"):
        ev.run(Namespace(output=tmp_path, episodes=5, seed=17))
