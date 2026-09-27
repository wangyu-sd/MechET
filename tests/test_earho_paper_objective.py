"""Executable, bounded-horizon credit contracts from paper Section 3."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mechet.natural_language_anchor_branch_rl import (
    assign_local_advantages,
    paper_earho_return,
)
from scripts.natural_language_anchor_branch_stage import _score_rollout
from scripts.run_earho_v2 import load_earho_config, validate_contract
from scripts.run_natural_language_anchor_branch_rl import worker_command


WEIGHTS = {"lambda_s": 0.25, "lambda_e": 0.05,
           "lambda_c": 0.25, "lambda_n": 0.25}


def _row(fingerprint, *, endpoint=False, successor=False, decisions=1,
         first_state="CN", failure="", horizon=4):
    score = {
        "correct": endpoint,
        "reference_first_successor_exact": successor,
        "first_successor_state": first_state,
        "target_retained": False,
        "failure": failure,
        "decisions": decisions,
    }
    score["earho_return_terms"] = paper_earho_return(
        score, anchor_state="[CH3:1][OH:2]", horizon=horizon, weights=WEIGHTS,
    )
    return {
        "id": "reaction", "anchor": {"state_hash": "same"},
        "prompt_mode": "unified", "action_fingerprint": fingerprint,
        "score": score, "reward": score["earho_return_terms"]["return"],
    }


def test_paper_return_is_the_actual_bounded_objective():
    row = _row("reference", successor=True, decisions=2)
    terms = row["score"]["earho_return_terms"]
    assert terms["R_end"] == 0
    assert terms["R_succ"] == 1
    assert terms["R_exec"] == pytest.approx(0.5)
    assert terms["return"] == pytest.approx(0.275)
    assert _row("endpoint", endpoint=True, decisions=4)["reward"] == pytest.approx(1.05)
    assert _row("wrong", decisions=4)["reward"] == pytest.approx(0.05)
    assert _row("cycle", decisions=1, failure="ValueError:STATE_CYCLE")["reward"] == pytest.approx(-0.2375)
    assert _row("noop", first_state="CO")["reward"] == pytest.approx(-0.2375)
    finish = _row("finish", endpoint=True, first_state="CO")
    finish["score"]["first_successor_terminal"] = True
    finish_terms = paper_earho_return(
        finish["score"], anchor_state="[CH3:1][OH:2]", horizon=4,
        weights=WEIGHTS,
    )
    assert finish_terms["R_noop"] == 0
    with pytest.raises(ValueError, match="outside horizon"):
        _row("overlong", decisions=5)


def test_paper_advantages_use_rollout_return_but_never_promote_wrong_action():
    records = [
        _row("reference", successor=True, decisions=1),
        _row("reference", successor=True, decisions=3),
        _row("endpoint", endpoint=True, decisions=4),
        _row("wrong", decisions=4),
    ]
    summary = assign_local_advantages(
        records, success_gated=True, paper_objective=True,
    )
    assert summary["effective"]
    assert records[0]["anchor_action_q"] == records[1]["anchor_action_q"]
    assert records[0]["advantage"] == records[1]["advantage"]
    assert 0 < records[0]["advantage"] < records[2]["advantage"]
    assert records[3]["advantage"] < 0
    assert records[3]["anchor_action_q"] == pytest.approx(0.05)
    all_wrong = [_row("wrong-a", decisions=4), _row("wrong-b", decisions=1)]
    result = assign_local_advantages(
        all_wrong, success_gated=True, paper_objective=True,
    )
    assert not result["effective"]
    assert all(row["advantage"] == 0 and not row["update_eligible"] for row in all_wrong)


def test_paper_collector_does_not_compute_legacy_potential(monkeypatch):
    import scripts.natural_language_anchor_branch_stage as stage

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy endpoint-similarity shaping was called")

    monkeypatch.setattr(stage, "endpoint_shaped_reward", forbidden)
    task = SimpleNamespace(
        anchor_state="[CH3:1][OH:2]", target="CO", expected_precursor="CN"
    )
    node = SimpleNamespace(state="[CH3:1][NH2:2]", terminal=True, actions=[])
    score = _score_rollout(
        task, node, "", 2, first_successor_state="[CH3:1][NH2:2]",
        invalid_penalty=0.1, wrong_terminal_penalty=0.5,
        endpoint_similarity_weight=0.45, first_successor_progress_weight=0.25,
        nonexact_reward_ceiling=0.01, target_retained_penalty=0.75,
        reference_first_successor_state="[CH3:1][NH2:2]", reference_first_successor_weight=0.25,
        paper_weights=WEIGHTS, continuation_limit=3,
    )
    assert score["correct"]
    assert score["reward"] == pytest.approx(1 + 0.25 + 0.05 * 2 / 3)
    assert score["reward_terms"]["return"] == score["reward"]
    assert score["earho_return_terms"]["horizon"] == 3


def test_reference_successor_credit_requires_matching_terminal_status():
    task = SimpleNamespace(
        anchor_state="[CH3:1][OH:2]", target="CO", expected_precursor="CN"
    )
    node = SimpleNamespace(state="[CH3:1][OH:2]", terminal=True, actions=[])
    score = _score_rollout(
        task, node, "", 1, first_successor_state=node.state,
        first_successor_terminal=True,
        invalid_penalty=0.1, wrong_terminal_penalty=0.5,
        endpoint_similarity_weight=0.45, first_successor_progress_weight=0.25,
        nonexact_reward_ceiling=0.01, target_retained_penalty=0.75,
        reference_first_successor_state=node.state,
        reference_first_successor_terminal=False,
        reference_first_successor_weight=0.25,
        paper_weights=WEIGHTS, continuation_limit=1,
    )
    assert not score["reference_first_successor_exact"]
    assert not score["earho_return_terms"]["verified_positive"]


@pytest.mark.parametrize("name", [
    "earho_paper_mech_uspto31k_prefixv3_8a100.yaml",
    "earho_paper_flower_strict_prefixv3_8h20.yaml",
])
def test_paper_config_is_separate_and_reaches_collector(name, tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_earho_config(root / "configs/agent" / name)
    assert cfg["reward"]["contract"] == "paper_earho_bounded_horizon_v1"
    assert "prefixv3" in cfg["output_dir"]
    command = worker_command(
        cfg, tmp_path / "source.jsonl", tmp_path / "actor",
        tmp_path / "rollout.jsonl", 0, frontier=2,
        round_index=0, evaluation=False,
    )
    assert "--paper-earho-objective" in command
    assert command[command.index("--paper-lambda-s") + 1] == "0.25"


def test_formal_driver_rejects_old_unified_prompt_contract():
    with pytest.raises(ValueError, match="distinct SFT-aligned tool and text prefixes"):
        validate_contract({
            "protocol_version": "trajectory_history_v2",
            "prompt_prefix_contract": "qwen_sft_template_generation_prefix_v2",
        })


@pytest.mark.parametrize("name", [
    "earho_paper_flower_strict_8h20.yaml",
    "earho_paper_mech_uspto31k_8a100.yaml",
    "earho_paper_mech_uspto31k_8h20.yaml",
    "earho_paper_flower_strict_k2_gt_smoke_8a100.yaml",
    "earho_paper_flower_strict_k2_prefixv2_8a100.yaml",
])
def test_historical_paper_configs_cannot_be_resubmitted_as_v3(name):
    root = Path(__file__).resolve().parents[1]
    cfg = load_earho_config(root / "configs/agent" / name)
    assert "prefixv3" not in cfg["output_dir"]
    with pytest.raises(ValueError, match="distinct SFT-aligned tool and text prefixes"):
        validate_contract(cfg)


def test_k2_gt_smoke_keeps_reward_and_parent_but_limits_actor_candidates(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_earho_config(
        root / "configs/agent/earho_paper_flower_strict_k2_gt_smoke_prefixv3_8a100.yaml"
    )
    baseline = load_earho_config(
        root / "configs/agent/earho_paper_flower_strict_prefixv3_8h20.yaml"
    )
    assert cfg["candidates_per_product"] == 2
    assert cfg["rounds"] == 1
    assert cfg["products_per_round"] == 128
    assert cfg["initial_adapter_model_sha256"] == baseline["initial_adapter_model_sha256"]
    assert cfg["reward"] == baseline["reward"]
    assert cfg["optimization"] == baseline["optimization"]
    assert cfg["output_dir"] != baseline["output_dir"]
    command = worker_command(
        cfg, tmp_path / "source.jsonl", tmp_path / "actor",
        tmp_path / "rollout.jsonl", 0, frontier=2,
        round_index=0, evaluation=False,
    )
    assert command[command.index("--k") + 1] == "2"
    assert "--paper-earho-objective" in command
    assert "--success-gated-advantages" in command
