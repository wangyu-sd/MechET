"""Executable CPU regressions, not reaction/model smoke experiments."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from mechet.graph_earho import AdaptiveHorizon, VerifiedOutcome
from mechet.graph_earho import successor_group_advantages
from mechet.graph_earho import ExecutedAlternative, pool_executed_successors
from mechet.endpoints import split_precursor_endpoints
from scripts import train_crn_graph_earho as trainer


def test_reference_alignment_keeps_explicit_hydrogen_event_addresses():
    pytest.importorskip("crn_jepa")
    state = "[O-:1].[H:33][O:34][CH3:35]"
    moves = (
        {"source": {"kind": "BOND", "atoms": [33, 34]},
         "sink": {"kind": "ATOM", "atoms": [34]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [1]},
         "sink": {"kind": "BOND", "atoms": [1, 33]}, "electrons": 2},
    )
    aligned = trainer.align_reference_moves(state, state, moves)
    assert aligned is not None
    assert aligned[0].source.atom_maps == (33, 34)
    assert aligned[1].sink.atom_maps == (1, 33)


def test_accepted_gt_terminal_retaining_target_gets_the_benchmark_reward():
    # Real FlowER val_2565 retains its selected target while other imported
    # components react. The endpoint return must not add a second no-op gate.
    tree = ast.parse(Path(trainer.__file__).read_text())
    call = next(node.value for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "endpoint_hit"
                        for t in node.targets))
    expression = ast.fix_missing_locations(ast.Expression(call))
    state = "[CH3:1][OH:2].[Cl-:3]"
    scope = dict(vars(trainer), endings=[SimpleNamespace(terminal=True, state=state)],
                 anchor=SimpleNamespace(target="CO"), endpoint="[CH3:1][OH:2]",
                 product_origin_mapped="[CH3:1][OH:2]")
    assert eval(compile(expression, trainer.__file__, "eval"), scope) is True


def test_advantage_call_has_real_trainer_binding():
    # Execute the actual call expression in the trainer's imported namespace.
    tree = ast.parse(Path(trainer.__file__).read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "successor_group_advantages")
    expression = ast.fix_missing_locations(ast.Expression(body=call))
    scope = dict(vars(trainer), outcomes=[VerifiedOutcome(True, endpoint_exact=True),
                                        VerifiedOutcome(False)],
                 args=SimpleNamespace(actor_reward="endpoint"))
    try:
        result = eval(compile(expression, trainer.__file__, "eval"), scope)
    except NameError as exc:
        pytest.fail(f"actual training call is unbound: {exc}")
    assert result == (1.0, -1.0)


def test_main_conditional_loss_call_uses_complete_support_and_draw_returns():
    import torch
    tree = ast.parse(Path(trainer.__file__).read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "conditional_endpoint_policy_loss")
    scores = torch.tensor([0., 0.], requires_grad=True)
    scope = dict(vars(trainer), logps=list(scores), draw_indices=(0, 1),
                 draw_returns=[1., 0.], args=SimpleNamespace(temperature=1.), full_requested=True)
    loss = eval(compile(ast.fix_missing_locations(ast.Expression(call)), trainer.__file__, "eval"), scope)
    gradient, = torch.autograd.grad(loss, scores)
    torch.testing.assert_close(gradient, torch.tensor([-.5, .5]))


def test_training_parser_composes_with_shared_inference_options():
    try:
        parser = trainer.training_parser()
    except AttributeError:
        pytest.fail("trainer lacks an independently verifiable composed CLI")
    args = parser.parse_args([
        "--data", "train.jsonl", "--output", "output",
        "--source-sha256", "frozen", "--policy-adapter", "adapter",
        "--crn-root", "graph", "--graph-checkpoint", "graph.pt",
        "--tool-route", "greedy_name",
    ])
    assert args.source_sha256 == "frozen"
    assert args.tool_route == "greedy_name"
    # A reward continuation must execute the same K=1 follower as held-out inference.
    assert args.rollout_beam_width == args.graph_flow_topk == 1
    assert args.actor_reward == "endpoint"
    assert getattr(args, "policy_objective", None) == "conditional_endpoint"
    assert getattr(args, "exploration_proposals", None) == 4


def test_frozen_follower_precision_is_bound_to_resume_and_export(tmp_path):
    from scripts.pr83_resume import prepare_training_resume
    tree = ast.parse(Path(trainer.__file__).read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    manifest_node = next(node.value for node in assignments
                         if any(isinstance(t, ast.Name) and t.id == "manifest"
                                for t in node.targets))
    settings_node = next(value for key, value in zip(manifest_node.keys, manifest_node.values)
                         if isinstance(key, ast.Constant) and key.value == "settings")
    controller_node = next(node.value for node in assignments
                           if any(isinstance(t, ast.Name) and t.id == "controller"
                                  for t in node.targets))
    args = trainer.training_parser().parse_args([
        "--data", "train.jsonl", "--output", "output", "--policy-adapter", "adapter",
        "--crn-root", "graph", "--graph-checkpoint", "graph.pt",
    ])

    def evaluate(node):
        expression = ast.fix_missing_locations(ast.Expression(node))
        return eval(compile(expression, trainer.__file__, "eval"), {"args": args})

    frozen_settings = evaluate(settings_node)
    frozen_controller = evaluate(controller_node)
    assert frozen_settings.get("dtype") == frozen_controller.get("dtype") == "bfloat16"
    assert frozen_settings.get("no_4bit") is frozen_controller.get("no_4bit") is True
    frozen = {"settings": frozen_settings}
    prepare_training_resume(tmp_path, manifest=frozen)
    (tmp_path / "latest_train_state.pt").write_bytes(b"unit-test-checkpoint")
    assert prepare_training_resume(tmp_path, manifest=frozen, resume=True)
    for name, alternate in (("dtype", "float16"), ("no_4bit", False)):
        original = getattr(args, name)
        setattr(args, name, alternate)
        with pytest.raises(ValueError, match="changed"):
            prepare_training_resume(tmp_path, manifest={"settings": evaluate(settings_node)}, resume=True)
        setattr(args, name, original)


def test_local_match_alone_cannot_create_endpoint_actor_gradient():
    assert successor_group_advantages([
        VerifiedOutcome(True, reference_successor=True), VerifiedOutcome(False),
    ], endpoint_only=True) == (0.0, 0.0)


def test_known_endpoint_winner_has_credit_but_censored_successor_is_not_negative():
    assert successor_group_advantages([
        VerifiedOutcome(True, endpoint_exact=True),
        VerifiedOutcome(True, reference_successor=True, exhausted=True),
        VerifiedOutcome(False),
    ], endpoint_only=True) == (1.0, 0.0, -1.0)


def test_successor_pool_keeps_product_origin_for_endpoint_reward():
    origin = "[CH3:1][CH2:2][OH:3]"
    a = "[CH3:1][OH:6].[Br-:4].[CH3:5][Br:7].[CH2:2]=[O:3]"
    b = "[CH3:5][OH:6].[Br-:7].[CH3:1][Br:4].[CH2:2]=[O:3]"
    assert trainer.visible(a) == trainer.visible(b) == "C=O.CBr.CO.[Br-]"
    assert trainer.visible(split_precursor_endpoints(a, origin).structural) == "C=O.CO"
    assert trainer.visible(split_precursor_endpoints(b, origin).structural) == "C=O.CBr"
    if not hasattr(trainer, "executed_successor_key"):
        pytest.fail("reward-equivalent successor key is missing")
    groups = pool_executed_successors([
        ExecutedAlternative(0, trainer.executed_successor_key(a), True, -1.0),
        ExecutedAlternative(1, trainer.executed_successor_key(b), True, -2.0),
    ])
    assert len(groups) == 2
    reordered = ".".join(reversed(a.split(".")))
    assert trainer.executed_successor_key(reordered) == trainer.executed_successor_key(a)


def test_successor_pool_preserves_stereochemistry():
    if not hasattr(trainer, "executed_successor_key"):
        pytest.fail("stereochemistry-aware successor key is missing")
    assert trainer.executed_successor_key("[F:1]/[CH:2]=[CH:3]/[F:4]") != \
           trainer.executed_successor_key("[F:1]/[CH:2]=[CH:3]\\[F:4]")


def test_local_only_support_does_not_promote_endpoint_horizon():
    horizon = AdaptiveHorizon(current=2, maximum=8, min_effective_groups=4)
    promoted, flag = horizon.advance(total_groups=4, effective_groups=4,
                                    productive_groups=4, endpoint_groups=0)
    assert flag is False
    assert promoted.current == 2


def test_endpoint_support_can_promote_horizon():
    horizon = AdaptiveHorizon(current=2, maximum=8, min_effective_groups=4)
    promoted, flag = horizon.advance(total_groups=4, effective_groups=4,
                                    productive_groups=4, endpoint_groups=2)
    assert flag is True
    assert promoted.current == 3


def test_endpoint_curriculum_evidence_is_validated():
    with pytest.raises(ValueError):
        AdaptiveHorizon().advance(total_groups=4, effective_groups=4,
                                 productive_groups=2, endpoint_groups=3)


def test_full_epoch_seal_accepts_new_head_and_rejects_old_ablation(tmp_path, monkeypatch):
    import json
    from scripts import run_pr83_formal_campaign as campaign
    monkeypatch.setattr(campaign, "DATA_ROOT", tmp_path)
    root = tmp_path / "outputs/agent/pr83_graph_earho_conditional_stage1_train257167_20261012"
    root.mkdir(parents=True)
    head = root / "graph_earho_policy_head.pt"
    head.write_bytes(b"unit-test-only-head")
    report = {"algorithm": "graph_earho_conditional_endpoint_v4",
              "stats": {"processed": 257167, "selected": 257167},
              "graph_earho_head_sha256": campaign.sha(head)}
    (root / "train_report.json").write_text(json.dumps(report))
    campaign.seal_stage("rl")
    report["algorithm"] = "success_gated_truncated_slate_policy_gradient_v1"
    (root / "train_report.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="algorithm"):
        campaign.seal_stage("rl")
