"""Verify actor selection for the independent PR116 distillation task."""
import hashlib
import json
from pathlib import Path

import pytest

from mechet.route_joint_distillation import bind_distillation_source


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def joint_run(tmp_path):
    run = tmp_path / 'joint'
    cfg = tmp_path / 'contract.json'
    write(cfg, {'output_dir': str(run), 'step_budget': {'arm': 'J'}})
    digest = sha(cfg)
    for step, fraction in ((32, .2), (64, .1)):
        cp = run / 'checkpoints' / f'round-{step:06d}'
        cp.mkdir(parents=True)
        adapter = cp / 'adapter_model.safetensors'
        adapter.write_bytes(str(step).encode())
        write(cp / 'complete.json', dict(contract_sha256=digest, optimizer_step=step,
              next_round=step, payload_sha256={adapter.name: sha(adapter)}))
        summary = run / 'validation' / f'checkpoint-{step:06d}' / 'summary.json'
        write(summary, dict(binding={'contract_sha256': digest}, complete=True, denominator=512,
              actor_sha256=sha(adapter), checkpoint_round=step, checkpoint_path=str(cp),
              validation_reference_split='validation', verified_closed_at_4_lower_bound=fraction,
              censored_root_fraction=0, decision_requests_per_root=10))
    write(run / 'completion.json', dict(contract_sha256=digest, visited_target_episodes=2048,
                                        actor_sha256=sha(run/'checkpoints/round-000064/adapter_model.safetensors')))
    return cfg, run


def test_chooses_best_trained_actor_on_full_validation(tmp_path):
    cfg, run = joint_run(tmp_path)
    result = bind_distillation_source(cfg)
    assert Path(result['actor_directory']) == run / 'checkpoints/round-000032'
    assert result['actor_sha256'] == sha(Path(result['actor_directory'])/'adapter_model.safetensors')
    assert result['source_contract_sha256'] == sha(cfg)


def test_requires_joint_training_completion(tmp_path):
    cfg, run = joint_run(tmp_path)
    (run/'completion.json').unlink()
    with pytest.raises(ValueError, match='completed Joint'):
        bind_distillation_source(cfg)


def test_refuses_validation_receipt_from_different_actor(tmp_path):
    cfg, run = joint_run(tmp_path)
    path = run/'validation/checkpoint-000032/summary.json'
    data = json.loads(path.read_bytes());data['actor_sha256'] = '0'*64;write(path,data)
    with pytest.raises(ValueError, match='validation actor'):
        bind_distillation_source(cfg)


def test_distillation_schedule_uses_a_frozen_source_without_forward_teacher(monkeypatch):
    from mechet import route_joint_schedule as schedule

    monkeypatch.setattr(schedule, 'file_sha', {'train': schedule.TRAIN_SHA,
                                               'valid': schedule.VALID_SHA}.__getitem__)
    cfg = dict(step_budget=dict(arm='J-distill', roots='train', valid='valid'),
               joint_forward=dict(lambda_f=0., lambda_d=0., margin=.25, beta_d=.1),
               distill_source={'actor_sha256': 'verified-by-launcher'},
               parent_sha256=schedule.ACTOR_SHA, products_per_round=32, epochs=1,
               candidates=4, validation_targets=512,
               refit=dict(matched_continuations=1, max_forks_per_tree=2),
               acceleration=dict(refit_lanes=dict(credit_forks_per_tree=2, exploration_forks_per_tree=0)))
    schedule.validate(cfg)
    cfg['distill_source'] = None
    with pytest.raises(ValueError, match='source binding'):
        schedule.validate(cfg)
