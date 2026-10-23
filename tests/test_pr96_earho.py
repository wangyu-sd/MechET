"""EARHO contracts: fail if diagnostic data, stale actors or relative failures slip in."""
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/pr96_earho_contract.py'
    assert path.exists(), 'EARHO protocol implementation is missing'
    spec = importlib.util.spec_from_file_location('earho_contract', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_every_train_id_visited_once_including_partial_round():
    m = module()
    ids = [str(i) for i in range(259)]
    chunks = list(m.rounds(ids, 128))
    assert list(map(len, chunks)) == [128, 128, 3]
    assert [rid for chunk in chunks for rank in range(8) for rid in chunk[rank::8]] != []
    assert sorted(rid for chunk in chunks for rank in range(8) for rid in chunk[rank::8]) == sorted(ids)
    with pytest.raises(ValueError, match='DUPLICATE'):
        list(m.rounds(['a', 'a'], 128))


def test_all_wrong_candidates_have_zero_policy_advantages():
    m = module()
    rows = [{'first_state': 'A', 'first_terminal': False, 'reward': -.1, 'positive': False},
            {'first_state': 'B', 'first_terminal': False, 'reward': 0., 'positive': False}]
    assert m.assign_advantages(rows, lambda a, b: a == b)['effective'] is False
    assert [r['advantage'] for r in rows] == [0., 0.]


def test_equivalent_successors_do_not_create_fake_reward_contrast():
    m = module()
    rows = [{'first_state': 'same', 'first_terminal': False, 'reward': 1., 'positive': True},
            {'first_state': 'same', 'first_terminal': False, 'reward': .25, 'positive': True}]
    assert m.assign_advantages(rows, lambda a, b: a == b)['classes'] == 1
    assert [r['advantage'] for r in rows] == [0., 0.]


def test_successor_classes_receive_signed_rloo_and_duplicate_weighting():
    m = module()
    rows = [{'first_state': 'good', 'first_terminal': False, 'reward': 1., 'positive': True},
            {'first_state': 'bad', 'first_terminal': False, 'reward': 0., 'positive': False},
            {'first_state': 'good', 'first_terminal': False, 'reward': 1., 'positive': True}]
    report = m.assign_advantages(rows, lambda a, b: a == b)
    assert report['effective']
    assert [r['advantage'] for r in rows] == [1., -1., 1.]
    assert [r['policy_weight'] for r in rows] == [.25, .5, .25]


def test_horizon_does_not_promote_from_replay_or_rehearsal_only():
    m = module()
    s = [{'effective': False, 'productive': True, 'rehearsal': False}] * 20
    assert m.promote(2, s, 12, 16, .1) == 2
    s = [{'effective': True, 'productive': True, 'rehearsal': True}] * 20
    assert m.promote(2, s, 12, 16, .1) == 2
    s = [{'effective': True, 'productive': True, 'rehearsal': False}] * 16
    assert m.promote(2, s, 12, 16, .1) == 3
    assert m.promote(12, s, 12, 16, .1) == 12


def test_correct_terminal_overrides_an_earlier_reference_divergence():
    m = module()
    assert m.first_divergence({'structural_exact': True}, [{'depth': 0, 'error': 'bad'}], ['a', 'b'], lambda a,b:a==b) is None
    calls = [{'depth': 0, 'state_after': 'b', 'error': ''},
             {'depth': 1, 'state_after': 'wrong', 'error': ''}]
    assert m.first_divergence({'structural_exact': False}, calls, ['a', 'b', 'c'], lambda a,b:a==b) == 1


def test_early_finish_is_a_divergence_even_when_state_is_unchanged():
    m = module()
    calls = [{'depth': 0, 'state_after': 'same', 'error': '', 'predicted_name': 'finish_trace'}]
    assert m.first_divergence({'structural_exact': False}, calls, ['initial', 'same', 'last'],
                             lambda a,b:a==b, reference_terminals=[False, False, True]) == 0


def test_formal_contract_rejects_diagnostic_parent_and_test_sources():
    m = module()
    base = {'algorithm': 'pr96_earho_v1', 'parent_sha256': m.PARENT_SHA,
            'train_reactions': 10137, 'world_size': 8, 'candidates': 2,
            'train_file': '/data/train.jsonl', 'history_file': '/views/history/train.jsonl'}
    m.validate(base)
    with pytest.raises(ValueError, match='PARENT'):
        m.validate({**base, 'parent_sha256': 'b33e4b542e70b1709a608c02cb5a340a31d1571a0b339ba5bd158d86cca0d200'})
    with pytest.raises(ValueError, match='TRAIN_ONLY'):
        m.validate({**base, 'train_file': '/data/test.jsonl'})
    with pytest.raises(ValueError, match='DENOMINATOR'):
        m.validate({**base, 'train_reactions': 64})


def test_hardware_guard_accepts_declared_a100_but_not_wrong_or_mixed_devices():
    m = module()
    m.validate_hardware(['NVIDIA A100-SXM4-40GB']*8, 'A100')
    m.validate_hardware(['NVIDIA H20']*8, 'H20')
    with pytest.raises(ValueError, match='HARDWARE'):
        m.validate_hardware(['NVIDIA H20']*8, 'A100')
    with pytest.raises(ValueError, match='HARDWARE'):
        m.validate_hardware(['NVIDIA A100-SXM4-40GB']*7, 'A100')
    with pytest.raises(ValueError, match='HARDWARE'):
        m.validate_hardware(['NVIDIA A100-SXM4-40GB']*7+['NVIDIA V100'], 'A100')


def test_sampled_kl_uses_fixed_reference_not_new_behavior():
    m = module()
    import torch
    new = torch.tensor([-.5], requires_grad=True)
    loss, pg, kl = m.policy_loss(new, torch.tensor([-.5]), torch.tensor([-1.]), 1., .2, .02)
    assert pg.item() == pytest.approx(-1.)
    assert kl.item() == pytest.approx(0.10653066)
    loss.backward()
    assert torch.isfinite(new.grad).all()


def test_loss_uses_actual_model_device_and_causal_response_positions():
    m = module()
    import torch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 5, 4))
        def forward(self, input_ids, attention_mask, logits_to_keep, use_cache):
            assert input_ids.device.type == 'cpu'
            assert logits_to_keep.tolist() == [1, 2, 3]
            return type('Result', (), {'logits': self.logits[:, logits_to_keep]})()
    model = Tiny()
    values = m.token_logps(model, [0, 1], [2, 3, 0])
    assert values.tolist() == pytest.approx([-1.38629436]*3)
    (-values.mean()).backward()
    assert model.logits.grad[0, 0].abs().sum() == 0
    assert model.logits.grad[0, 4].abs().sum() == 0


def test_continuation_horizon_includes_first_action_and_censors_nonterminal():
    from types import SimpleNamespace
    m = module()
    seen = []
    branches = [{'node': SimpleNamespace(state=1, terminal=False), 'error': ''}]
    def generate(nodes):
        seen.extend(n.state for n in nodes)
        return [n.state + 1 for n in nodes]
    def execute(node, value):
        return SimpleNamespace(state=value, terminal=(value == 3)), ''
    m.continue_branches(branches, 2, generate, execute, lambda *args: None)
    assert seen == [1]
    assert branches[0]['node'].state == 2
    assert branches[0]['censored'] is True
    seen.clear()
    m.continue_branches(branches, 2, generate, execute, lambda *args: None)
    assert seen == [2]
    assert branches[0]['node'].terminal
    assert branches[0]['censored'] is False


def checkpoint_fixture(tmp_path):
    import hashlib
    import json
    path = tmp_path/'round-00003'
    path.mkdir()
    for name in ['optimizer.pt', 'adapter_model.safetensors', 'adapter_config.json']+[f'rng.rank{r:02d}.pt' for r in range(8)]:
        (path/name).write_bytes(b'unit')
    state = {'next_round': 3, 'horizon': 2, 'optimizer_step': 7,
             'visited_train_roots': 384, 'train_denominator': 10137,
             'completed_train_ids_sha256': 'a'*64}
    (path/'trainer_state.json').write_text(json.dumps(state))
    manifest = {'contract_sha256': 'frozen', 'files': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}, **state}
    (path/'complete.json').write_text(json.dumps(manifest))
    return path


def test_checkpoint_rejects_wrong_contract_or_tampered_optimizer(tmp_path):
    m = module()
    path = checkpoint_fixture(tmp_path)
    assert m.verify_checkpoint(path, 'frozen')['next_round'] == 3
    with pytest.raises(ValueError, match='CONTRACT'):
        m.verify_checkpoint(path, 'other')
    (path/'optimizer.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='HASH'):
        m.verify_checkpoint(path, 'frozen')


def test_resume_discovers_committed_checkpoint_even_if_latest_pointer_was_not_written(tmp_path):
    m = module()
    root = tmp_path/'checkpoints'
    root.mkdir()
    path = checkpoint_fixture(root)
    (root/'round-00004.building').mkdir()
    assert m.discover_checkpoint(tmp_path, 'frozen') == path


def test_cursor_tampering_and_omitted_rank_state_are_rejected(tmp_path):
    import json
    m = module()
    path = checkpoint_fixture(tmp_path)
    manifest = json.loads((path/'complete.json').read_text())
    manifest['next_round'] = 999
    (path/'complete.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='CURSOR'):
        m.verify_checkpoint(path, 'frozen')
    manifest['next_round'] = 3
    del manifest['files']['rng.rank07.pt']
    (path/'complete.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='COMPONENT'):
        m.verify_checkpoint(path, 'frozen')


def test_interrupted_checkpoint_attempt_does_not_collide_with_retried_save(tmp_path):
    m = module()
    first = m.checkpoint_staging(tmp_path, 1)
    (first/'partial').write_bytes(b'keep')
    second = m.checkpoint_staging(tmp_path, 1)
    assert first != second
    assert (first/'partial').read_bytes() == b'keep'
    assert second.parent == tmp_path/'checkpoints'


def _gradient_worker(rank, init_file, output):
    import torch
    import torch.distributed as dist
    m = module()
    dist.init_process_group('gloo', init_method='file://'+init_file, rank=rank, world_size=2)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    if rank == 1:
        (3*model(torch.ones(1, 1)).sum()).backward()
    m.synchronize_gradients(model, 3)
    torch.save(model.weight.grad, Path(output)/f'rank{rank}.pt')
    dist.destroy_process_group()


def test_distributed_missing_local_gradient_has_unbiased_partial_batch_mean(tmp_path):
    import torch
    import torch.multiprocessing as mp
    mp.spawn(_gradient_worker, args=(str(tmp_path/'gloo'), str(tmp_path)), nprocs=2, join=True)
    assert torch.load(tmp_path/'rank0.pt', weights_only=True).item() == 1.
    assert torch.load(tmp_path/'rank1.pt', weights_only=True).item() == 1.


def test_reference_replay_preserves_real_pr96_history_prefix_and_endpoint():
    import json
    path = Path(__file__).resolve().parents[1]/'scripts/train_pr96_earho.py'
    spec = importlib.util.spec_from_file_location('train_pr96_earho_unit', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
    train = root/'data/pr96_mech31k_mainonly_source_arrows_v2_20261022_01/inverse/train.jsonl'
    if not train.is_file():
        pytest.skip('large shared reference artifact is not installed')
    with train.open() as reader:
        source = json.loads(next(reader))
    history = []
    with (root/'outputs/agent/pr96_mech31k_qwen35_08b_mainonly_20261022_01/views/history/train.jsonl').open() as reader:
        for line in reader:
            row = json.loads(line)
            if str(row['source_id']) == str(source['source_id']):
                history.append(row)
            elif history:
                break
    nodes = runner.replay(source, history)
    assert len(nodes) == len(history)+1
    assert nodes[-1].terminal
    assert nodes[0].target == 'Oc1ccc(OCc2ccccc2)c(Br)c1'
