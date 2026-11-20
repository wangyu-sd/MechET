from mechet.external_route_validation import summarize_validation, validation_outcomes
from scripts.train_external_route_rl import make_episodes


def original(report_id, *, status='rejected_action', records=(), suffixes=()):
    return dict(task_id='target', uid=report_id, status=status, error='', reward=0.0,
                reward_evidence=dict(verified_stock_closed=status == 'stock_solved'),
                records=list(records), reactions=[], refit=dict(suffix_reports=list(suffixes)))


def test_refit_suffixes_count_toward_closure_without_double_counting_prefixes():
    from mechet.route_acceleration import SKIPPED, skipped_reward

    shared = dict(request_id='a', raw='x')
    finish = dict(request_id='b', accepted=True,
                  raw='<tool_call>{"name":"finish_trace","arguments":{}}</tool_call>')
    pruned = original('pruned', status=SKIPPED, records=[shared, dict(request_id='c', raw='x')])
    pruned.update(reward=None, reward_evidence=skipped_reward(pruned))
    solved = original('suffix', status='stock_solved', records=[shared, finish])
    base = original('root', records=[shared], suffixes=[solved, pruned])
    rows = [dict(task_id='target', target='CO')]
    outcomes = validation_outcomes(rows, [[base]], 1, max_forks=2)
    summary = summarize_validation(['target'], outcomes, 1)
    assert summary['verified_closed_roots'] == 1
    assert summary['sampled_route_episodes'] == 1
    assert summary['refit_suffix_episodes'] == 2
    assert summary['refit_pruned_suffixes'] == 1
    assert summary['censored_roots'] == 0
    assert summary['model_decision_requests'] == 3
    assert summary['completed_reaction_programs'] == 1


def test_search_validation_preserves_policy_root_seed_and_owns_fork_requests():
    from mechet.route_refit import RefitRouteEpisode

    rows = [dict(task_id='target', target='CO')]
    cfg = dict(candidates=1, seed=17, max_action_tokens=256, world_size=1,
               executor_workers_per_rank=1, _validation_rng_namespace='frozen')
    policy = make_episodes(rows, set(), [], cfg, 'actor')[0]
    search = make_episodes(rows, set(), [], dict(cfg, refit={}), 'actor')[0]
    assert isinstance(search, RefitRouteEpisode)
    assert policy.next_decision().request_id == search.next_decision().request_id
    branch = search.fork(search.uid + '/fork-1')
    assert branch.next_decision().request_id != search.next_decision().request_id


def test_distillation_pairs_full_cohort_at_both_actor_versions(monkeypatch, tmp_path):
    from scripts import train_external_route_rl as trainer

    rows = [dict(task_id=str(i), target='CO') for i in range(512)]
    parent_model, child_model = object(), object()
    parent_path = str(tmp_path/'parent')
    final_path = str(tmp_path/'checkpoints'/'round-000064')
    calls = []

    def record(rows_seen, stock, protected, model, tokenizer, cfg, root, step, digest, fractions, *, search):
        assert rows_seen is rows
        calls.append((model, cfg['_rollout_adapter_path'], step, digest, search))
        return {'complete': True, 'denominator': 512}

    monkeypatch.setattr(trainer, 'full_validation', record)
    result = trainer.compare_distillation(rows, set(), [], child_model, parent_model, None,
        {'parent': parent_path, 'parent_sha256': 'parent'}, tmp_path, 64, 'child', {0: [0.], 64: [1.]})
    assert list(result['joint']) == list(result['distilled']) == ['policy_only', 'small', 'full']
    assert calls == [
        (parent_model, parent_path, 0, 'parent', None),
        (parent_model, parent_path, 0, 'parent', ('joint', 'small')),
        (parent_model, parent_path, 0, 'parent', ('joint', 'full')),
        (child_model, final_path, 64, 'child', None),
        (child_model, final_path, 64, 'child', ('distilled', 'small')),
        (child_model, final_path, 64, 'child', ('distilled', 'full')),
    ]
