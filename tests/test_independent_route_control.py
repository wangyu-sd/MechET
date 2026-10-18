"""Software fixtures for independent rollouts; not model/chemistry experiments."""
import importlib

import pytest


def implementation():
    try:
        return importlib.import_module('scripts.independent_route_control')
    except ModuleNotFoundError:
        pytest.fail('independent complete-route rollout controller missing')


def row():
    return {'id': 't', 'product_unmapped': 'CO', 'cohort': 'c',
            'ground_truth': 'SECRET_GT', 'reference_routes': ['SECRET_REFERENCE']}


def test_independent_episodes_have_no_shared_frontier_or_cross_episode_seeds():
    cls = implementation().IndependentRouteEpisode
    a = cls(row(), {'C'}, episode_index=0, seed=17)
    b = cls(row(), {'C'}, episode_index=1, seed=17)
    again = cls(row(), {'C'}, episode_index=0, seed=17)
    ra, rb = a.next_request(), b.next_request()
    assert a.uid != b.uid and ra['seed'] != rb['seed']
    assert ra['seed'] == again.next_request()['seed']
    assert a.network is not b.network and a.network.shared_store is None
    assert b.network.trees['CO'].shared_store is None
    assert a.network.trees['CO'].root_key == b.network.trees['CO'].root_key


def test_policy_observation_matches_original_stageII_and_never_sees_gold():
    from scripts.build_natural_language_event_sft import SYSTEM
    from scripts.run_natural_language_value_search import policy_prompt
    episode = implementation().IndependentRouteEpisode(row(), {'C'}, episode_index=0, seed=17)
    request = episode.next_request(); node = episode.network.contexts[request['context_key']].node
    assert request['messages'] == [{'role': 'system', 'content': SYSTEM}, {'role': 'user',
        'content': policy_prompt(node.target, node.state, include_inventory=True,
                                  actions=node.actions, compact_history=True)}]
    text = str(request)
    assert 'SECRET_GT' not in text and 'SECRET_REFERENCE' not in text
    assert 'ANNOTATED CURRENT STATE:' in text
    assert request['completion_budget'] == 384


def test_interleaved_import_is_executed_and_import_only_finish_is_not_stock_success():
    from mechet.dynamic_electron_tree import Proposal
    e = implementation().IndependentRouteEpisode(row(), {'C'}, episode_index=0, seed=17)
    r = e.next_request()
    e.accept(r['context_key'], Proposal('import_fragments', {'fragments': [
        {'smiles': 'O', 'count': 1, 'purpose': 'electron_participant'}]}, 'raw import', -1, 5),
        completion_room=1000, max_tokens=384)
    r = e.next_request()
    assert 'SECRET' not in str(r)
    assert len(e.network.contexts[r['context_key']].node.actions) == 1
    e.accept(r['context_key'], Proposal('finish_trace', {}, 'raw finish', -1, 3),
             completion_room=1000, max_tokens=384)
    assert e.next_request() is None
    report = e.report()
    assert report['stock_solved'] is False and report['draws'] == 2
    assert report['work']['sampled_tokens'] == 8
    assert report['episode_index'] == 0 and report['id'] == 't'
    assert len(e.network.rejected_route_terminals) == 1


def test_invalid_action_ends_one_attempt_without_backtracking_or_selective_retry():
    from mechet.dynamic_electron_tree import Proposal
    e = implementation().IndependentRouteEpisode(row(), {'C'}, episode_index=0, seed=17)
    r = e.next_request()
    e.accept(r['context_key'], Proposal('wrong_tool', {}, 'bad', -1, 7),
             completion_room=1000, max_tokens=384)
    assert e.next_request() is None
    assert e.report()['final_status'] == 'rejected_action'
    assert e.report()['work']['raw_candidates'] == 1
    assert e.report()['stock_solved'] is False
    assert e.network.trees['CO'].edges[0]['status'] == 'rejected'


def test_truncation_uses_original_adaptive_action_room_and_then_censors_model_limit():
    from mechet.dynamic_electron_tree import Proposal
    e = implementation().IndependentRouteEpisode(row(), {'C'}, episode_index=0, seed=17)
    a = e.next_request()
    e.accept(a['context_key'], Proposal('', {}, 'cut', -1, 384, truncated=True),
             completion_room=500, max_tokens=384)
    b = e.next_request()
    assert b['context_key'] == a['context_key'] and b['seed'] != a['seed']
    assert b['completion_budget'] == 768
    e.accept(b['context_key'], Proposal('', {}, 'cut', -1, 500, truncated=True),
             completion_room=500, max_tokens=500)
    assert e.next_request() is None
    r = e.report()
    assert r['final_status'] == 'resource_censored'
    assert r['draws'] == 2 and r['work']['sampled_tokens'] == 884
    assert not r['search_complete']


def test_already_stock_zero_step_is_retained_but_not_a_generated_solution():
    e = implementation().IndependentRouteEpisode(row(), {'CO'}, episode_index=0, seed=17)
    assert e.next_request() is None
    assert e.report()['final_status'] == 'already_stock'
    assert e.report()['already_in_stock'] is True
    assert e.report()['stock_solved'] is False


def test_out_of_order_or_repeated_decision_cannot_mutate_an_episode():
    from mechet.dynamic_electron_tree import Proposal
    e = implementation().IndependentRouteEpisode(row(), {'C'}, episode_index=0, seed=17)
    e.next_request()
    with pytest.raises(ValueError, match='pending'):
        e.accept('foreign_context', Proposal('finish_trace', {}, 'finish', -1, 3),
                 completion_room=1000, max_tokens=384)
    assert e.report()['draws'] == 0


def test_reported_generation_room_cannot_change_the_frozen_sampling_protocol():
    from mechet.dynamic_electron_tree import Proposal
    e = implementation().IndependentRouteEpisode(row(), {'C'}, episode_index=0, seed=17)
    r = e.next_request()
    with pytest.raises(ValueError, match='generation receipt'):
        e.accept(r['context_key'], Proposal('finish_trace', {}, 'finish', -1, 3),
                 completion_room=1000, max_tokens=1000)
    assert e.report()['draws'] == 0


def test_recursive_episode_closes_all_precursor_obligations_not_one_leaf():
    from test_reverse_route_network import terminal
    e = implementation().IndependentRouteEpisode({**row(), 'product_unmapped': 'CCO'},
                                                 {'C', 'O'}, episode_index=0, seed=17)
    # Accepted-terminal boundary fixtures, not generated model predictions.
    e.network._register_terminal('CCO', terminal('[CH4:1].[CH3:2][OH:3]', 'CCO'))
    assert not e.report()['stock_solved'] and 'CO' in e.network.trees
    e.network._register_terminal('CO', terminal('[CH4:1].[OH2:2]', 'CO'))
    assert e.next_request() is None
    r = e.report()
    assert r['stock_solved'] and r['final_status'] == 'stock_solved'
    assert sorted(r['stock_leaves']) == ['C', 'C', 'O']


def frozen_plan_files(tmp_path):
    import gzip, hashlib, json
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def write(path, value): path.write_text(json.dumps(value)); return path
    rows = [{'id': f'{cohort}:{i}', 'product_unmapped': 'CO', 'cohort': cohort,
             'reference_routes': ['SECRET_GT']} for cohort, count in
            [('ursa_expert100', 100), ('ursa_drugs100', 100),
             ('retrocast_mkt_cnv160', 160), ('paroutes_derived_n5_100', 100)] for i in range(count)]
    targets = tmp_path/'targets.jsonl'
    targets.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    stock = tmp_path/'stock.txt'; stock.write_text('C\nO\n')
    assets = write(tmp_path/'assets.json', {'stock_preparation_complete': True,
        'full_target_rows': 460, 'full_targets_path': str(targets), 'full_targets_sha256': sha(targets),
        'normalization': 'mapless_canonical_stereo_isotope_charge_preserved_whole_salts_v1',
        'stocks': {'stock': {'canonical_path': str(stock), 'canonical_sha256': sha(stock), 'normalized_rows': 2}},
        'cohorts': {c: {'rows': n, 'stock': 'stock'} for c, n in
            [('ursa_expert100', 100), ('ursa_drugs100', 100),
             ('retrocast_mkt_cnv160', 160), ('paroutes_derived_n5_100', 100)]}})
    adapter = tmp_path/'adapter'; adapter.mkdir()
    (adapter/'adapter_model.safetensors').write_bytes(b'software fixture; not model weights')
    actor_sha = sha(adapter/'adapter_model.safetensors')
    write(adapter/'adapter_manifest.json', {'base_model': 'Qwen/Qwen3-0.6B',
        'base_model_revision': 'c1899de289a04d12100db370d81485cdf75e47ca',
        'tokenizer_revision': 'c1899de289a04d12100db370d81485cdf75e47ca',
        'environment_revision': 'natural_language_electron_event_history_v2',
        'executor_revision': 'MECH_PROOF_v1_full_coverage_v4', 'adapter_sha256': '0'*64})
    write(adapter/'data_contract.json', {'environment_revision': 'natural_language_electron_event_history_v2'})
    write(adapter/'adapter_config.json', {'base_model_name_or_path': 'Qwen/Qwen3-0.6B'})
    freeze = tmp_path/'freeze'; freeze.mkdir()
    population = freeze/'target_population.jsonl.gz'
    with gzip.open(population, 'wt') as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({'run_id': 'r1_mcts_pr88', 'target_id': r['id'],
                               'report_status': 'observed' if i < 200 else 'pending'})+'\n')
    headers = []
    for rank in range(8):
        path = write(tmp_path/f'header{rank}.json', {'model': 'Qwen/Qwen3-0.6B',
            'revision': 'c1899de289a04d12100db370d81485cdf75e47ca', 'adapter_sha': actor_sha,
            'rank': rank, 'world': 8, 'full_denominator': 460,
            'history_contract': 'natural_language_electron_event_history_v2', 'max_model_len': 8192})
        headers.append({'run_id': 'r1_mcts_pr88', 'manifest_path': str(path), 'manifest_sha256': sha(path),
                        'actor_sha256': actor_sha})
    write(freeze/'prediction_freeze_manifest.json', {'complete': True, 'actor_sha256': actor_sha,
        'route_assets_manifest_sha256': sha(assets), 'source_reports': headers,
        'output_files': {population.name: {'sha256': sha(population)}}})
    return freeze, assets, adapter, actor_sha


def test_prepared_plan_preserves_full460_and_fixed200_without_exposing_references(tmp_path):
    import json
    freeze, assets, adapter, actor_sha = frozen_plan_files(tmp_path)
    result = implementation().prepare_plan(freeze, assets, adapter, tmp_path/'prepared',
        expected_actor_sha=actor_sha)
    assert result['prepared_only'] is True and result['new_model_inference'] is False
    assert result['full_target_denominator'] == 460 and result['paired_observed_denominator'] == 200
    assert result['episode_counts'] == [1, 4, 10]
    assert result['actor_weight_sha256'] == actor_sha
    assert result['original_adapter_manifest_sha_field_matches_weights'] is False
    targets = (tmp_path/'prepared/targets_product_only.jsonl').read_text()
    assert 'SECRET_GT' not in targets and 'reference_routes' not in targets
    assert len(targets.splitlines()) == 460
    assert len(json.loads((tmp_path/'prepared/paired_observed_target_ids.json').read_text())) == 200
    with pytest.raises(FileExistsError):
        implementation().prepare_plan(freeze, assets, adapter, tmp_path/'prepared', expected_actor_sha=actor_sha)


@pytest.mark.parametrize('mutation', ['missing_target', 'wrong_weights', 'changed_header'])
def test_prepare_refuses_changed_actor_or_filtered_target_universe(tmp_path, mutation):
    import json, hashlib
    freeze, assets, adapter, actor_sha = frozen_plan_files(tmp_path)
    if mutation == 'wrong_weights':
        (adapter/'adapter_model.safetensors').write_bytes(b'changed')
    if mutation == 'changed_header':
        (tmp_path/'header0.json').write_text('{}')
    if mutation == 'missing_target':
        p = tmp_path/'targets.jsonl'; p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
        a = json.loads(assets.read_text()); a['full_targets_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        assets.write_text(json.dumps(a))
        mp = freeze/'prediction_freeze_manifest.json'; m = json.loads(mp.read_text())
        m['route_assets_manifest_sha256'] = hashlib.sha256(assets.read_bytes()).hexdigest(); mp.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        implementation().prepare_plan(freeze, assets, adapter, tmp_path/'must-not-exist', expected_actor_sha=actor_sha)
    assert not (tmp_path/'must-not-exist').exists()
