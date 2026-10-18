"""Full460 software manifests and scripted records, never model experiments."""
import hashlib
import json
from pathlib import Path

import pytest


def fixture_plan(tmp_path, *, all_stock=False):
    from test_independent_route_control import frozen_plan_files
    from scripts.independent_route_control import prepare_plan
    from scripts.independent_route_cli import validate_plan
    freeze, assets, adapter, actor_sha = frozen_plan_files(tmp_path)
    if all_stock:
        stock = tmp_path/'stock.txt'; stock.write_text('CO\n')
        metadata = json.loads(assets.read_bytes())
        metadata['stocks']['stock'].update(canonical_sha256=hashlib.sha256(stock.read_bytes()).hexdigest(), normalized_rows=1)
        assets.write_text(json.dumps(metadata))
        source = freeze/'prediction_freeze_manifest.json'; value = json.loads(source.read_bytes())
        value['route_assets_manifest_sha256'] = hashlib.sha256(assets.read_bytes()).hexdigest()
        source.write_text(json.dumps(value))
    root = tmp_path/'prepared'
    prepare_plan(freeze, assets, adapter, root, expected_actor_sha=actor_sha)
    path = root/'independent_plan.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = validate_plan(path, assets, expected_actor_sha=actor_sha, expected_plan_sha=digest)
    return loaded, path, assets, actor_sha, digest


def test_validator_rederives_full_plan_and_never_exposes_reference_fields(tmp_path):
    loaded, _, _, _, _ = fixture_plan(tmp_path)
    assert len(loaded.rows) == 460 and len(loaded.paired_ids) == 200
    assert set(loaded.stocks) == {'ursa_expert100', 'ursa_drugs100',
                                  'retrocast_mkt_cnv160', 'paroutes_derived_n5_100'}
    assert all(stock == {'C', 'O'} for stock in loaded.stocks.values())
    assert all(set(r) == {'id', 'product_unmapped', 'cohort'} for r in loaded.rows)
    assert 'SECRET' not in json.dumps(loaded.rows)


@pytest.mark.parametrize('mutation', ['product_only_file', 'plan_budget', 'weights'])
def test_validator_rejects_filtered_poisoned_or_stale_inputs_even_with_resealed_plan(tmp_path, mutation):
    from scripts.independent_route_cli import validate_plan
    _, path, assets, actor_sha, digest = fixture_plan(tmp_path)
    if mutation == 'product_only_file':
        (path.parent/'targets_product_only.jsonl').write_text('{}\n')
    elif mutation == 'plan_budget':
        value = json.loads(path.read_bytes()); value['episode_counts'] = [1]
        path.write_text(json.dumps(value)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        (tmp_path/'adapter/adapter_model.safetensors').write_bytes(b'wrong actor')
    with pytest.raises(ValueError):
        validate_plan(path, assets, expected_actor_sha=actor_sha, expected_plan_sha=digest)


def test_eight_shards_are_disjoint_and_cover_all460_original_targets(tmp_path):
    from scripts.independent_route_cli import shard_contract
    loaded, _, _, _, _ = fixture_plan(tmp_path)
    contracts = [shard_contract(loaded, rank) for rank in range(8)]
    assert [len(c['assigned_target_ids']) for c in contracts] == [58, 58, 58, 58, 57, 57, 57, 57]
    combined = [i for c in contracts for i in c['assigned_target_ids']]
    assert len(set(combined)) == len(combined) == 460
    assert all(c['episodes_per_target'] == 10 and c['world'] == 8 for c in contracts)
    assert all(c['full_target_denominator'] == 460 and c['paired_observed_denominator'] == 200 for c in contracts)
    with pytest.raises(ValueError): shard_contract(loaded, 8)


def test_empty_collection_preserves_full_and_paired_denominators(tmp_path):
    from scripts.independent_route_cli import collect_results
    loaded, _, _, _, _ = fixture_plan(tmp_path)
    value = collect_results(loaded, tmp_path/'no-run')
    assert not value['complete'] and value['missing_ranks'] == list(range(8))
    assert value['prefixes']['10']['target_denominator'] == 460
    assert value['prefixes']['10']['pending_or_censored_episodes'] == 4600
    assert value['prefixes']['10']['paired']['target_denominator'] == 200
    assert value['reported_episode_output_tokens'] == value['output_tokens_lower_bound'] == 0


def populate_one_software_episode(loaded, root):
    from mechet.shared_electron_search import durable_json
    from scripts.independent_route_cli import shard_contract
    from scripts.independent_route_runtime import run_episodes
    from test_independent_route_runtime import Tokenizer, ScriptedTransport
    shard = root/'tree/rank00'; shard.mkdir(parents=True)
    durable_json(shard/'manifest.json', shard_contract(loaded, 0))
    runtime = shard/'runtime'
    run_episodes(loaded.rows[:1], loaded.stocks, Tokenizer(), ScriptedTransport(runtime),
                 runtime, episode_count=1, inflight=1, cpu_workers=1)
    return shard


def test_cross_shard_collection_counts_real_saved_episode_not_progress_percent(tmp_path):
    from scripts.independent_route_cli import collect_results
    from test_independent_route_runtime import IMPORT, FINISH
    loaded, _, _, _, _ = fixture_plan(tmp_path)
    root = tmp_path/'scripted-software-run'
    populate_one_software_episode(loaded, root)
    value = collect_results(loaded, root)
    assert not value['complete'] and value['missing_ranks'] == list(range(1, 8))
    assert value['prefixes']['10']['reported_episodes'] == 1
    assert value['prefixes']['10']['finished_episodes'] == 1
    assert value['prefixes']['10']['pending_or_censored_episodes'] == 4599
    assert value['prefixes']['10']['stock_solved_targets'] == 0
    assert value['output_tokens_lower_bound'] == len(IMPORT.encode())+len(FINISH.encode())
    assert value['reported_episode_output_tokens'] == value['output_tokens_lower_bound']
    assert not value['work_complete']


@pytest.mark.parametrize('mutation', ['missing_raw', 'foreign_report', 'wrong_header', 'missing_journal'])
def test_collection_rejects_incomplete_raw_evidence_or_wrong_shard_ownership(tmp_path, mutation):
    from mechet.shared_electron_search import durable_json, read_record
    from scripts.independent_route_cli import collect_results
    loaded, _, _, _, _ = fixture_plan(tmp_path)
    root = tmp_path/'scripted-software-run'; shard = populate_one_software_episode(loaded, root)
    if mutation == 'missing_raw':
        next(shard.glob('runtime/episodes/*/raw_outputs/*.json')).unlink()
    elif mutation == 'foreign_report':
        path = next(shard.glob('runtime/episodes/*/report.json'))
        report = read_record(path); report['id'] = loaded.rows[1]['id']; durable_json(path, report)
    elif mutation == 'wrong_header':
        header = read_record(shard/'manifest.json'); header['rank'] = 1; durable_json(shard/'manifest.json', header)
    else:
        next(shard.glob('runtime/episodes/*/journal/*/events/000000001.json')).unlink()
    with pytest.raises(ValueError): collect_results(loaded, root)


def test_run_cli_without_authorization_cannot_create_output_or_import_a_model(tmp_path):
    from scripts.independent_route_cli import main
    with pytest.raises(SystemExit):
        main(['run', '--plan', str(tmp_path/'missing-plan.json'), '--route-assets',
              str(tmp_path/'missing-assets.json'), '--output', str(tmp_path/'never-created'), '--rank', '0'])
    assert not (tmp_path/'never-created').exists()


def test_authorized_worker_preserves_shard_scope_and_does_not_count_already_stock_as_synthesis(tmp_path):
    from scripts.independent_route_cli import run_worker, collect_results
    from test_independent_route_runtime import Tokenizer
    loaded, _, _, _, _ = fixture_plan(tmp_path, all_stock=True)
    root = tmp_path/'zero-step-software-fixture'
    class NoPolicyTransport:
        def submit(self, request):
            raise AssertionError('already-stock targets must never request a policy sample')
        def poll(self, timeout=0.):
            return None
    def factory(*args, **kwargs):
        assert kwargs['authorized_model_execution'] is True
        return Tokenizer(), NoPolicyTransport(), {'software_fixture': True}
    progress = run_worker(loaded, root, rank=0, authorized_model_execution=True,
                          transport_factory=factory, inflight=8, cpu_workers=2)
    assert progress['expected_episodes'] == progress['finished_episodes'] == 580
    assert progress['output_tokens'] == 0
    value = collect_results(loaded, root)
    assert value['prefixes']['10']['already_stock_targets'] == 58
    assert value['prefixes']['10']['stock_solved_targets'] == 0
    assert value['prefixes']['10']['pending_or_censored_episodes'] == 4020
    assert not value['complete']  # seven shards absent, despite the one worker's100% progress
