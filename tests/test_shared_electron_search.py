"""Persistence/reuse contracts, with real chemistry; not model benchmarks."""
from dataclasses import replace
import importlib
import json

import pytest


def store_module():
    # Missing implementation should fail the behavior tests, not collection.
    module = importlib.import_module('mechet.dynamic_electron_tree')
    assert hasattr(module, 'SharedSearchStore'), 'shared persistent search is missing'
    return importlib.import_module('mechet.shared_electron_search')


def proposal():
    from mechet.dynamic_electron_tree import Proposal
    return Proposal('import_fragments', {'fragments': [
        {'smiles': '[Cl-]', 'count': 1, 'purpose': 'electron_participant'}]},
        'import chloride', -2., 4, token_ids=[11, 12, 13, 14])


def test_saved_candidates_reused_after_reopening_without_another_generation(tmp_path):
    m = store_module()
    a = m.SharedSearchStore(tmp_path, {'model': 'frozen', 'executor': 'v1'})
    request = {'context': 'native-state-and-history', 'prompt_ids': [1, 2],
               'temperature': 1, 'seed': 17, 'n': 1, 'max_tokens': 384}
    lease, cached = a.acquire_request(request)
    assert cached is None
    lease.commit({'generations': [vars(proposal())], 'engine_finish_reasons': ['stop']})
    lease.close()
    b = m.SharedSearchStore(tmp_path, {'executor': 'v1', 'model': 'frozen'})
    lease, cached = b.acquire_request(request)
    assert lease is None
    assert cached['generations'][0]['raw'] == 'import chloride'
    assert cached['generations'][0]['token_ids'] == [11, 12, 13, 14]
    assert cached['generations'][0]['logprob'] == -2.


def test_concurrent_worker_defers_busy_context_then_reads_committed_result(tmp_path):
    m = store_module()
    a = m.SharedSearchStore(tmp_path, {'model': 'frozen'})
    b = m.SharedSearchStore(tmp_path, {'model': 'frozen'})
    lease, _ = a.acquire_request({'state': 'CO'})
    blocked, cached = b.acquire_request({'state': 'CO'})
    assert blocked is None and cached is None
    lease.commit({'generations': []}); lease.close()
    blocked, cached = b.acquire_request({'state': 'CO'})
    assert blocked is None and cached == {'generations': []}


@pytest.mark.parametrize('changed', [{'model': 'different'}, {'executor': 'different'}])
def test_changed_model_or_executor_cannot_reuse_other_namespace(tmp_path, changed):
    m = store_module()
    a = m.SharedSearchStore(tmp_path, {'model': 'frozen', 'executor': 'v1'})
    lease, _ = a.acquire_request({'context': 'same'})
    lease.commit({'generations': []}); lease.close()
    b = m.SharedSearchStore(tmp_path, {'model': 'frozen', 'executor': 'v1', **changed})
    lease, cached = b.acquire_request({'context': 'same'})
    assert cached is None and lease is not None
    lease.close()


@pytest.mark.parametrize('field,value', [('prompt_ids', [1, 3]), ('seed', 18),
                                       ('max_tokens', 768), ('temperature', 0)])
def test_changed_policy_input_or_sampling_does_not_reuse_candidates(tmp_path, field, value):
    m = store_module()
    store = m.SharedSearchStore(tmp_path, {'model': 'frozen'})
    request = {'context': 'same', 'prompt_ids': [1, 2], 'seed': 17,
               'max_tokens': 384, 'temperature': 1}
    lease, _ = store.acquire_request(request)
    lease.commit({'generations': []}); lease.close()
    lease, cached = store.acquire_request({**request, field: value})
    assert cached is None and lease is not None
    lease.close()


def test_shared_execution_preserves_current_path_likelihood_and_native_state(tmp_path):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree
    store = m.SharedSearchStore(tmp_path, {'executor': 'v1'})
    a = ElectronTree('route-target', 'CO', shared_store=store)
    a.incorporate(a.root_key, [proposal()])
    b = ElectronTree('single-reaction-target', 'CO', shared_store=store)
    b.incorporate(b.root_key, [replace(proposal(), logprob=-9., tokens=6)])
    child = b.contexts[b.edges[-1]['child_context']].node
    assert b.work['executor_calls'] == 0
    assert b.work['execution_cache_hits'] == 1
    assert child.logprob == -9. and child.tokens == 6
    assert child.state == '[CH3:1][OH:2].[Cl-:3]'
    assert child.imported == {'[Cl-]': 1}


def test_unfinished_tree_is_recoverable_before_finish_or_normal_exit(tmp_path):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    tree = ElectronTree('case', 'CO')
    journal = m.SearchJournal(tmp_path, tree.case_id, {'source_sha': 'source'})
    journal.initialize(tree)
    edges = tree.incorporate(tree.root_key, [proposal(), Proposal('unknown', {}, 'bad', -1, 2)])
    journal.record(tree, tree.root_key, edges)
    restored = m.SearchJournal(tmp_path, 'case', {'source_sha': 'source'}).restore()
    assert restored.stop_reason == ''
    assert len(restored.edges) == 2
    assert restored.edges[1]['status'] == 'rejected'
    assert restored.contexts[restored.edges[0]['child_context']].node.imported == {'[Cl-]': 1}
    assert restored.contexts[restored.root_key].stats.draws == 2
    assert restored.contexts[restored.root_key].stats.batches == 1
    assert restored.work['sampled_tokens'] == 6
    assert restored.next_context() is not None


def test_route_journal_recovers_outer_obligations_and_terminal_program(tmp_path):
    m = store_module()
    from mechet.reverse_route_network import RouteNetwork
    from mechet.dynamic_electron_tree import Proposal
    route = RouteNetwork('route', 'CO', {'[CH3+]', '[OH-]'}, mode='tree')
    journal = m.SearchJournal(tmp_path, route.case_id, {'stock_sha': 'stock'})
    journal.initialize(route)
    key = route.next_context().key
    # Real reverse heterolytic C--O cleavage, then real finish.
    from mechet.natural_language_electron_flow import render_event_arguments
    flow = Proposal('apply_electron_flow', render_event_arguments(
        '[CH3:1][OH:2]', [{'source': {'kind': 'BOND', 'atoms': [1, 2]},
                            'sink': {'kind': 'ATOM', 'atoms': [2]}, 'electrons': 2}]),
        'flow', -2, 4)
    edges = route.incorporate(key, [flow])
    journal.record(route, key, edges)
    child = edges[0]['child_context']
    edges = route.incorporate(child, [Proposal('finish_trace', {}, 'finish', -1, 2)])
    journal.record(route, child, edges)
    recovered = journal.restore(stock={'[CH3+]', '[OH-]'})
    assert recovered.to_dict()['reaction_edges'] == route.to_dict()['reaction_edges']
    assert recovered.to_dict()['reaction_trees'] == route.to_dict()['reaction_trees']
    assert len(recovered.reaction_edges) == 1
    assert recovered.report()['stock_solved']
    assert len(recovered.report()['complete_reaction_edges']) == 1


def test_journal_rejects_changed_run_contract(tmp_path):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree
    j = m.SearchJournal(tmp_path, 'case', {'source_sha': 'one'})
    j.initialize(ElectronTree('case', 'CO'))
    with pytest.raises(ValueError, match='contract'):
        m.SearchJournal(tmp_path, 'case', {'source_sha': 'two'})


def test_policy_wave_generates_only_missing_requests_and_defers_busy_ones(tmp_path):
    m = store_module()
    store = m.SharedSearchStore(tmp_path, {'model': 'frozen'})
    lease, _ = store.acquire_request({'context': 'cached'})
    lease.commit({'generations': [{'raw': 'saved'}]}); lease.close()
    busy, _ = store.acquire_request({'context': 'busy'})
    calls = []
    def generate(requests):
        calls.extend(requests)
        return [{'generations': [{'raw': 'new'}]} for _ in requests]
    results = m.resolve_policy_requests(store, [
        {'context': 'cached'}, {'context': 'new'}, {'context': 'busy'}], generate)
    assert calls == [{'context': 'new'}]
    assert results[0] == ({'generations': [{'raw': 'saved'}]}, True)
    assert results[1] == ({'generations': [{'raw': 'new'}]}, False)
    assert results[2] is None
    busy.close()


def test_changed_history_and_native_atom_addresses_do_not_share_execution(tmp_path):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree, context_key
    store = m.SharedSearchStore(tmp_path, {'executor': 'v1'})
    a = ElectronTree('first', 'CO', shared_store=store)
    a.incorporate(a.root_key, [proposal()])
    b = ElectronTree('second', 'CO', shared_store=store)
    b.contexts[b.root_key].node.visited.add('CCl')
    node = b.contexts.pop(b.root_key).node
    b.root_key = context_key(node)
    from mechet.dynamic_electron_tree import SearchContext, chemical_key, digest
    b.contexts[b.root_key] = SearchContext(b.root_key, node, digest(chemical_key(node.state)))
    b.incorporate(b.root_key, [proposal()])
    assert b.work['executor_calls'] == 1
    assert b.work['execution_cache_hits'] == 0


def test_truncated_partial_commit_is_ignored_but_corrupt_published_commit_fails(tmp_path):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree
    j = m.SearchJournal(tmp_path, 'case', {'source': 'one'})
    j.initialize(ElectronTree('case', 'CO'))
    (j.root / 'events' / '.incomplete.partial').write_text('{')
    assert j.restore().target == 'CO'
    path = j.root / 'events' / '000000000.json'
    data = json.loads(path.read_text()); data['payload']['new_trees']['single']['target'] = 'WRONG'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='corrupt'):
        j.restore()


def test_failed_generation_releases_claim_for_retry(tmp_path):
    m = store_module()
    store = m.SharedSearchStore(tmp_path, {'model': 'frozen'})
    def fail(requests):
        raise RuntimeError('engine failed')
    with pytest.raises(RuntimeError, match='engine failed'):
        m.resolve_policy_requests(store, [{'state': 'CO'}], fail)
    lease, record = store.acquire_request({'state': 'CO'})
    assert record is None and lease is not None
    lease.close()


def test_resource_rejection_is_not_a_shared_chemical_rejection(tmp_path, monkeypatch):
    m = store_module()
    import mechet.dynamic_electron_tree as module
    store = m.SharedSearchStore(tmp_path, {'executor': 'v1'})
    monkeypatch.setattr(module, 'available_host_memory', lambda: 1)
    first = module.ElectronTree('low-memory', 'CO', shared_store=store)
    first.incorporate(first.root_key, [proposal()])
    assert first.edges[-1]['error'] == 'STATE_ALLOCATION_RESOURCE_FAILURE'
    monkeypatch.setattr(module, 'available_host_memory', lambda: 1_000_000_000)
    second = module.ElectronTree('enough-memory', 'CO', shared_store=store)
    second.incorporate(second.root_key, [proposal()])
    assert second.edges[-1]['status'] == 'accepted'
    assert second.work['executor_calls'] == 1


def test_real_runner_saves_unfinished_graph_and_reuses_policy_in_new_attempt(tmp_path, monkeypatch):
    """Only external GPU/tokenizer/decoder are doubled; real runner/tools/store."""
    import sys
    from types import SimpleNamespace, ModuleType
    from collections import Counter
    from pathlib import Path
    from scripts import infer_dynamic_electron_trees as runner
    m = store_module()
    data = tmp_path / 'source.jsonl'
    data.write_text(json.dumps({'id': 'unit-fixture', 'product_unmapped': 'CO',
                               'structural_precursor': 'CO'}) + '\n')
    adapter = tmp_path / 'adapter'; adapter.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'unit fixture, not weights')
    (adapter / 'data_contract.json').write_text(json.dumps(
        {'environment_revision': 'natural_language_electron_event_history_v2'}))
    monkeypatch.setattr(runner, 'ADAPTER_SHA', runner.sha256(adapter / 'adapter_model.safetensors'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0'); monkeypatch.setenv('VLLM_USE_V1', '0')
    monkeypatch.setattr(runner, 'render_qwen_sft_tool_prefix',
        lambda tokenizer, messages, tools: json.dumps(messages, sort_keys=True))
    monkeypatch.setattr(runner, 'prediction_call', lambda raw, tokenizer: ('finish_trace', {}, ''))
    class Params:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    class Engine:
        calls = 0
        def __init__(self, **kwargs): pass
        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda s, **kw: list(s.encode()),
                                   decode=lambda ids, **kw: 'finish_trace')
        def generate(self, prompts, params, **kwargs):
            Engine.calls += 1
            if Engine.calls > 1: raise RuntimeError('unit-controlled engine interruption')
            return [SimpleNamespace(request_id=str(i), num_cached_tokens=0, outputs=[
                SimpleNamespace(token_ids=[11, 12], cumulative_logprob=-1., finish_reason='stop')
                for _ in range(param.n)]) for i, param in enumerate(params)]
    vllm = ModuleType('vllm'); vllm.__version__ = '0.8.5'; vllm.LLM = Engine; vllm.SamplingParams = Params
    lora = ModuleType('vllm.lora.request'); lora.LoRARequest = lambda *args: None
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.lora', ModuleType('vllm.lora'))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', lora)
    monkeypatch.setattr(runner, 'ForwardCallCounter', lambda engine:
                        SimpleNamespace(total_calls=0, by_request=Counter()))
    for attempt in ['first', 'second']:
        output = tmp_path / attempt
        monkeypatch.setattr(sys, 'argv', ['infer', '--data', str(data), '--source-sha', runner.sha256(data),
            '--expected-rows', '1', '--adapter', str(adapter), '--output', str(output),
            '--rank', '0', '--world', '1', '--shared-cache', str(tmp_path / 'shared')])
        with pytest.raises(RuntimeError, match='unit-controlled engine interruption'):
            runner.main()
        rank = output / 'tree' / 'rank00'
        manifest = json.loads((rank / 'manifest.json').read_text())
        restored = m.SearchJournal(rank / 'journals', 'unit-fixture', manifest).restore()
        assert len(restored.edges) == 16
        assert len(restored.terminals()) == 1
        assert restored.next_context() is not None
        raw = json.loads((rank / 'raw_generations.jsonl').read_text().splitlines()[0])
        assert raw['shared_cache_hit'] == (attempt == 'second')
        assert not (rank / 'complete.json').exists()
    # First attempt generated wave1+failed wave2. Second reused wave1 and
    # called the engine only for the missing wave2, not the cached wave1.
    assert Engine.calls == 3


def test_materialize_unfinished_journal_without_model_or_executor(tmp_path, monkeypatch):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree
    tree = ElectronTree('case', 'CO')
    journal = m.SearchJournal(tmp_path / 'journals', 'case', {'source_sha': 'source'})
    journal.initialize(tree)
    edges = tree.incorporate(tree.root_key, [proposal()])
    journal.record(tree, tree.root_key, edges)
    import importlib.util
    assert importlib.util.find_spec('scripts.materialize_electron_search') is not None, 'journal reader is missing'
    from scripts.materialize_electron_search import main
    import sys
    output = tmp_path / 'graph.json'
    monkeypatch.setattr(sys, 'argv', ['materialize', '--journal', str(journal.root), '--output', str(output)])
    assert main() == 0
    saved = json.loads(output.read_text())
    assert len(saved['edges']) == 1
    assert len(saved['contexts']) == 2
    assert saved['persistence']['search_complete'] is False
    assert saved['persistence']['journal_commits'] == 2


@pytest.mark.parametrize('invalid_first', [False, True])
def test_shared_finish_cache_does_not_accept_nonempty_arguments(tmp_path, invalid_first):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    store = m.SharedSearchStore(tmp_path, {'executor': 'v1'})
    for invalid in [invalid_first, not invalid_first]:
        tree = ElectronTree(str(invalid), 'CO', shared_store=store)
        arguments = {'instruction': 'finish now'} if invalid else {}
        tree.incorporate(tree.root_key, [Proposal('finish_trace', arguments, 'finish', -1, 2)])
        assert bool(tree.terminals()) == (not invalid)
        if invalid: assert 'FINISH_ARGUMENTS_NOT_EMPTY' in tree.edges[-1]['error']


def test_same_topology_better_route_program_updates_immediate_certificate(tmp_path):
    m = store_module()
    from mechet.dynamic_electron_tree import Proposal
    from mechet.reverse_route_network import RouteNetwork
    from mechet.natural_language_electron_flow import render_event_arguments
    route = RouteNetwork('route', 'CO', {'[CH3+]', '[OH-]'}, mode='tree')
    journal = m.SearchJournal(tmp_path, route.case_id, {'stock_sha': 'stock'})
    journal.initialize(route)
    key = route.next_context().key
    flow = Proposal('apply_electron_flow', render_event_arguments('[CH3:1][OH:2]', [
        {'source': {'kind': 'BOND', 'atoms': [1, 2]},
         'sink': {'kind': 'ATOM', 'atoms': [2]}, 'electrons': 2}]), 'flow', -2, 4)
    edges = route.incorporate(key, [flow]); journal.record(route, key, edges)
    child = edges[0]['child_context']
    for score, expected in [(-1., 3.), (-.1, 2.1)]:
        edges = route.incorporate(child, [Proposal('finish_trace', {}, 'finish', score, 2)])
        journal.record(route, child, edges)
        saved = m.read_record(journal.root / 'best_route.json')
        assert saved['certificate']['negative_loglikelihood'] == pytest.approx(expected)


def test_executor_memory_exception_is_not_reused_by_healthy_task(tmp_path, monkeypatch):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree
    from scripts import run_natural_language_value_search as runtime
    store = m.SharedSearchStore(tmp_path, {'executor': 'v1'})
    first = ElectronTree('unhealthy', 'CO', shared_store=store)
    original = runtime.map_unmapped_fragment
    def exhausted(*args, **kwargs): raise MemoryError('temporary allocation failure')
    monkeypatch.setattr(runtime, 'map_unmapped_fragment', exhausted)
    first.incorporate(first.root_key, [proposal()])
    assert first.edges[0]['error'].startswith('MemoryError:')
    monkeypatch.setattr(runtime, 'map_unmapped_fragment', original)
    second = ElectronTree('healthy', 'CO', shared_store=store)
    second.incorporate(second.root_key, [proposal()])
    assert second.edges[0]['status'] == 'accepted'
    assert second.work['executor_calls'] == 1


def test_stale_writer_cannot_overwrite_committed_graph_expansion(tmp_path):
    m = store_module()
    from mechet.dynamic_electron_tree import ElectronTree
    first = m.SearchJournal(tmp_path, 'case', {'source': 'one'})
    tree = ElectronTree('case', 'CO'); first.initialize(tree)
    stale = m.SearchJournal(tmp_path, 'case', {'source': 'one'})
    edges = tree.incorporate(tree.root_key, [proposal()]); first.record(tree, tree.root_key, edges)
    with pytest.raises(ValueError, match='writer|committed'):
        stale.record(tree, tree.root_key, [])
    recovered = m.SearchJournal(tmp_path, 'case', {'source': 'one'}).restore()
    assert len(recovered.edges) == 1
    assert recovered.contexts[recovered.edges[0]['child_context']].node.imported == {'[Cl-]': 1}


def test_materialization_rejects_other_cohort_stock_from_same_run(tmp_path, monkeypatch):
    m = store_module()
    from mechet.reverse_route_network import RouteNetwork
    from scripts.materialize_electron_search import main
    from scripts.infer_dynamic_electron_trees import sha256
    import sys
    good = tmp_path / 'good.txt'; good.write_text('[CH3+]\n[OH-]\n')
    bad = tmp_path / 'other.txt'; bad.write_text('CO\n')
    contract = {'stock_canonical_shas': {'cohort_a': sha256(good), 'cohort_b': sha256(bad)},
                'case_cohort': 'cohort_a', 'case_stock_sha': sha256(good)}
    j = m.SearchJournal(tmp_path / 'journals', 'route', contract)
    j.initialize(RouteNetwork('route', 'CO', {'[CH3+]', '[OH-]'}, mode='tree'))
    output = tmp_path / 'graph.json'
    monkeypatch.setattr(sys, 'argv', ['materialize', '--journal', str(j.root),
        '--output', str(output), '--stock', str(bad)])
    with pytest.raises(SystemExit): main()
    assert not output.exists()
    monkeypatch.setattr(sys, 'argv', ['materialize', '--journal', str(j.root),
        '--output', str(output), '--stock', str(good)])
    assert main() == 0
    assert json.loads(output.read_text())['already_in_stock'] is False
