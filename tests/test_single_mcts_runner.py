import pytest
from types import SimpleNamespace


def test_missing_nll_is_not_silently_ranked_as_zero():
    from scripts.infer_single_reaction_mcts import decode_request
    output=SimpleNamespace(request_id='x',num_cached_tokens=0,outputs=[SimpleNamespace(
        token_ids=[1],cumulative_logprob=None,finish_reason='stop')])
    with pytest.raises(RuntimeError,match='NLL'):
        decode_request(output,SimpleNamespace(decode=lambda *a,**k:'x'),lambda *a:('finish_trace',{},None))


def test_raw_request_record_keeps_tokens_scores_and_finish_reason():
    from scripts.infer_single_reaction_mcts import decode_request
    output=SimpleNamespace(request_id='x',num_cached_tokens=2,outputs=[SimpleNamespace(
        token_ids=[5,6],cumulative_logprob=-3.,finish_reason='length')])
    record=decode_request(output,SimpleNamespace(decode=lambda *a,**k:'raw'),lambda *a:('finish_trace',{},None))
    assert record['generations'][0]['logprob']==-3.
    assert record['generations'][0]['token_ids']==[5,6]
    assert record['generations'][0]['truncated'] is True


def test_row_projection_never_passes_gold_fields_to_search(tmp_path):
    from scripts.infer_single_reaction_mcts import make_controller
    from scripts.single_reaction_mcts import SingleReactionMCTSStore
    row={'id':'test','product_unmapped':'CO','reactants':'INVALID GOLD','reference':'do not expose'}
    c=make_controller(row,mcts_store=SingleReactionMCTSStore(tmp_path,{}),shared_store=None,
                      checkpoint_path=None,program_goal=10)
    assert c.target=='CO' and c.case_id=='test'
    assert not hasattr(c,'reference') and not hasattr(c,'reactants')
    c.close()


def test_resource_limited_reports_do_not_seal_a_complete_search():
    from scripts.infer_single_reaction_mcts import search_is_complete
    assert search_is_complete([{'stop_reason':'MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'}],1)
    assert not search_is_complete([{'stop_reason':'CONTEXT_MEMORY_BUDGET'}],1)
    assert not search_is_complete([],1)


def test_short_truncated_completion_does_not_pretend_to_exhaust_full_context():
    from scripts.infer_single_reaction_mcts import completion_hit_context_limit
    from mechet.dynamic_electron_tree import Proposal
    truncated=[Proposal('','', 'partial',-1.,384,truncated=True)]
    assert not completion_hit_context_limit({'max_tokens':384,'prompt_ids':[1]*1000},truncated,8192)
    assert completion_hit_context_limit({'max_tokens':32,'prompt_ids':[1]*8160},truncated,8192)


@pytest.fixture
def cpu_single_mcts_runner(tmp_path, monkeypatch):
    """Double only external GPU/tokenizer; run the real scheduler/store/tools."""
    import json
    import sys
    from types import ModuleType
    from scripts import infer_single_reaction_mcts as runner

    source = tmp_path / 'unit.jsonl'
    source.write_text(json.dumps({'id': 'unit-case', 'product_unmapped': 'CO',
                                  'structural_precursor': 'CO'}) + '\n')
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'unit fixture, not weights')
    (adapter / 'data_contract.json').write_text(json.dumps(
        {'environment_revision': 'natural_language_electron_event_history_v2'}))
    monkeypatch.setattr(runner, 'SOURCE_SHA', runner.sha256(source))
    monkeypatch.setattr(runner, 'ADAPTER_SHA', runner.sha256(adapter / 'adapter_model.safetensors'))
    original_load = runner.load_source
    monkeypatch.setattr(runner, 'load_source', lambda path, expected: original_load(path, 1))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('VLLM_USE_V1', '0')
    monkeypatch.setattr(runner, 'render_qwen_sft_tool_prefix',
                        lambda tokenizer, messages, tools: json.dumps(messages, sort_keys=True))
    imported = '<tool_call>{"name":"import_fragments","arguments":{"fragments":[{"smiles":"[Cl-]","count":1,"purpose":"electron_participant"}]}}</tool_call>'
    finished = '<tool_call>{"name":"finish_trace","arguments":{}}</tool_call>'

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Engine:
        calls = 0

        def __init__(self, **kwargs):
            self.llm_engine = self
            self.active = {}
            self.request_count = 0
            self.model_executor = SimpleNamespace(execute_model=lambda *a, **kw: None)

        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda text, **kw: list(text.encode()),
                                   decode=lambda ids, **kw: imported if ids[0] == 11 else finished,
                                   all_special_tokens=[])

        def add_request(self, identifier, prompt, params, **kwargs):
            Engine.calls += 1
            self.request_count += 1
            self.active[identifier] = (params.n, [11, 12] if self.request_count == 1 else [21, 22])

        def step(self):
            results = [SimpleNamespace(request_id=key, finished=True, num_cached_tokens=0,
                         outputs=[SimpleNamespace(token_ids=tokens, cumulative_logprob=-1.,
                                                  finish_reason='stop') for _ in range(n)])
                       for key, (n, tokens) in self.active.items()]
            self.active.clear()
            return results

        def abort_request(self, identifiers):
            for identifier in identifiers:
                self.active.pop(identifier, None)

    vllm = ModuleType('vllm')
    vllm.__version__ = '0.8.5'
    vllm.LLM, vllm.SamplingParams = Engine, Params
    lora = ModuleType('vllm.lora.request')
    lora.LoRARequest = lambda *a: None
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.lora', ModuleType('vllm.lora'))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', lora)

    def run(attempt):
        output = tmp_path / attempt
        monkeypatch.setattr(sys, 'argv', ['infer', '--data', str(source), '--adapter', str(adapter),
            '--output', str(output), '--rank', '0', '--world', '1', '--program-goal', '1',
            '--shared-cache', str(tmp_path / 'cache'), '--mcts-shared-root', str(tmp_path / 'planning' / attempt)])
        return runner.main(), output / 'tree' / 'rank00'

    return run, runner, Engine


def test_real_rolling_runner_keeps_raw_journal_and_reuses_cache(cpu_single_mcts_runner):
    import json
    from mechet.shared_electron_search import SearchJournal
    run, runner, engine = cpu_single_mcts_runner
    for attempt in ('first', 'second'):
        status, root = run(attempt)
        assert status == 0
        sealed = json.loads((root / 'complete.json').read_text())
        assert len(sealed['rows']) == 1 and sealed['search_complete']
        assert sealed['manifest']['policy_persistence'] == 'bounded_async_commit_before_execute_v1'
        progress = json.loads((root / 'progress.json').read_text())
        assert progress['policy_persistence']['pending_batches'] == 0
        assert progress['rollout_liveness']['local_checks'] > 0
        assert progress['rollout_liveness']['filesystem_checks'] == 0
        raw = [json.loads(line) for line in (root / 'raw_generations.jsonl').read_text().splitlines()]
        assert len(raw) == 2
        assert all(row['shared_cache_hit'] == (attempt == 'second') for row in raw)
        assert all(len(row['generations']) == 16 for row in raw)
        assert {tuple(row['generations'][0]['token_ids']) for row in raw} == {(11, 12), (21, 22)}
        assert all(g['logprob'] == -1. for row in raw for g in row['generations'])
        prompts = [json.loads(line) for line in (root / 'policy_contexts.jsonl').read_text().splitlines()]
        assert len(prompts) == 2 and all(row['token_ids'] for row in prompts)
        manifest = json.loads((root / 'manifest.json').read_text())
        recovered = SearchJournal(root / 'journals', 'unit-case', manifest).restore(stock=set())
        assert recovered.work['raw_candidates'] == 32
        assert not (root / 'complete.json.partial').exists()
    assert engine.calls == 2  # Second attempt reused durable results, no new sampling.


def test_runner_never_executes_or_seals_after_policy_disk_failure(cpu_single_mcts_runner, monkeypatch):
    from pathlib import Path
    from mechet import shared_electron_search as storage
    from scripts.single_reaction_mcts import SingleReactionMCTS
    run, runner, _ = cpu_single_mcts_runner
    original = storage.durable_json

    def fail_policy_disk(path, value):
        if 'policy' in Path(path).parts:
            raise OSError('unit policy disk full')
        return original(path, value)

    def must_not_execute(*args, **kwargs):
        pytest.fail('executor observed a policy result whose durable commit failed')

    monkeypatch.setattr(storage, 'durable_json', fail_policy_disk)
    monkeypatch.setattr(SingleReactionMCTS, 'incorporate', must_not_execute)
    with pytest.raises(OSError, match='policy disk full'):
        run('broken')
    # The temporary output is a sibling of the fake adapter/cache paths.
    import sys
    output = Path(sys.argv[sys.argv.index('--output') + 1]) / 'tree' / 'rank00'
    assert not (output / 'complete.json').exists()


def test_resource_shutdown_counts_generated_results_still_in_io_queue(cpu_single_mcts_runner, monkeypatch):
    import json
    from scripts.mcts_runtime_io import DurablePolicyWriter
    run, runner, _ = cpu_single_mcts_runner
    admitted = {'value': False}
    original_submit = DurablePolicyWriter.submit

    def signal_admission(self, *args, **kwargs):
        result = original_submit(self, *args, **kwargs)
        admitted['value'] = True
        return result

    class Guard:
        def __init__(self, **kwargs):
            pass

        def exhausted(self, **kwargs):
            return admitted['value']

    monkeypatch.setattr(DurablePolicyWriter, 'submit', signal_admission)
    monkeypatch.setattr(runner, 'HostMemoryGuard', Guard)
    _, root = run('resource-stop')
    sealed = json.loads((root / 'complete.json').read_text())
    raw = json.loads((root / 'raw_generations.jsonl').read_text())
    assert not sealed['search_complete'] and sealed['resource_stop'] == 'TASK_HOST_MEMORY_BUDGET'
    assert sealed['tokens'] == 32
    assert sum(g['tokens'] for g in raw['generations']) == 32
    assert sealed['rows'][0]['work']['policy_request_batches'] == 1


def test_resource_shutdown_drains_completed_gpu_results_behind_full_io_queue(
        cpu_single_mcts_runner, tmp_path, monkeypatch):
    import json
    import sys
    import threading
    from pathlib import Path
    from mechet import shared_electron_search as storage
    from scripts.infer_dynamic_electron_trees import load_source
    run, runner, engine = cpu_single_mcts_runner
    source = tmp_path / 'unit.jsonl'
    rows = [{'id': 'unit-case', 'product_unmapped': 'CO', 'structural_precursor': 'CO'},
            {'id': 'unit-case-b', 'product_unmapped': 'CCO', 'structural_precursor': 'CCO'}]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    monkeypatch.setattr(runner, 'SOURCE_SHA', runner.sha256(source))
    monkeypatch.setattr(runner, 'load_source', lambda path, expected: load_source(path, 2))
    blocked, release = threading.Event(), threading.Event()
    original_disk = storage.durable_json

    def block_first_policy_commit(path, value):
        if 'policy' in Path(path).parts and not blocked.is_set():
            blocked.set()
            assert release.wait(5), 'test failed to reach the backpressured GPU completion'
        return original_disk(path, value)

    pipes = []
    original_pipe = runner.ContinuousBatchEngine

    class Pipe(original_pipe):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pipes.append(self)

    class Guard:
        def __init__(self, **kwargs):
            pass

        def exhausted(self, **kwargs):
            stop = blocked.is_set() and engine.calls == 2 and not pipes[0].outputs.empty()
            if stop:
                release.set()
            return stop

    original_parse = runner.argparse.ArgumentParser.parse_args

    def single_slot(parser, *args, **kwargs):
        parsed = original_parse(parser, *args, **kwargs)
        parsed.inflight_requests = 1
        return parsed

    monkeypatch.setattr(storage, 'durable_json', block_first_policy_commit)
    monkeypatch.setattr(runner, 'ContinuousBatchEngine', Pipe)
    monkeypatch.setattr(runner, 'HostMemoryGuard', Guard)
    monkeypatch.setattr(runner.argparse.ArgumentParser, 'parse_args', single_slot)
    try:
        _, root = run('resource-backpressure')
    finally:
        release.set()
    sealed = json.loads((root / 'complete.json').read_text())
    raw = [json.loads(line) for line in (root / 'raw_generations.jsonl').read_text().splitlines()]
    assert not sealed['search_complete'] and sealed['resource_stop'] == 'TASK_HOST_MEMORY_BUDGET'
    assert engine.calls == 2 and sealed['tokens'] == 64
    assert len(raw) == 2 and {r['case_id'] for r in raw} == {'unit-case', 'unit-case-b'}
    assert sum(r['work']['policy_request_batches'] for r in sealed['rows']) == 2
    assert json.loads((root / 'progress.json').read_text())['GPU_inflight_requests'] == 0
