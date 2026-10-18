"""CPU software fixtures only; these scripted completions are not predictions."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['tools'] and kwargs['add_generation_prompt'] is False
        return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n'
                       for m in messages)

    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


def output(request_id, text, *, logprob=-2., finish='stop'):
    return SimpleNamespace(request_id=request_id, num_cached_tokens=0, outputs=[
        SimpleNamespace(token_ids=list(text.encode()), cumulative_logprob=logprob,
                        finish_reason=finish)])


IMPORT = '<tool_call>'+json.dumps({'name': 'import_fragments', 'arguments': {
    'fragments': [{'smiles': 'O', 'count': 1, 'purpose': 'electron_participant'}]}})+'</tool_call>'
FINISH = '<tool_call>{"name":"finish_trace","arguments":{}}</tool_call>'


class ScriptedTransport:
    """Replace only the expensive GPU boundary; execute/persist real tool steps."""
    def __init__(self, root, *, first_batch_size=1):
        self.root = Path(root)
        self.queue = []
        self.draws = {}
        self.peak = 0
        self.first_batch_size = first_batch_size
        self.first_batch_released = False
        self.deadline = time.monotonic()+5

    def submit(self, request):
        from mechet.shared_electron_search import read_record
        # A missing pre-generation durable input must fail the software test.
        saved = read_record(self.root/'episodes'/request['episode_uid']/
                            'requests'/f"{request['draw_index']:09d}.json")
        assert saved == request
        assert 'SECRET' not in json.dumps(request)
        uid = request['episode_uid']; draw = self.draws.get(uid, 0)
        self.draws[uid] = draw+1
        rid = str(sum(self.draws.values()))
        self.queue.append((rid, output(rid, IMPORT if draw == 0 else FINISH)))
        self.peak = max(self.peak, len(self.queue))
        return rid

    def poll(self, timeout=0.):
        # An explicit asynchronous boundary is essential: instantaneous fake
        # completions otherwise make peak concurrency depend on OS scheduling.
        if not self.first_batch_released:
            if len(self.queue) < self.first_batch_size:
                if time.monotonic() > self.deadline:
                    raise AssertionError('software transport never received its concurrent first batch')
                return None
            self.first_batch_released = True
        # Deliberately finish out of submission order.
        return self.queue.pop() if self.queue else None


def rows():
    return [{'id': x, 'product_unmapped': 'CO', 'cohort': 'fixture',
             'ground_truth': 'SECRET_GT'} for x in ['a', 'b']]


def test_requests_preserve_stageII_prefix_unwarped_sampling_and_actual_room():
    from scripts.independent_route_control import IndependentRouteEpisode
    from scripts.independent_route_runtime import prepare_request, sampling_kwargs
    e = IndependentRouteEpisode(rows()[0], {'C'}, episode_index=0, seed=17)
    req = prepare_request(e, Tokenizer(), max_model_len=8192)
    assert req['prompt'].endswith('<|im_end|>\n<|im_start|>assistant\n')
    assert '<think>' not in req['prompt'] and 'SECRET' not in req['prompt']
    params = sampling_kwargs(req)
    assert params['n'] == 1 and params['temperature'] == 1 and params['top_p'] == 1
    assert params['top_k'] == -1 and params['repetition_penalty'] == 1
    assert params['presence_penalty'] == params['frequency_penalty'] == 0
    assert params['max_tokens'] == 384 and params['logprobs'] == 0
    assert req['completion_room'] == 8192-len(req['prompt_ids'])
    assert params['seed'] == e.pending['seed']
    assert 'messages' not in params and 'episode_uid' not in params


def test_oversized_prompt_censors_without_silently_truncating_history():
    from scripts.independent_route_control import IndependentRouteEpisode
    from scripts.independent_route_runtime import prepare_request
    e = IndependentRouteEpisode(rows()[0], {'C'}, episode_index=0, seed=17)
    assert prepare_request(e, Tokenizer(), max_model_len=20) is None
    assert e.report()['final_status'] == 'resource_censored'
    assert e.report()['draws'] == 0 and not e.report()['search_complete']


@pytest.mark.parametrize('invalid', ['two_outputs', 'missing_probability'])
def test_decoder_refuses_multiple_candidates_or_unavailable_nll(invalid):
    from scripts.independent_route_runtime import decode_completion
    value = output('r', FINISH)
    if invalid == 'two_outputs': value.outputs *= 2
    else: value.outputs[0].cumulative_logprob = None
    with pytest.raises(ValueError): decode_completion(value, Tokenizer())


def test_rolling_episodes_keep_every_raw_input_output_and_private_journal(tmp_path):
    from mechet.shared_electron_search import read_record
    from scripts.independent_route_runtime import run_episodes
    root = tmp_path/'software-fixture'; transport = ScriptedTransport(root, first_batch_size=2)
    result = run_episodes(rows(), {'fixture': {'C'}}, Tokenizer(), transport, root,
                          episode_count=2, inflight=4, cpu_workers=2)
    assert transport.peak >= 2  # catches a serial request/execute whole-run loop
    assert len(transport.draws) == 4 and set(transport.draws.values()) == {2}
    assert result['expected_episodes'] == result['finished_episodes'] == 4
    assert result['resource_censored_episodes'] == 0
    assert result['output_tokens'] == 4*(len(IMPORT.encode())+len(FINISH.encode()))
    reports = [read_record(p) for p in root.glob('episodes/*/report.json')]
    assert len(reports) == 4 and not any(r['stock_solved'] for r in reports)
    assert {(r['id'], r['episode_index']) for r in reports} == {
        ('a', 0), ('a', 1), ('b', 0), ('b', 1)}
    for p in root.glob('episodes/*'):
        assert len(list((p/'requests').glob('*.json'))) == 2
        generated = [read_record(g) for g in sorted((p/'generations').glob('*.json'))]
        assert [g['proposal']['raw'] for g in generated] == [IMPORT, FINISH]
        assert all(g['proposal']['token_ids'] for g in generated)
        assert all(g['proposal']['logprob'] == -2 for g in generated)
        assert read_record(p/'tree.json')['rejected_route_terminals']
        assert len(list((p/'journal').glob('*/events/*.json'))) == 3
    with pytest.raises(FileExistsError):
        run_episodes(rows(), {'fixture': {'C'}}, Tokenizer(), transport, root,
                     episode_count=2)


def test_prefix_aggregation_uses_episode_order_retains_pending_and_excludes_zero_step():
    from scripts.independent_route_runtime import summarize_reports
    targets = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
    reports = [
        {'id': 'a', 'episode_index': 0, 'stock_solved': False, 'already_in_stock': False,
         'final_status': 'rejected_action', 'search_complete': True, 'work': {'sampled_tokens': 8}},
        {'id': 'a', 'episode_index': 3, 'stock_solved': True, 'already_in_stock': False,
         'final_status': 'stock_solved', 'search_complete': True, 'work': {'sampled_tokens': 12}},
        {'id': 'b', 'episode_index': 0, 'stock_solved': False, 'already_in_stock': False,
         'final_status': 'resource_censored', 'search_complete': False, 'work': {'sampled_tokens': 5}},
        {'id': 'c', 'episode_index': 0, 'stock_solved': False, 'already_in_stock': True,
         'final_status': 'already_stock', 'search_complete': True, 'work': {'sampled_tokens': 0}}]
    result = summarize_reports(targets, reports, episode_counts=[1, 4], paired_ids=['a', 'c'])
    assert result['prefixes']['1']['stock_solved_targets'] == 0
    assert result['prefixes']['4']['stock_solved_targets'] == 1
    assert result['prefixes']['4']['target_denominator'] == 3
    assert result['prefixes']['4']['pending_or_censored_episodes'] == 9
    assert result['prefixes']['4']['already_stock_targets'] == 1
    assert result['prefixes']['4']['paired']['target_denominator'] == 2
    assert result['prefixes']['4']['paired']['stock_solved_targets'] == 1
    assert result['output_tokens'] == 25 and result['matched_compute_claim'] is False
    with pytest.raises(ValueError): summarize_reports(targets, reports+[reports[0]], episode_counts=[1, 4])


def test_model_transport_requires_explicit_authorization_before_loading_vllm():
    from scripts.independent_route_runtime import make_vllm_transport
    with pytest.raises(PermissionError, match='authorization'):
        make_vllm_transport('/not-model-weights')


def test_wrong_actor_is_rejected_before_vllm_import_or_gpu_initialization(tmp_path):
    from scripts.independent_route_runtime import make_vllm_transport
    (tmp_path/'adapter_model.safetensors').write_bytes(b'not the frozen actor')
    with pytest.raises(ValueError, match='actor weight'):
        make_vllm_transport(tmp_path, authorized_model_execution=True)


def test_decoder_error_keeps_raw_output_work_and_censors_instead_of_chemical_failure(tmp_path):
    from mechet.shared_electron_search import read_record
    from scripts.independent_route_runtime import run_episodes
    class MissingProbability(ScriptedTransport):
        def poll(self, timeout=0.):
            done = super().poll(timeout)
            if done: done[1].outputs[0].cumulative_logprob = None
            return done
    root = tmp_path/'broken-software-fixture'
    with pytest.raises(ValueError, match='probability'):
        run_episodes(rows()[:1], {'fixture': {'C'}}, Tokenizer(), MissingProbability(root),
                     root, episode_count=1, inflight=1, cpu_workers=1)
    raw_files = list(root.glob('episodes/*/raw_outputs/*.json'))
    assert len(raw_files) == 1
    raw = read_record(raw_files[0])
    assert raw['outputs'][0]['raw'] == IMPORT and raw['outputs'][0]['cumulative_logprob'] is None
    progress = read_record(root/'progress.json')
    assert progress['output_tokens'] == len(IMPORT.encode())
    assert progress['resource_censored_episodes'] == 1 and not progress['full_result_complete']
    report = read_record(next(root.glob('episodes/*/report.json')))
    assert report['final_status'] == 'resource_censored' and not report['stock_solved']


def test_preexisting_interrupt_keeps_full_denominator_and_never_submits(tmp_path):
    import threading
    from scripts.independent_route_runtime import run_episodes
    event = threading.Event(); event.set()
    root = tmp_path/'interrupted-fixture'; transport = ScriptedTransport(root)
    progress = run_episodes(rows(), {'fixture': {'C'}}, Tokenizer(), transport, root,
                            episode_count=2, stop_event=event)
    assert progress['expected_episodes'] == progress['pending_or_censored_episodes'] == 4
    assert progress['finished_percent'] == 0 and not progress['full_result_complete']
    assert not transport.draws


def test_foreign_engine_result_is_preserved_but_never_executed_on_another_episode(tmp_path):
    from mechet.shared_electron_search import read_record
    from scripts.independent_route_runtime import run_episodes
    class ForeignResult(ScriptedTransport):
        def poll(self, timeout=0.):
            done = super().poll(timeout)
            if done: done[1].request_id = 'foreign-engine-result'
            return done
    root = tmp_path/'foreign-response-fixture'
    with pytest.raises(ValueError, match='request identity'):
        run_episodes(rows()[:1], {'fixture': {'C'}}, Tokenizer(), ForeignResult(root),
                     root, episode_count=1, inflight=1, cpu_workers=1)
    raw = read_record(next(root.glob('episodes/*/raw_outputs/*.json')))
    assert raw['request_id'] == 'foreign-engine-result'
    report = read_record(next(root.glob('episodes/*/report.json')))
    assert report['draws'] == 0 and report['final_status'] == 'resource_censored'
