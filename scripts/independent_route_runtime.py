"""Prepared independent-route actor transport; no model execution on import.

Private episodes share only read-only weights, prefix computation and stock.
They do not share sampled actions, execution caches, frontiers or statistics.
The low-level scheduler accepts software fixtures; a formal CLI must validate
the prepared full-population manifest before calling it. No run is submitted
by this module, and model loading requires separate explicit authorization.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import math
import os
from pathlib import Path
import threading
import time

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.dynamic_electron_tree import Proposal
from mechet.shared_electron_search import SearchJournal, durable_json
from mechet.analysis.literature_sources import file_sha256
from scripts.build_natural_language_event_sft import TOOLS
from scripts.eval_natural_language_event_local import prediction_call
from scripts.independent_route_control import IndependentRouteEpisode


_SAMPLING_KEYS = ('n', 'max_tokens', 'temperature', 'top_p', 'top_k',
                  'repetition_penalty', 'presence_penalty', 'frequency_penalty',
                  'skip_special_tokens', 'logprobs', 'min_tokens', 'seed')


def sampling_kwargs(request):
    """Only documented sampler fields reach vLLM, never controller metadata."""
    return {key: request[key] for key in _SAMPLING_KEYS}


def prepare_request(episode, tokenizer, *, max_model_len=8192):
    decision = episode.next_request()
    if decision is None:
        return None
    prompt = render_qwen_sft_tool_prefix(tokenizer, decision['messages'], tools=TOOLS)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    room = max_model_len-len(ids)
    if room < 2:
        episode.censor('CONTEXT_MEMORY_BUDGET')
        return None
    return {'episode_uid': episode.uid, 'draw_index': episode.draws,
        'context': decision['context_key'], 'messages': decision['messages'],
        'prompt': prompt, 'prompt_ids': ids, 'completion_room': room,
        'n': 1, 'max_tokens': min(decision['completion_budget'], room),
        'temperature': 1, 'top_p': 1, 'top_k': -1, 'repetition_penalty': 1,
        'presence_penalty': 0, 'frequency_penalty': 0, 'skip_special_tokens': False,
        'logprobs': 0, 'min_tokens': 0, 'seed': decision['seed']}


def decode_completion(result, tokenizer):
    if len(result.outputs) != 1:
        raise ValueError('independent decision requires exactly one generated candidate')
    generated = result.outputs[0]
    ids = list(generated.token_ids)
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    logprob = generated.cumulative_logprob
    if logprob is None or not math.isfinite(logprob):
        raise ValueError('missing/nonfinite unwarped model log probability')
    name, arguments, error = prediction_call(raw, tokenizer)
    return {'request_id': str(result.request_id),
        'cached_prefix_tokens': getattr(result, 'num_cached_tokens', None),
        'finish_reason': generated.finish_reason,
        'proposal': asdict(Proposal(name, arguments, raw, float(logprob), len(ids),
                                   str(error or ''), generated.finish_reason == 'length', ids))}


def make_vllm_transport(adapter, *, authorized_model_execution=False, inflight=128, seed=17):
    """Construct the archived V0/TP1 backend only after separate authorization."""
    if not authorized_model_execution:
        raise PermissionError('separate model-execution authorization is required')
    if file_sha256(Path(adapter)/'adapter_model.safetensors') != (
            '40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75'):
        raise ValueError('frozen actor weight checksum differs')
    if os.environ.get('VLLM_USE_V1', '0') != '0':
        raise ValueError('archived vLLM V0 engine required')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('one explicit GPU per independent worker required')
    os.environ['VLLM_USE_V1'] = '0'
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from scripts.rolling_vllm_search import ContinuousBatchEngine
    from scripts.vllm_safe_lora_expand import install_safe_lora_expand
    if vllm.__version__.split('+')[0] != '0.8.5':
        raise ValueError('archived vLLM0.8.5 runtime required')
    runtime = install_safe_lora_expand(vllm.__version__)
    engine = LLM(model='Qwen/Qwen3-0.6B', tokenizer='Qwen/Qwen3-0.6B',
        revision='c1899de289a04d12100db370d81485cdf75e47ca', trust_remote_code=True,
        dtype='bfloat16', tensor_parallel_size=1, gpu_memory_utilization=.82,
        max_model_len=8192, max_num_seqs=256, enable_prefix_caching=True,
        enable_lora=True, max_lora_rank=16, max_loras=1, enforce_eager=True,
        num_scheduler_steps=1, seed=seed)
    pipe = ContinuousBatchEngine(engine.llm_engine,
        params_factory=lambda request: SamplingParams(**sampling_kwargs(request)),
        lora=LoRARequest('frozen_flowER_stage2_independent', 1, str(Path(adapter).resolve())),
        max_inflight=inflight)
    return engine.get_tokenizer(), pipe, runtime


def summarize_reports(rows, reports, *, episode_counts=(1, 4, 10), paired_ids=None):
    """Episode-prefix unions, never NLL ranking; missing work stays missing."""
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('nonempty unique target population required')
    if (not episode_counts or any(type(k) is not int or k < 1 for k in episode_counts)
            or len(set(episode_counts)) != len(episode_counts)):
        raise ValueError('positive distinct episode prefixes required')
    paired = set(ids if paired_ids is None else paired_ids)
    if not paired <= set(ids):
        raise ValueError('paired target outside full population')
    indexed = {}
    for r in reports:
        key = r['id'], r['episode_index']
        if (key[0] not in ids or type(key[1]) is not int or not 0 <= key[1] < max(episode_counts)
                or key in indexed or (r['stock_solved'] and r['already_in_stock'])):
            raise ValueError('duplicate, foreign or inconsistent independent episode report')
        indexed[key] = r

    def prefix(k, selected):
        observed = [r for (identifier, ep), r in indexed.items() if identifier in selected and ep < k]
        finished = [r for r in observed if r['search_complete']]
        return {'target_denominator': len(selected), 'expected_episodes': len(selected)*k,
            'reported_episodes': len(observed), 'finished_episodes': len(finished),
            'pending_or_censored_episodes': len(selected)*k-len(finished),
            'stock_solved_targets': len({r['id'] for r in observed if r['stock_solved'] and not r['already_in_stock']}),
            'already_stock_targets': len({r['id'] for r in observed if r['already_in_stock']}),
            'resource_censored_episodes': sum(r['final_status'] == 'resource_censored' for r in observed)}

    result = {'schema': 'independent_route_prefix_summary_v1', 'prefixes': {},
        'output_tokens': sum(r.get('observed_generation_work', r['work'])['sampled_tokens'] for r in reports),
        'matched_compute_claim': False, 'chemical_plausibility': 'not_independently_verified',
        'ranking': 'none; union by predeclared episode index'}
    for k in episode_counts:
        result['prefixes'][str(k)] = {**prefix(k, set(ids)), 'paired': prefix(k, paired)}
    return result


def run_episodes(rows, stocks, tokenizer, transport, output_root, *, episode_count=10,
                 inflight=128, cpu_workers=16, seed=17, max_model_len=8192,
                 heartbeat_seconds=45, stop_event=None):
    """Rolling independent episodes with durable before-execution receipts.

    The caller supplies a continuous-batching actor transport. Only external
    model decoding is replaceable in CPU unit tests; tool execution, full
    prompts, journals, stock closure and filesystem persistence remain real.
    Resource interruption is censored and cannot be a completed experiment.
    """
    if any(type(v) is not int or v < 1 for v in (episode_count, inflight, cpu_workers, max_model_len)):
        raise ValueError('positive episode/concurrency/context capacities required')
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError('positive finite heartbeat interval required')
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique target population required')
    if any(r['cohort'] not in stocks for r in rows):
        raise ValueError('missing cohort stock')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    stop_event = stop_event or threading.Event()
    jobs = iter((row, ep) for ep in range(episode_count) for row in rows)
    active, preparing, executing, gpu = {}, {}, {}, {}
    ready, reports = deque(), []
    exhausted = False
    started = time.monotonic()
    last_progress = -math.inf
    expected = len(rows)*episode_count
    aborted = 0
    observed_work = Counter()
    by_episode_work = {}
    pool = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix='independent-tools')
    contract = {'algorithm': 'independent_complete_route_v1', 'shared_frontier': False,
                'shared_policy_samples': False, 'reference_visible_to_policy': False,
                'seed': seed, 'max_model_len': max_model_len}

    def finish(uid):
        episode, journal = active.pop(uid)
        path = root/'episodes'/uid
        durable_json(path/'tree.json', episode.network.to_dict())
        report = {**episode.report(), 'cohort': next(r['cohort'] for r in rows if r['id'] == episode.target_id),
                  'observed_generation_work': dict(by_episode_work[uid])}
        durable_json(path/'report.json', report)
        reports.append(report)

    def prepare(uid):
        episode, journal = active[uid]
        request = prepare_request(episode, tokenizer, max_model_len=max_model_len)
        if request is not None:
            durable_json(root/'episodes'/uid/'requests'/f"{request['draw_index']:09d}.json", request)
        elif episode.final_status == 'resource_censored':
            for key, context in episode.network.contexts.items():
                if context.resource_closed:
                    journal.record(episode.network, key, [])
        return request

    def execute(uid, request, result, expected_request_id):
        episode, journal = active[uid]
        raw = {'request_id': str(result.request_id), 'episode_uid': uid,
            'draw_index': request['draw_index'], 'context': request['context'],
            'cached_prefix_tokens': getattr(result, 'num_cached_tokens', None), 'outputs': []}
        for generated in result.outputs:
            lp = generated.cumulative_logprob
            raw['outputs'].append({'token_ids': list(generated.token_ids),
                'raw': tokenizer.decode(list(generated.token_ids), skip_special_tokens=False),
                'cumulative_logprob': lp if lp is None or math.isfinite(lp) else str(lp),
                'finish_reason': generated.finish_reason})
        durable_json(root/'episodes'/uid/'raw_outputs'/f"{request['draw_index']:09d}.json", raw)
        if str(result.request_id) != str(expected_request_id):
            raise ValueError('actor response request identity differs from pending episode')
        record = decode_completion(result, tokenizer)
        if record['proposal']['tokens'] > request['max_tokens']:
            raise ValueError('actor returned more tokens than requested')
        # Raw output is durable even if the subsequent parser/executor fails.
        durable_json(root/'episodes'/uid/'generations'/f"{request['draw_index']:09d}.json", record)
        edges = episode.accept(request['context'], Proposal(**record['proposal']),
            completion_room=request['completion_room'], max_tokens=request['max_tokens'])
        journal.record(episode.network, request['context'], edges)

    def progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and now-last_progress < heartbeat_seconds:
            return
        last_progress = now
        censored = sum(r['final_status'] == 'resource_censored' for r in reports)
        value = {'phase': 'independent-complete-route', 'target_denominator': len(rows),
            'episodes_per_target': episode_count, 'expected_episodes': expected,
            'reported_episodes': len(reports), 'finished_episodes': len(reports)-censored,
            'finished_percent': 100*(len(reports)-censored)/expected,
            'resource_censored_episodes': censored, 'pending_or_censored_episodes': expected-len(reports)+censored,
            'GPU_inflight_requests': len(gpu), 'CPU_preparing': len(preparing), 'CPU_executing': len(executing),
            'output_tokens': observed_work['sampled_tokens'],
            'prompt_tokens': observed_work['prompt_tokens'], 'model_requests': observed_work['model_requests'],
            'aborted_inflight_requests': aborted, 'elapsed_seconds': now-started,
            'full_result_complete': len(reports)-censored == expected,
            'matched_compute_claim': False}
        durable_json(root/'progress.json', value)
        print(value, flush=True)
        return value

    error = None
    try:
        progress(True)
        while not exhausted or active:
            if stop_event.is_set():
                break
            while len(active) < inflight and not exhausted:
                try:
                    row, ep = next(jobs)
                except StopIteration:
                    exhausted = True
                    break
                episode = IndependentRouteEpisode(row, stocks[row['cohort']], episode_index=ep, seed=seed)
                journal = SearchJournal(root/'episodes'/episode.uid/'journal', episode.uid, contract)
                journal.initialize(episode.network)
                active[episode.uid] = episode, journal
                by_episode_work[episode.uid] = Counter(sampled_tokens=0, prompt_tokens=0, model_requests=0)
                ready.append(episode.uid)
            while ready and len(preparing)+len(executing) < 2*cpu_workers:
                uid = ready.popleft()
                preparing[pool.submit(prepare, uid)] = uid
            for future in [f for f in preparing if f.done()]:
                uid = preparing.pop(future)
                request = future.result()
                if request is None:
                    finish(uid)
                else:
                    identifier = transport.submit(request)
                    if identifier in gpu:
                        raise ValueError('duplicate in-flight model request ID')
                    gpu[identifier] = uid, request
                    work = {'prompt_tokens': len(request['prompt_ids']), 'model_requests': 1}
                    observed_work.update(work); by_episode_work[uid].update(work)
            done = transport.poll(timeout=.01)
            if done is not None:
                identifier, result = done
                uid, request = gpu.pop(identifier)
                work = {'sampled_tokens': sum(len(g.token_ids) for g in result.outputs)}
                observed_work.update(work); by_episode_work[uid].update(work)
                executing[pool.submit(execute, uid, request, result, identifier)] = uid
            for future in [f for f in executing if f.done()]:
                uid = executing.pop(future)
                future.result()
                ready.append(uid)
            progress()
            if done is None:
                stop_event.wait(.001)
    except BaseException as exc:
        error = exc
    finally:
        # Do not mutate/censor a controller while a tool thread owns it.
        pool.shutdown(wait=True)
        for future in list(preparing)+list(executing):
            try:
                future.result()
            except BaseException as exc:
                if error is None:
                    error = exc
        aborted = len(gpu)
        # This scheduler owns transport shutdown as well as tool-thread drain.
        # Returned-but-invalid receipts are preserved; unfinished decoder work
        # has unknown token cost and is explicitly flagged, not assumed free.
        try:
            close = getattr(transport, 'close', None)
            if close is not None:
                close()
        except BaseException as exc:
            if error is None:
                error = exc
        if active:
            for uid, (episode, _) in list(active.items()):
                episode.censor('RUNTIME_ERROR' if error is not None else 'RUN_INTERRUPTED')
                finish(uid)
        final = progress(True)
        final['unobserved_aborted_output_work'] = bool(aborted)
        durable_json(root/'progress.json', final)
    if error is not None:
        raise error
    return final
