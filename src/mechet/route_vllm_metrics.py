"""Native-engine observations, distinct from application Future concurrency.

Prefill/decode times are summed request latencies, not nonoverlapping GPU kernel
time. Running/decode batches come from native SchedulerStats/IterationStats.
"""
from collections import Counter,deque,OrderedDict
import math
from threading import RLock


def _number(value,name,*,integer=False):
    if (type(value) not in ((int,) if integer else (int,float))
            or not math.isfinite(value) or value<0):
        raise ValueError('VLLM_ENGINE_STAT_INVALID:'+name)
    return value


class NativeEngineMetrics:
    def __init__(self):
        self.lock=RLock();self.start_version(None)

    def start_version(self,version):
        with self.lock:
            self.version=version
            self.totals=Counter(dict.fromkeys(('native_output_observations','native_iteration_observations',
                'actual_running_requests_peak','actual_running_requests_sum','actual_waiting_requests_peak',
                'kv_cache_usage_peak','prefix_tokens_queried','prefix_tokens_hit','prefill_tokens_total',
                'prefill_tokens_computed','prefill_tokens_cached','generated_tokens','actual_decode_batch_sum',
                'actual_decode_batch_peak','actual_prefill_requests','request_prefill_seconds',
                'request_decode_seconds','request_queue_seconds','finished_requests',
                'unconsumed_finished_stats_evicted'),0))
            self.finished=OrderedDict()
            self.decode_samples=deque(maxlen=4096);self.running_samples=deque(maxlen=4096)

    def record(self,scheduler,iteration):
        with self.lock:
            self.totals['native_output_observations']+=1
            if scheduler is not None:
                running=_number(getattr(scheduler,'num_running_reqs',0),'running',integer=True)
                waiting=_number(getattr(scheduler,'num_waiting_reqs',0),'waiting',integer=True)
                kv=_number(getattr(scheduler,'kv_cache_usage',0.),'kv_usage')
                if kv>1:raise ValueError('VLLM_ENGINE_STAT_INVALID:kv_usage')
                self.totals['actual_running_requests_peak']=max(self.totals['actual_running_requests_peak'],running)
                self.totals['actual_running_requests_sum']+=running
                self.totals['actual_waiting_requests_peak']=max(self.totals['actual_waiting_requests_peak'],waiting)
                self.totals['kv_cache_usage_peak']=max(self.totals['kv_cache_usage_peak'],kv)
                self.running_samples.append(running)
                prefix=getattr(scheduler,'prefix_cache_stats',None)
                if prefix is not None:
                    queries=_number(prefix.queries,'prefix_queries',integer=True)
                    hits=_number(prefix.hits,'prefix_hits',integer=True)
                    if hits>queries:raise ValueError('VLLM_ENGINE_STAT_INVALID:prefix_hits')
                    self.totals['prefix_tokens_queried']+=queries;self.totals['prefix_tokens_hit']+=hits
            if iteration is None:return
            self.totals['native_iteration_observations']+=1
            prompt=iteration.prompt_token_stats
            for name,field in [('prefill_tokens_total','total'),('prefill_tokens_computed','computed'),
                               ('prefill_tokens_cached','cached_tokens')]:
                self.totals[name]+=_number(getattr(prompt,field),field,integer=True)
            if prompt.computed+prompt.cached_tokens!=prompt.total:
                raise ValueError('VLLM_ENGINE_STAT_INVALID:prefill_conservation')
            self.totals['generated_tokens']+=_number(iteration.num_generation_tokens,'generated',integer=True)
            decode=len(iteration.inter_token_latencies_iter)
            self.totals['actual_decode_batch_sum']+=decode
            self.totals['actual_decode_batch_peak']=max(self.totals['actual_decode_batch_peak'],decode)
            self.totals['actual_prefill_requests']+=len(iteration.time_to_first_tokens_iter)
            self.decode_samples.append(decode)
            for request in iteration.finished_requests:
                values={k:_number(getattr(request,k),k) for k in
                        ('prefill_time','decode_time','queued_time','e2e_latency')}
                self.totals['request_prefill_seconds']+=values['prefill_time']
                self.totals['request_decode_seconds']+=values['decode_time']
                self.totals['request_queue_seconds']+=values['queued_time']
                self.totals['finished_requests']+=1
                uid=getattr(request,'request_id',None)
                if uid:
                    self.finished[uid]=values
                    if len(self.finished)>8192:
                        self.finished.popitem(last=False);self.totals['unconsumed_finished_stats_evicted']+=1

    def take_finished(self,uid):
        with self.lock:return self.finished.pop(uid,None)

    def snapshot(self):
        with self.lock:
            result=dict(self.totals,behavior_version=self.version,
                scope='native engine observations; request latency sums are not GPU kernel time',
                decode_batch_observation='non-prefill outputs in each native engine output interval',
                quantile_sample_limit=4096)
            n=self.totals['native_iteration_observations']
            result['actual_decode_batch_mean']=self.totals['actual_decode_batch_sum']/n if n else 0.
            return result
