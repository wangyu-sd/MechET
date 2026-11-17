"""Isolated vLLM transport and durable native probability receipts.

No Torch/vLLM import is permitted here: this module lives in the HF learner.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict,is_dataclass
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

from .route_policy_sampling import read_owned_journal
from .route_vllm_protocol import build_request,validate_completion


class DurableGroupCommit:
    """Acknowledge a group only after fsync; poison subsequent writes on failure."""
    def __init__(self,path,*,delay=.02,max_rows=32):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.delay,self.max_rows=delay,max_rows
        self.queue=queue.Queue();self.error=None;self.closed=False
        self.stats_lock=threading.Lock()
        self.stats=dict(rows=0,bytes=0,fsync_barriers=0,write_seconds=0.,fsync_seconds=0.)
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def append(self,value):
        future=Future()
        if self.closed:future.set_exception(RuntimeError('JOURNAL_CLOSED'))
        else:self.queue.put((value,future))
        return future

    def _run(self):
        while True:
            first=self.queue.get()
            if first is None:return
            batch=[first];deadline=time.monotonic()+self.delay;stop=False
            while len(batch)<self.max_rows:
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                try:item=self.queue.get(timeout=remaining)
                except queue.Empty:break
                if item is None:stop=True;break
                batch.append(item)
            try:
                if self.error:raise self.error
                began=time.monotonic()
                payload=''.join(json.dumps(value,ensure_ascii=False)+'\n' for value,_ in batch)
                with self.path.open('a') as sink:
                    sink.write(payload);sink.flush();written=time.monotonic()
                    os.fsync(sink.fileno());durable=time.monotonic()
                with self.stats_lock:
                    self.stats['rows']+=len(batch);self.stats['bytes']+=len(payload.encode('utf-8'))
                    self.stats['fsync_barriers']+=1;self.stats['write_seconds']+=written-began
                    self.stats['fsync_seconds']+=durable-written
            except Exception as exc:
                self.error=exc
                for _,future in batch:future.set_exception(exc)
            else:
                for _,future in batch:future.set_result(None)
            if stop:return

    def close(self):
        if not self.closed:
            self.closed=True;self.queue.put(None);self.thread.join()
    def metrics(self):
        with self.stats_lock:return dict(self.stats)
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def recover_native_receipts(path,version,*,events=None):
    requests={};receipts={}
    for event in read_owned_journal(path) if events is None else events:
        if event['event'] not in ('vllm-submitted','vllm-completed'):
            raise ValueError('NON_NATIVE_EVENT_IN_VLLM_JOURNAL')
        request=event['request'];uid=request['request_id']
        if request['behavior_version']!=version:raise ValueError('RECOVERY_BEHAVIOR_VERSION_MISMATCH')
        if uid in requests and requests[uid]!=request:raise ValueError('CONFLICTING_VLLM_REQUEST')
        requests[uid]=request
        if event['event']=='vllm-completed':
            value=validate_completion(event['receipt'],request)
            if uid in receipts and receipts[uid]!=value:raise ValueError('CONFLICTING_VLLM_COMPLETION')
            receipts[uid]=value
    for uid,request in requests.items():
        if uid not in receipts:
            receipts[uid]=dict(request_id=uid,behavior_version=version,
                prompt_token_ids=request['prompt_token_ids'],raw='',generated_token_ids=[],
                behavior_logprobs=[],resource_error='INTERRUPTED_GENERATION_NO_OUTPUT')
    return receipts


class VllmProcess:
    """Persistent multiplexed JSON-lines child; stdout belongs to protocol only."""
    def __init__(self,command,*,environment,log_path,timeout=1800.,startup_timeout=None):
        startup_timeout=timeout if startup_timeout is None else startup_timeout
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
               for v in (timeout,startup_timeout)):
            raise ValueError('VLLM_INVALID_PROCESS_TIMEOUT')
        self.log=Path(log_path).open('ab');self.timeout=timeout
        self.process=subprocess.Popen(command,env=environment,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,start_new_session=True)
        self.lock=threading.Lock();self.write_lock=threading.Lock()
        self.pending={};self.counter=0;self.error=None
        self.deadlines={};self.outbound=queue.Queue(maxsize=128);self.stopping=threading.Event()
        self.writer=threading.Thread(target=self._write_loop,daemon=True);self.writer.start()
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
        self.watchdog=threading.Thread(target=self._watch,daemon=True);self.watchdog.start()
        try:self.rpc('ready',{},timeout=startup_timeout).result(timeout=startup_timeout)
        except BaseException:
            self.close(force=True)
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                value=json.loads(line)
                with self.lock:
                    future=self.pending.pop(value['rpc_id']);self.deadlines.pop(value['rpc_id'],None)
                if 'error' in value:future.set_exception(RuntimeError(value['error']))
                else:future.set_result(value['result'])
            raise RuntimeError('VLLM_PROCESS_EOF: inspect isolated runtime log')
        except Exception as exc:
            self._fail_all(exc)

    def _fail_all(self,exc):
        with self.lock:
            self.error=exc;pending=list(self.pending.values());self.pending.clear();self.deadlines.clear()
        for future in pending:future.set_exception(exc)

    def _write_loop(self):
        while True:
            item=self.outbound.get()
            if item is None:return
            try:
                self.process.stdin.write(item);self.process.stdin.flush()
            except Exception as exc:
                self._fail_all(exc);return

    def _watch(self):
        while not self.stopping.wait(.05):
            with self.lock:expired=any(v<=time.monotonic() for v in self.deadlines.values())
            if expired:
                self._fail_all(TimeoutError('VLLM_REQUEST_TIMEOUT_INCLUDING_PIPE_WRITE'))
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                return

    def rpc(self,operation,payload,*,timeout=None):
        timeout=self.timeout if timeout is None else timeout
        future=Future()
        with self.lock:
            if self.error:future.set_exception(self.error);return future
            if len(self.pending)>=128:
                future.set_exception(RuntimeError('VLLM_TRANSPORT_CAPACITY'));return future
            self.counter+=1;uid=str(self.counter);self.pending[uid]=future
            self.deadlines[uid]=time.monotonic()+timeout
        try:
            # Return immediately: deadlines cover blocked writes as well as GPU
            # work. Only the daemon sender ever touches the blocking input pipe.
            self.outbound.put_nowait(json.dumps(dict(rpc_id=uid,operation=operation,payload=payload))+'\n')
        except Exception as exc:
            with self.lock:
                pending=self.pending.pop(uid,None);self.deadlines.pop(uid,None)
            if pending is not None:pending.set_exception(exc)
        return future

    def close(self,*,force=False):
        if getattr(self,'closed',False):return
        self.closed=True;self.stopping.set()
        self._fail_all(RuntimeError('VLLM_PROCESS_CLOSED'))
        try:
            # The dedicated process group contains this engine's owned workers,
            # including when its immediate parent failed before readiness.
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.process.wait(timeout=5)
        finally:
            try:os.killpg(self.process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:self.outbound.put_nowait(None)
            except queue.Full:pass
            self.writer.join(timeout=1);self.reader.join(timeout=1);self.watchdog.join(timeout=1)
            self.process.stdin.close();self.process.stdout.close();self.log.close()


class VllmRouteSampler:
    def __init__(self,process,tokenizer,cfg,raw_path):
        self.process,self.tokenizer,self.cfg=process,tokenizer,cfg
        self.raw_path=Path(raw_path);self.writer=DurableGroupCommit(raw_path)
        self.pool=ThreadPoolExecutor(max_workers=cfg['generation_batch_size'])

    def submit(self,request):return self.pool.submit(self._generate_one,request)

    def _generate_one(self,request):
        from .route_policy_protocol import route_tool_prefix
        from .route_action_feedback import tools_for_request
        prompt=route_tool_prefix(self.tokenizer,request.messages,tools_for_request(request),self.cfg)
        ids=self.tokenizer(prompt,add_special_tokens=False)['input_ids']
        try:wire=build_request(request,ids,dict(self.cfg,eos_token_id=self.tokenizer.eos_token_id))
        except ValueError as exc:
            if str(exc)!='VLLM_CONTEXT_CAPACITY':raise
            return dict(request_id=request.request_id,behavior_version=request.behavior_version,
                prompt_token_ids=ids,raw='',generated_token_ids=[],behavior_logprobs=[],
                resource_error='PHYSICAL_CONTEXT_LIMIT')
        self.writer.append(dict(event='vllm-submitted',request=wire,
            decision_request=asdict(request) if is_dataclass(request) else deepcopy(vars(request)))).result()
        result=self.process.rpc('generate',wire).result(timeout=self.process.timeout)
        value=validate_completion(result,wire)
        value['raw']=self.tokenizer.decode(value['generated_token_ids'],skip_special_tokens=False)
        # Completion IDs and actual engine likelihoods share ONE durable record.
        self.writer.append(dict(event='vllm-completed',request=wire,receipt=value)).result()
        return value

    def close(self):
        self.pool.shutdown(wait=True);self.writer.close()


def start_vllm_process(cfg):
    """Create once per learner rank, before copies of validation config exist."""
    if '_vllm_process' in cfg:return cfg['_vllm_process']
    import atexit
    from .route_training_contract import atomic_json
    from .route_vllm_environment import child_environment
    rank=int(os.environ.get('LOCAL_RANK','0'))
    directory=Path(cfg['output_dir'])/'runtime'/f'vllm-rank{rank:02d}'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'engine.json'
    engine_cfg={k:cfg[k] for k in ('base_path','max_context','generation_batch_size','vllm_runtime')}
    engine_cfg.update({k:cfg[k] for k in ('base_manifest','_sha','policy_precision',
                         'public_observation_contract','acceleration') if k in cfg})
    engine_cfg['physical_gpu_rank']=rank
    atomic_json(path,engine_cfg)
    environment=child_environment(cfg,rank)
    environment['MECHET_ROUTE_VLLM_CONFIG']=str(path)
    process=VllmProcess([cfg['vllm_runtime']['python'],
        str(Path(__file__).resolve().parents[2]/'scripts/serve_route_vllm.py'),
        '--config',str(path)],environment=environment,log_path=directory/'engine.log',
        timeout=cfg['vllm_runtime']['request_timeout_seconds'],
        startup_timeout=cfg['vllm_runtime'].get('startup_timeout_seconds',
                                               cfg['vllm_runtime']['request_timeout_seconds']))
    cfg['_vllm_process']=process;atexit.register(process.close)
    return process


def bind_rollout_adapter(cfg,version):
    from .route_vllm_adapter import prepare_text_adapter
    sha=version.rsplit(':',1)[-1]
    source=(cfg['parent'] if version.startswith('validation000000:')
            else cfg['_rollout_adapter_path'])
    rank=int(os.environ.get('LOCAL_RANK','0'))
    path=Path(cfg['output_dir'])/'runtime'/f'vllm-rank{rank:02d}'/'adapters'/sha
    mapped=prepare_text_adapter(source,path,sha)
    process=start_vllm_process(cfg)
    return process.rpc('load_adapter',dict(version=version,path=str(mapped),
        source_sha256=sha)).result(timeout=process.timeout)
