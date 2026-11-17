"""Scheduling and crash recovery checks, independent of GPU/model availability."""
from concurrent.futures import Future
import threading
from types import SimpleNamespace

import pytest


def test_derived_async_requests_wait_for_durable_execution_prefix():
    from mechet.route_async_collection import collect_async_route_groups
    pending=[];durable=[]
    class Episode:
        records=[]
        def next_decision(self):
            return SimpleNamespace(request_id=str(len(self.records))) if len(self.records)<2 else None
        def accept(self,receipt):self.records=[*self.records,receipt]
        def finish_report(self):return self.records
    class Sampler:
        cfg={'generation_batch_size':1}
        def submit(self,request):
            if request.request_id=='1':assert durable==[{'request_id':'0'}]
            result=Future();result.set_result({'request_id':request.request_id});return result
    def flush():
        durable.extend(pending);pending.clear()
    result=collect_async_route_groups([Episode()],Sampler(),workers=1,
        journal=pending.append,before_dispatch=flush)
    assert len(result[0])==2


def test_slow_sibling_does_not_block_chemistry_or_successor():
    from mechet.route_async_collection import collect_async_route_groups
    successor=threading.Event()
    class Episode:
        def __init__(self,uid):self.uid=uid;self.records=[];self.status=None
        def next_decision(self):
            return None if self.status else SimpleNamespace(request_id=self.uid+'/'+str(len(self.records)))
        def accept(self,value):
            self.records.append(value)
            if self.uid=='fast' and len(self.records)==1:successor.set()
            if self.uid=='slow' or len(self.records)==2:self.status='done'
        def finish_report(self):return self.uid
    class Sampler:
        cfg={'generation_batch_size':2}
        def submit(self,r):
            f=Future()
            if r.request_id=='slow/0':
                def complete():
                    assert successor.wait(2), 'early completion was held behind slow sibling'
                    f.set_result({'request_id':r.request_id})
                threading.Thread(target=complete).start()
            else:f.set_result({'request_id':r.request_id})
            return f
    records=[]
    assert collect_async_route_groups([Episode('fast'),Episode('slow')],Sampler(),
        workers=2,journal=records.append)==['fast','slow']
    assert sorted(v['request_id'] for v in records)==['fast/0','fast/1','slow/0']


def test_async_error_is_not_turned_into_a_chemical_failure():
    from mechet.route_async_collection import collect_async_route_groups
    class Episode:
        def next_decision(self):return SimpleNamespace(request_id='x')
    class Sampler:
        cfg={'generation_batch_size':1}
        def submit(self,r):
            f=Future();f.set_exception(RuntimeError('worker exited'));return f
    with pytest.raises(RuntimeError,match='worker exited'):
        collect_async_route_groups([Episode()],Sampler(),workers=1,journal=lambda _:None)


def test_group_commit_ack_follows_fsync_and_propagates_write_error(tmp_path,monkeypatch):
    from mechet.route_vllm_backend import DurableGroupCommit
    import os
    calls=[]
    monkeypatch.setattr(os,'fsync',lambda fd:calls.append(fd))
    with DurableGroupCommit(tmp_path/'wal',delay=.001) as writer:
        writer.append({'event':'complete','ids':[1,2]}).result(timeout=2)
        assert calls
        assert '"complete"' in (tmp_path/'wal').read_text()
    def fail(_fd):raise OSError('storage failed')
    monkeypatch.setattr(os,'fsync',fail)
    with DurableGroupCommit(tmp_path/'failed',delay=.001) as writer:
        with pytest.raises(OSError,match='storage failed'):
            writer.append({'event':'complete'}).result(timeout=2)
        with pytest.raises(OSError,match='storage failed'):
            writer.append({'event':'next'}).result(timeout=2)


def test_recovery_preserves_actual_logprobs_and_rejects_conflicting_uid(tmp_path):
    from mechet.route_vllm_backend import recover_native_receipts
    import json
    from test_route_vllm_protocol import response,build_wire
    wire=build_wire();completion=response()
    path=tmp_path/'wal'
    events=[dict(event='vllm-submitted',request=wire),
            dict(event='vllm-completed',request=wire,receipt=completion)]
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    recovered=recover_native_receipts(path,wire['behavior_version'])
    assert recovered[wire['request_id']]['behavior_logprobs']==[-.2,-.3]
    events.append(dict(event='vllm-completed',request=wire,
                       receipt=response(behavior_logprobs=[-.4,-.3])))
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError,match='CONFLICTING'):
        recover_native_receipts(path,wire['behavior_version'])


def test_submitted_without_output_is_censored_without_resampling(tmp_path):
    from mechet.route_vllm_backend import recover_native_receipts
    import json
    from test_route_vllm_protocol import build_wire
    wire=build_wire();path=tmp_path/'wal'
    path.write_text(json.dumps(dict(event='vllm-submitted',request=wire))+'\n')
    value=recover_native_receipts(path,wire['behavior_version'])[wire['request_id']]
    assert value['resource_error']=='INTERRUPTED_GENERATION_NO_OUTPUT'
    assert value['generated_token_ids']==[]


def test_native_recovery_never_accepts_a_changed_prompt_or_weight_version(tmp_path):
    from mechet.route_vllm_backend import recover_native_receipts
    from test_route_vllm_protocol import build_wire
    import json
    wire=build_wire();changed=dict(wire,prompt_token_ids=[2,3,4])
    path=tmp_path/'wal';path.write_text('\n'.join(json.dumps(dict(
        event='vllm-submitted',request=r)) for r in (wire,changed))+'\n')
    with pytest.raises(ValueError,match='CONFLICTING'):
        recover_native_receipts(path,wire['behavior_version'])
    with pytest.raises(ValueError,match='VERSION'):
        recover_native_receipts(path,'another')


def test_process_transport_multiplexes_out_of_order_and_propagates_exit(tmp_path):
    from mechet.route_vllm_backend import VllmProcess
    import os,sys
    child=tmp_path/'server.py'
    child.write_text('''import json,sys
held=None
for line in sys.stdin:
 m=json.loads(line)
 if m['operation']=='hold':held=m;continue
 if m['operation']=='exit':sys.exit(2)
 print(json.dumps(dict(rpc_id=m['rpc_id'],result=m['payload'])),flush=True)
 if held:
  print(json.dumps(dict(rpc_id=held['rpc_id'],result=held['payload'])),flush=True)
  held=None
''')
    process=VllmProcess([sys.executable,'-S','-B',str(child)],
        environment=dict(os.environ),log_path=tmp_path/'server.log',timeout=5)
    try:
        slow=process.rpc('hold',{'uid':'first'})
        fast=process.rpc('now',{'uid':'second'})
        assert fast.result(timeout=5)=={'uid':'second'}
        assert slow.result(timeout=5)=={'uid':'first'}
        with pytest.raises(RuntimeError,match='PROCESS_EOF'):
            process.rpc('exit',{}).result(timeout=5)
        with pytest.raises(RuntimeError,match='PROCESS_EOF'):
            process.rpc('after_exit',{}).result(timeout=5)
    finally:process.close()


def test_blocked_request_write_never_blocks_delivery_of_completed_request():
    from mechet.route_vllm_backend import VllmProcess
    import json,queue
    blocked=threading.Event();release=threading.Event();existing=Future()
    class Input:
        def write(self,_):blocked.set();release.wait(2)
        def flush(self):pass
    transport=VllmProcess.__new__(VllmProcess)
    transport.lock=threading.Lock();transport.write_lock=threading.Lock()
    transport.pending={'existing':existing};transport.counter=0;transport.error=None
    transport.deadlines={};transport.outbound=queue.Queue();transport.timeout=2
    transport.process=SimpleNamespace(stdin=Input(),stdout=[json.dumps(
        dict(rpc_id='existing',result='completed'))])
    writer=threading.Thread(target=transport._write_loop);writer.start()
    transport.rpc('generate',{})
    assert blocked.wait(1)
    reader=threading.Thread(target=transport._read);reader.start()
    try:assert existing.result(timeout=.5)=='completed'
    finally:release.set();transport.outbound.put(None);writer.join();reader.join()


def test_ready_timeout_reaps_owned_child(tmp_path,monkeypatch):
    from mechet.route_vllm_backend import VllmProcess
    import os,sys,subprocess
    actual=subprocess.Popen;children=[]
    def start(*args,**kwargs):
        child=actual(*args,**kwargs);children.append(child);return child
    monkeypatch.setattr(subprocess,'Popen',start)
    try:
        with pytest.raises(TimeoutError):
            VllmProcess([sys.executable,'-S','-B','-c','import time;time.sleep(10)'],
                environment=dict(os.environ),log_path=tmp_path/'timeout.log',timeout=.1)
        assert children[0].poll() is not None, 'failed startup leaked its child'
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait()


def test_wedged_reader_times_out_including_large_request_write(tmp_path):
    from mechet.route_vllm_backend import VllmProcess
    import os,sys,time
    script="import json,sys,time;m=json.loads(sys.stdin.readline());print(json.dumps(dict(rpc_id=m['rpc_id'],result=True)),flush=True);time.sleep(10)"
    process=VllmProcess([sys.executable,'-S','-B','-c',script],environment=dict(os.environ),
        log_path=tmp_path/'wedged.log',timeout=.3)
    began=time.monotonic()
    try:
        future=process.rpc('generate',{'large_prompt':'x'*1024*1024})
        assert time.monotonic()-began<.2
        with pytest.raises(TimeoutError,match='INCLUDING_PIPE_WRITE'):
            future.result(timeout=2)
    finally:process.close(force=True)
    assert time.monotonic()-began<3
