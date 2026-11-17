"""Round-scoped request RPC: CPU owners hold chemistry; GPUs hold actors."""
from concurrent.futures import Future
from dataclasses import asdict
from multiprocessing.managers import BaseManager
from threading import Condition,Thread
import secrets

from .external_route_data import signature
from .route_policy_episode import DecisionRequest


def broker_server(broker,host):
    # A thread, not a fork after CUDA initialization. Only the allocated job's
    # workers receive the ephemeral authenticated socket address.
    class Manager(BaseManager):pass
    Manager.register('broker',callable=lambda:broker)
    key=secrets.token_bytes(32);manager=Manager(address=(host,0),authkey=key)
    server=manager.get_server()
    def serve():
        try:server.serve_forever()
        except SystemExit:pass
    Thread(target=serve,daemon=True).start()
    return server,server.address,key


def connect_broker(address,key):
    class Manager(BaseManager):pass
    Manager.register('broker')
    manager=Manager(address=address,authkey=key);manager.connect()
    return manager.broker()


class LockedTokenizer:
    """Fast-tokenizer padding configuration is mutable, even in encode calls."""
    def __init__(self,tokenizer):
        from threading import RLock
        self.tokenizer=tokenizer;self.lock=RLock()
    def __getattr__(self,name):return getattr(self.tokenizer,name)
    def __call__(self,*args,**kwargs):
        with self.lock:return self.tokenizer(*args,**kwargs)
    def pad(self,*args,**kwargs):
        with self.lock:return self.tokenizer.pad(*args,**kwargs)


class RemoteSampler:
    def __init__(self,broker,owner,cfg,*,prompt_ids):
        self.broker,self.owner,self.cfg,self.prompt_ids=broker,owner,cfg,prompt_ids
        self.cv=Condition();self.pending={};self.closed=False
        self.thread=Thread(target=self._receive,daemon=True);self.thread.start()

    def submit(self,request):
        future=Future()
        try:
            with self.cv:
                if self.closed:raise RuntimeError('REMOTE_SAMPLER_CLOSED')
                if request.request_id in self.pending:raise ValueError('REMOTE_REQUEST_ALREADY_PENDING')
                binding='ids:'+signature(self.prompt_ids(request))
                self.broker.submit(asdict(request),self.owner,prompt_binding=binding)
                self.pending[request.request_id]=future;self.cv.notify_all()
        except Exception as exc:future.set_exception(exc)
        return future

    def _receive(self):
        while True:
            with self.cv:
                while not self.pending and not self.closed:self.cv.wait()
                if self.closed and not self.pending:return
                ids=list(self.pending)
            try:
                values=self.broker.wait_many(self.owner,ids,.25)
                with self.cv:
                    for uid,value in values:self.pending.pop(uid).set_result(value)
            except BaseException as exc:
                with self.cv:
                    for future in self.pending.values():future.set_exception(exc)
                    self.pending.clear();self.closed=True;return

    def close(self):
        with self.cv:
            self.closed=True
            # Collection must reconcile outstanding draws before leaving.
            if self.pending:
                for future in self.pending.values():future.set_exception(RuntimeError('REMOTE_REQUEST_UNRECONCILED'))
                self.pending.clear()
            self.cv.notify_all()
        self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('REMOTE_RECEIVER_SHUTDOWN_TIMEOUT')

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def serve_gpu(broker,sampler,gpu):
    try:
        model=getattr(sampler,'model',None)
        if model is not None:
            device=next(model.parameters()).device
            if device.type=='cuda':
                # CUDA current-device is thread-local, unlike model placement.
                # Triton/FLA scratch allocations must use this replica's GPU.
                import torch
                torch.cuda.set_device(device)
        while True:
            batch=broker.take(gpu,sampler.cfg['generation_batch_size'],.02)
            if not batch:
                if broker.snapshot()['stopped']:return
                continue
            requests=[DecisionRequest(**row['request']) for row in batch]
            if hasattr(sampler,'submit'):
                # Native continuous engines keep one Future per request; do
                # not block refill on the longest member of an artificial batch.
                from concurrent.futures import wait,FIRST_COMPLETED
                active={sampler.submit(r):r for r in requests}
                while active:
                    done,_=wait(active,timeout=.01,return_when=FIRST_COMPLETED)
                    for future in done:
                        request=active.pop(future)
                        broker.complete(gpu,request.request_id,future.result())
                    extra=broker.take(gpu,sampler.cfg['generation_batch_size']-len(active),0.) if len(active)<sampler.cfg['generation_batch_size'] else []
                    for row in extra:
                        r=DecisionRequest(**row['request']);active[sampler.submit(r)]=r
            else:
                values=sampler.generate(requests)
                if len(values)!=len(requests):raise ValueError('GLOBAL_BACKEND_DROPPED_REQUEST')
                for request,value in zip(requests,values):broker.complete(gpu,request.request_id,value)
    except BaseException as exc:
        broker.fail(type(exc).__name__+':'+str(exc))
