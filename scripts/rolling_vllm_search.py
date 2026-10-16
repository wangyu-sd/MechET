"""Single GPU-owner continuous add_request/step loop with rolling refill.

No concurrent calls into the synchronous vLLM engine. CPU tools/persistence run
in the caller or its pool while this dedicated thread continues GPU decoding.
"""
import queue,threading,uuid


class ContinuousBatchEngine:
    def __init__(self,engine,*,params_factory,lora,max_inflight=128):
        if max_inflight<1:raise ValueError('positive in-flight request capacity required')
        self.engine=engine;self.params_factory=params_factory;self.lora=lora;self.capacity=max_inflight
        self.inputs=queue.Queue();self.outputs=queue.Queue();self.stopping=threading.Event()
        self.thread=threading.Thread(target=self._run,name='vllm-gpu-owner',daemon=True)
        self.thread.start()

    def submit(self,request):
        identifier=uuid.uuid4().hex;self.inputs.put((identifier,request));return identifier

    def poll(self,timeout=0.):
        try:identifier,result,error=self.outputs.get(timeout=timeout)
        except queue.Empty:return None
        if error is not None:raise RuntimeError(str(error)) from error
        return identifier,result

    def _run(self):
        active=set()
        try:
            while not self.stopping.is_set():
                while len(active)<self.capacity:
                    try:identifier,request=self.inputs.get(timeout=.01 if not active else 0.)
                    except queue.Empty:break
                    self.engine.add_request(identifier,{'prompt_token_ids':request['prompt_ids']},
                        self.params_factory(request),lora_request=self.lora)
                    active.add(identifier)
                if active:
                    for output in self.engine.step():
                        if output.finished:
                            active.remove(str(output.request_id))
                            self.outputs.put((str(output.request_id),output,None))
        except BaseException as error:self.outputs.put((None,None,error))
        finally:
            if active:
                try:self.engine.abort_request(list(active))
                except Exception:pass

    def close(self):
        self.stopping.set();self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('GPU owner did not stop')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
