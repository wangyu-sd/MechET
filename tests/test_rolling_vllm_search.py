"""Verify our public-engine adapter's refill and error lifecycle on CPU."""
from types import SimpleNamespace
import threading,time


class ControlledEngine:
    def __init__(self):self.active={};self.threads=set()
    def add_request(self,identifier,prompt,params,**kwargs):
        self.threads.add(threading.get_ident());self.active[identifier]=[prompt['prompt_token_ids'][0],0]
    def step(self):
        self.threads.add(threading.get_ident());time.sleep(.002)
        result=[]
        for key,value in list(self.active.items()):
            value[1]+=1
            if value[1]>=value[0]:
                result.append(SimpleNamespace(request_id=key,finished=True,outputs=[]));del self.active[key]
        return result
    def abort_request(self,identifiers):
        for key in identifiers:self.active.pop(key,None)


def test_short_request_refills_while_long_request_is_still_running():
    from scripts.rolling_vllm_search import ContinuousBatchEngine
    engine=ControlledEngine()
    with ContinuousBatchEngine(engine,params_factory=lambda r:r,lora=None,max_inflight=2) as pipe:
        long=pipe.submit({'prompt_ids':[300]});short=pipe.submit({'prompt_ids':[1]})
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            done=pipe.poll(.05)
            if done:break
        assert done[0]==short
        third=pipe.submit({'prompt_ids':[1]})
        while time.monotonic()<deadline:
            done=pipe.poll(.05)
            if done:break
        assert done[0]==third  # old blocking whole-wave generate would return long first/all together
        assert len(engine.threads)==1 and threading.get_ident() not in engine.threads
    assert not engine.active


def test_engine_error_is_propagated_without_silent_hang():
    from scripts.rolling_vllm_search import ContinuousBatchEngine
    import pytest
    class Broken(ControlledEngine):
        def step(self):raise RuntimeError('decoder failed')
    with ContinuousBatchEngine(Broken(),params_factory=lambda r:r,lora=None,max_inflight=2) as pipe:
        pipe.submit({'prompt_ids':[1]})
        with pytest.raises(RuntimeError,match='decoder failed'):
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:pipe.poll(.05)
