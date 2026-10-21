import threading
import queue


def test_slow_cpu_tool_does_not_barrier_unrelated_ready_episode():
    from scripts.rsi_runtime import rolling_agent_batch
    blocked=threading.Event();unblock=threading.Event();observed=[]
    class Engine:
        def __init__(self):self.q=queue.Queue();self.i=0
        def submit(self,request):
            self.i+=1;observed.append(request);self.q.put((str(self.i),request));return str(self.i)
        def poll(self,timeout=0.):
            try:return self.q.get(timeout=timeout)
            except queue.Empty:return None
    def request(c):return {'id':c['id'],'step':c['step']}
    def consume(c,response):
        if c['id']=='slow':blocked.set();assert unblock.wait(3)
        if c['id']=='fast' and c['step']==1:
            assert blocked.wait(1);unblock.set()
        c['step']+=1;c['done']=c['step']==2;return c
    result=rolling_agent_batch([{'id':'slow','step':0},{'id':'fast','step':0}],request,consume,Engine(),cpu_workers=2)
    assert len(result)==2 and {'id':'fast','step':1} in observed
